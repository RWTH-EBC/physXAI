"""
Recursive (multi step) prediction with single step models.

A single step model predicts the next value (or the change) of a feature it also gets as input.
In an MPC it is applied recursively over the horizon: the prediction of step t becomes the input
of step t+1. This module performs exactly this rollout on windows of the measured data:

- The recursive feature (e.g. 'T' for an output 'Change(T)') and its lags ('T_lag1', ...) are
  taken from the predicted trajectory, all other inputs from the data (they are known to the MPC).
- Inputs holding values of the recursive feature are recomputed in every step from the predicted
  trajectory: its changes ('Change(T)'), derived data columns depending on it (e.g. a heat flow computed
  from 'T'), their lags ('Q_lag1') and changes ('Change(Q)'). Expressions inside the model are recomputed
  anyway, since they are part of the keras model.
- The rollout is a keras RNN around the (trained) step model. It is differentiable, so it can also
  be used to train a single step model on the multi step error (`physXAI.training.recursive`).

The resulting errors over the horizon (RMSE, MAE and bias per step) show how the error of the
model grows in an MPC, which the single step error does not.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from physXAI.expr.core import BuildContext
from physXAI.preprocessing.preprocessing import CHANGE_PATTERN, LAG_PATTERN, depends_on, output_feature
from physXAI.preprocessing.splitting import SPLITS
from physXAI.preprocessing.training_data import TrainingData

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402

logger = logging.getLogger("physXAI")


def _base_and_lag(name: str) -> tuple[str, int]:
    match = LAG_PATTERN.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


@dataclass
class RecursionSpec:
    """
    Describes how a single step model is applied recursively.

    Attributes:
        feature: The recursive feature, e.g. 'T'.
        difference: Whether the model predicts the change of the feature ('Change(T)').
        max_lag: Largest lag of the feature the model needs (inputs and derived columns).
        columns: Columns of one step of the rollout: the inputs of the model plus the variables of
                 derived inputs which depend on the feature.
        derived: Inputs that are derived columns depending on the feature (recomputed every step).
        expressions: Expression of every recomputed input in terms of `columns` (changes of the feature
                     'Change(T)' = T - T_lag1, derived columns, their lags and changes).
    """

    feature: str
    difference: bool
    max_lag: int
    columns: list[str]
    derived: list[str]
    expressions: dict = field(default_factory=dict)

    @property
    def history(self) -> list[str]:
        """Names of the state of the rollout: [feature, feature_lag1, ..., feature_lag<max_lag>]."""
        return [self.feature] + [f"{self.feature}_lag{k}" for k in range(1, self.max_lag + 1)]


def recursion_spec(td: TrainingData) -> RecursionSpec | None:
    """
    Derives the recursion of a single step model from the names of its inputs and outputs.

    Returns:
        RecursionSpec, or None if the model does not predict one of its own inputs (then there is
        nothing to feed back and the single step error already is the multi step error) or the targets
        are not the next step (shift != 1).
    """
    if len(td.outputs) != 1 or td.shift != 1:
        return None
    feature = output_feature(td.outputs[0])
    difference = CHANGE_PATTERN.match(td.outputs[0]) is not None

    # inputs holding values of the feature, other than the feature and its lags, are recomputed in every step from
    # the predicted history: its changes ('Change(T)' = T - T_lag1), derived columns of it, their lags ('Q_lag1') and
    # changes ('Change(Q)'), else the rollout would take them from the data (leakage of the measured trajectory)
    expressions = dict()
    for name in td.inputs:
        if _base_and_lag(name)[0] != feature and depends_on(name, {feature}, td.derived):
            expressions[name] = _in_history(name, feature, td.derived)
    columns = list(td.inputs)
    for expression in expressions.values():
        columns.extend(v for v in expression.free_vars() if v not in columns)
    lags = [_base_and_lag(c)[1] for c in columns if _base_and_lag(c)[0] == feature]
    if not lags:
        return None
    if feature not in td.all_columns:
        logger.info("Recursive evaluation skipped: the data holds no column '%s'.", feature)
        return None
    # lags of the feature come from the predicted history, lags of other columns from the data shifted by the lag
    # (e.g. 'mDot_lag1' in the history of 'Q_lag1'), they do not have to be columns of the data
    missing = [c for c in columns if c not in td.all_columns and _base_and_lag(c)[0] != feature
               and _base_and_lag(c)[0] not in td.all_columns]
    if missing:
        logger.info("Recursive evaluation skipped: the data holds no columns %s.", missing)
        return None
    derived = [c for c in td.inputs if c in td.derived and c in expressions]
    return RecursionSpec(feature, difference, max(lags), columns, derived, expressions)


def _in_history(name: str, feature: str, derived: dict):
    """
    A column written in the history of the rollout: the feature and its lags are history slots, changes become
    differences ('Change(T)_lag2' -> T_lag2 - T_lag3), derived columns their expressions ('Q_lag1' -> the expression
    of Q one step in the past), recursively. Columns independent of the feature stay data columns.
    """
    from physXAI.expr import Var

    base, lag = _base_and_lag(name)
    if base == feature or not depends_on(name, {feature}, derived):
        return Var(name)
    change = CHANGE_PATTERN.match(base)
    if change is not None:
        inner = _in_history(change.group(1).strip(), feature, derived)
        expression = inner - inner.lag(1)
    else:
        mapping = {v: _in_history(v, feature, derived) for v in derived[base].free_vars()}
        expression = derived[base].substitute(mapping)
    return expression.lag(lag)


def as_keras_step_model(model, n_inputs: int) -> keras.Model:
    """The model as keras model; scikit-learn linear regressions become a frozen dense layer."""
    if isinstance(model, keras.Model):
        return model
    from sklearn.linear_model import LinearRegression

    if isinstance(model, LinearRegression):
        inp = keras.Input(shape=(n_inputs,))
        dense = keras.layers.Dense(1, trainable=False)
        out = dense(inp)
        dense.set_weights(
            [
                np.asarray(model.coef_, dtype=np.float32).reshape(-1, 1),
                np.asarray(model.intercept_, dtype=np.float32).reshape(-1),
            ]
        )
        return keras.Model(inp, out)
    raise TypeError(f"Recursive predictions are not supported for {type(model).__name__}.")


def build_step_function(step_model: keras.Model, td: TrainingData, spec: RecursionSpec) -> keras.Model:
    """
    Maps one row of `spec.columns` to the output of the model, recomputing derived inputs.
    """
    row = keras.Input(shape=(len(spec.columns),))
    ctx = BuildContext(row, spec.columns)
    parts = [ctx.tensor(spec.expressions[c]) if c in spec.expressions else ctx.tensor(_var(c)) for c in td.inputs]
    x = parts[0] if len(parts) == 1 else keras.layers.Concatenate()(parts)
    return keras.Model(row, step_model(x), name="step_function")


def _var(column: str):
    from physXAI.expr import Var

    return Var(column)


class RecursiveCell(keras.layers.Layer):
    """
    RNN cell applying a step function recursively.

    State: history of the recursive feature [x_t, x_{t-1}, ..., x_{t-max_lag}].
    Input:  one data row of `spec.columns` at time t.
    Output: the predicted absolute value x_{t+1}.
    """

    def __init__(self, step_function: keras.Model, slots: np.ndarray, difference: bool, **kwargs):
        super().__init__(**kwargs)
        self.step_function = step_function
        # slots[k, j] = 1 if column j holds the feature with lag k
        self.slots = np.asarray(slots, dtype="float32")
        self.mask = self.slots.sum(axis=0, keepdims=True)
        self.difference = difference
        self.state_size = self.slots.shape[0]
        self.output_size = 1

    def call(self, inputs, states):
        history = states[0] if isinstance(states, (list, tuple)) else states
        row = inputs * (1.0 - self.mask) + keras.ops.matmul(history, self.slots)
        y = self.step_function(row)
        value = history[:, :1] + y if self.difference else y
        history = keras.ops.concatenate([value, history[:, :-1]], axis=-1) if self.state_size > 1 else value
        return value, [history]


def build_rollout_model(step_model, td: TrainingData, spec: RecursionSpec, horizon: int) -> keras.Model:
    """
    Keras model [rows (batch, horizon, columns), initial history (batch, max_lag + 1)] ->
    predicted feature (batch, horizon, 1). It shares the weights of `step_model`.
    """
    step_model = as_keras_step_model(step_model, len(td.inputs))
    step_function = build_step_function(step_model, td, spec)
    slots = np.zeros((spec.max_lag + 1, len(spec.columns)), dtype="float32")
    for j, column in enumerate(spec.columns):
        base, lag = _base_and_lag(column)
        if base == spec.feature:
            slots[lag, j] = 1.0
    rows = keras.Input(shape=(horizon, len(spec.columns)))
    history = keras.Input(shape=(spec.max_lag + 1,))
    cell = RecursiveCell(step_function, slots, spec.difference)
    out = keras.layers.RNN(cell, return_sequences=True)(rows, initial_state=[history])
    return keras.Model([rows, history], out, name="rollout")


def rollout_data(td: TrainingData, spec: RecursionSpec, split: str, horizon: int, stride: int = 1):
    """
    Windows for a recursive prediction over `horizon` steps.

    Returns:
        tuple: rows (n, horizon, columns), initial history (n, max_lag + 1) and the true trajectory of
               the feature (n, horizon, 1), i.e. x_{s+1}, ..., x_{s+horizon} for a window starting at s.
               None if the split holds no window.
    """
    starts = td.window_starts(split, horizon, stride)
    # lagged data columns which are no columns of the data are the data shifted by their lag
    shifted = {c: _base_and_lag(c) for c in spec.columns
               if c not in td.all_columns and _base_and_lag(c)[0] != spec.feature}
    starts = starts[starts >= max([spec.max_lag, *(lag for _, lag in shifted.values())])]
    feature = td.column(spec.feature)
    history = np.stack([feature[starts - k] for k in range(spec.max_lag + 1)], axis=1)
    keep = np.all(np.isfinite(history), axis=1)
    starts, history = starts[keep], history[keep]
    if len(starts) == 0:
        return None
    window = starts[:, None] + np.arange(horizon)[None, :]
    # columns of the feature (and its lags) are replaced by the predicted history in the rollout
    rows = np.zeros((len(starts), horizon, len(spec.columns)))
    for j, column in enumerate(spec.columns):
        if column in td.all_columns:
            rows[:, :, j] = td.column(column)[window]
        elif column in shifted:
            base, lag = shifted[column]
            rows[:, :, j] = td.column(base)[window - lag]
    target = td.column(td.targets[0])[window]
    truth = history[:, :1] + np.cumsum(target, axis=1) if spec.difference else target
    return rows.astype("float32"), history.astype("float32"), truth[..., None].astype("float32")


class HorizonMetrics:
    """
    Errors of a multi step prediction per step of the horizon: RMSE, MAE and bias (mean of
    prediction - truth) for every split.

    Args:
        predictions: {split: (prediction, truth)}, arrays of shape (windows, horizon, features).
        time_step: Sampling time in seconds (for an axis in hours).
        feature: Name of the predicted feature, or a list of names for several features.

    Attributes:
        kpis: {split: {'rmse': [...], 'mae': [...], 'bias': [...], 'std': [...], 'n_windows': n}} of the
              first feature ('std' is the standard deviation of the error per step).
        by_feature: The same for every feature, {feature: kpis}.
    """

    def __init__(
        self,
        predictions: dict[str, tuple[np.ndarray, np.ndarray]],
        time_step: float | None = None,
        feature: str | list[str] = "",
    ):
        features = [feature] if isinstance(feature, str) else list(feature)
        self.features = features
        self.feature = features[0]
        self.time_step = time_step
        self.by_feature: dict[str, dict[str, dict]] = {f: dict() for f in features}
        for split, (prediction, truth) in predictions.items():
            errors = prediction - truth
            for j, name in enumerate(features):
                error = errors[..., j]
                self.by_feature[name][split] = {
                    "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                    "mae": np.mean(np.abs(error), axis=0).tolist(),
                    "bias": np.mean(error, axis=0).tolist(),
                    "std": np.std(error, axis=0).tolist(),
                    "n_windows": int(error.shape[0]),
                }
        self.kpis: dict[str, dict] = self.by_feature[self.feature]
        for name, splits in self.by_feature.items():
            for split, kpis in splits.items():
                logger.info(
                    "%s horizon RMSE of %s (first / last step): %.4f / %.4f, bias: %.4f / %.4f",
                    split.capitalize(),
                    name,
                    kpis["rmse"][0],
                    kpis["rmse"][-1],
                    kpis["bias"][0],
                    kpis["bias"][-1],
                )

    @property
    def steps(self) -> np.ndarray:
        """Prediction steps 1, ..., horizon."""
        first = next(iter(self.kpis.values()))
        return np.arange(1, len(first["rmse"]) + 1)

    @property
    def hours(self) -> np.ndarray | None:
        """Prediction horizon in hours, if the time step is known."""
        return self.steps * self.time_step / 3600 if self.time_step else None

    def get_config(self) -> dict:
        config = {"feature": self.feature, "time_step": self.time_step, **self.kpis}
        if len(self.features) > 1:
            config["by_feature"] = self.by_feature
        return config


def evaluate_horizon(
    model,
    td: TrainingData,
    horizon: int | None = None,
    stride: int | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> HorizonMetrics | None:
    """
    Predicts all windows of the data recursively and stores predictions and metrics in `td`.

    Args:
        model: Trained single step model (keras or scikit-learn linear regression).
        td: Training data. Defaults for `horizon` and `stride` are taken from it.
        horizon: Number of steps.
        stride: Distance between two windows.

    Returns:
        HorizonMetrics, or None if the model has no recursion or no horizon is given.
    """
    horizon = horizon or td.eval_horizon
    stride = stride or td.eval_stride or 1
    spec = recursion_spec(td)
    if horizon is None or spec is None:
        return None
    rollout = build_rollout_model(model, td, spec, horizon)
    predictions = dict()
    for split in splits:
        if split not in SPLITS:
            continue
        data = rollout_data(td, spec, split, horizon, stride)
        if data is None:
            continue
        rows, history, truth = data
        prediction = rollout.predict([rows, history], verbose=0, batch_size=1024)
        predictions[split] = (prediction, truth)
    if not predictions:
        logger.warning("Recursive evaluation skipped: no window of %d steps fits into the splits.", horizon)
        return None
    td.horizon_predictions = predictions
    td.horizon_metrics = HorizonMetrics(predictions, td.time_step, spec.feature)
    return td.horizon_metrics
