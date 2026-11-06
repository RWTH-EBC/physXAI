"""
Loss and metrics of physics-informed training.

The loss of a model with one output y and physics targets p_1, ..., p_n (e.g. the prediction of a
physical model of the same quantity) is

    L = mse(y_hat, y) / s + sum_i w_i * mse(y_hat, p_i) / s

with the characteristic scale s = var(y_train) (`normalize_terms=True`, else s = 1). The scale makes the
terms dimensionless: the data term is 1 - R^2 on the training data and w_i is the relative importance of
physics target i. The terms are not averaged, so adding a physics term does not change the weight of the
data term. The weights encode the trust in the physics and are chosen by validation (e.g. the error over
the prediction horizon); they are not adapted during the training.

The targets passed to keras are `[y, p_1, ..., p_n]` (columns), the model predicts y only, so the trained
model can be used (and exported to AgentLib-MPC) like any other single output model.
"""

from __future__ import annotations

import os
import re

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
from keras import ops  # noqa: E402


@keras.saving.register_keras_serializable(package="physXAI", name="PhysicsLoss")
class PhysicsLoss(keras.losses.Loss):
    """
    Data loss plus weighted physics terms, all divided by the characteristic scale.

    Args:
        weights: Weight of each physics target (columns 1, 2, ... of y_true).
        scale: Characteristic scale (variance of the training targets).
    """

    def __init__(self, weights: list[float], scale: float = 1.0, name: str = "physics_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = [float(w) for w in weights]
        self.scale = float(scale)

    def call(self, y_true, y_pred):
        prediction = y_pred[:, 0]
        loss = ops.square(prediction - y_true[:, 0])
        for i, weight in enumerate(self.weights):
            loss = loss + weight * ops.square(prediction - y_true[:, i + 1])
        return loss / self.scale

    def get_config(self) -> dict:
        return {**super().get_config(), "weights": self.weights, "scale": self.scale}


@keras.saving.register_keras_serializable(package="physXAI", name="TermLoss")
class TermLoss(keras.metrics.Mean):
    """
    One unweighted term of the loss: mean of (y_hat - y_true[:, column])^2 / scale.
    Column 0 is the data term ('data_loss'), columns 1, ... the physics targets.
    """

    def __init__(self, column: int, scale: float = 1.0, name: str = "data_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.column = int(column)
        self.scale = float(scale)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.square(y_pred[:, 0] - y_true[:, self.column]) / self.scale
        return super().update_state(values, sample_weight=sample_weight)

    def get_config(self) -> dict:
        return {**super().get_config(), "column": self.column, "scale": self.scale}


@keras.saving.register_keras_serializable(package="physXAI", name="OutputRMSE")
class OutputRMSE(keras.metrics.RootMeanSquaredError):
    """RMSE of the prediction with respect to the data (column 0 of the targets), in output units."""

    def __init__(self, name: str = "rmse", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], y_pred[:, :1], sample_weight=sample_weight)


def physics_metrics(terms: list[str], scale: float) -> list[keras.metrics.Metric]:
    """Metrics of all terms (`data_loss`, `<term>_loss`) and the RMSE of the data."""
    metrics = [TermLoss(0, scale, name="data_loss")]
    metrics += [TermLoss(i + 1, scale, name=f"{term}_loss") for i, term in enumerate(terms)]
    return [*metrics, OutputRMSE()]


# --- trainable physics residuals -------------------------------------------------------------------
class PhysicsResidual:
    """
    A physics residual r(x, y; theta) to be close to zero, e.g. the disagreement of the prediction with a
    grey-box model with trainable physical parameters theta (`px.Param`, `px.positive`):

        dT = (1 - px.exp(-dt / tau)) * (v.T_amb - v.T + K_u * v.u_hp)       # exact zero order hold
        residual = PhysicsResidual(px.Prediction() - dT, weight=1.0)

    `px.Prediction()` is the output of the model. All other variables are data columns at the time of the
    inputs, `px.Target(name)` at the predicted time; they become loss columns of the preprocessing and do
    not have to be inputs of the model. Only the network is saved and exported, the residual is used in the
    training only.

    Args:
        residual: The residual expression. It has to contain `px.Prediction()`; parameters (`px.Param`) are
                  the only trainable parts (no blocks), so they keep their physical meaning.
        weight: Weight of the residual in the loss (relative to the data term).
        name: Name of the term, in lower case (history: '<name>_loss', '<name>_fit_loss').
        scale: Characteristic squared magnitude of the residual (the loss term is mean(r^2) / scale).
               Defaults to the variance of the training targets, i.e. a residual in output units.
    """

    def __init__(self, residual, weight: float = 1.0, name: str = "physics", scale: float | None = None):
        from physXAI.expr import Expr, Param, Prediction, from_config

        if isinstance(residual, dict):
            residual = from_config(residual)
        if not isinstance(residual, Expr):
            raise TypeError("The residual has to be an expression (physXAI.expr).")
        nodes = residual.nodes()
        if not any(isinstance(node, Prediction) for node in nodes):
            raise ValueError("The residual has to contain px.Prediction(), the output of the model.")
        blocks = [node.name for node in nodes if node.has_weights and not isinstance(node, Param)]
        if blocks:
            raise ValueError(f"Physics residuals can only train parameters (px.Param), not blocks: {blocks}.")
        if weight < 0:
            raise ValueError("The weight of a physics residual must not be negative.")
        self.residual = residual
        self.weight = float(weight)
        # keras writes metric names in snake case, the history keys are '<name>_loss' and '<name>_fit_loss'
        self.name = re.sub(r"[^0-9a-z_]", "_", str(name).lower())
        self.scale = None if scale is None else float(scale)

    @property
    def columns(self) -> list[str]:
        """Data columns the residual needs (loss columns of the preprocessing)."""
        return self.residual.free_vars()

    def parameters(self) -> list:
        """The parameters (`px.Param`) of the residual."""
        from physXAI.expr import Param

        return [node for node in self.residual.nodes() if isinstance(node, Param)]

    def named_values(self) -> list:
        """Named sub expressions of parameters only (e.g. `px.positive(..., 'tau')`), reported after training."""
        from physXAI.expr import Param, Prediction, Var

        named = list()
        for node in self.residual.nodes():
            if node._name is None or isinstance(node, Param):
                continue
            sub = node.nodes()
            if any(isinstance(n, Param) for n in sub) and not any(isinstance(n, (Var, Prediction)) for n in sub):
                named.append(node)
        return named

    def get_config(self) -> dict:
        from physXAI.expr import to_config

        return {"residual": to_config(self.residual), "weight": self.weight, "name": self.name, "scale": self.scale}

    @classmethod
    def from_config(cls, config: dict) -> PhysicsResidual:
        return cls(**config)


def _residual_loss(scale_y: float, scale_r: float, weight: float, fit_weight: float, data_weight: float = 1.0):
    def loss(y_true, outputs):
        data = ops.mean(ops.square(outputs[:, 0] - y_true[:, 0])) / scale_y
        physics = ops.mean(ops.square(outputs[:, 1])) / scale_r
        fit = ops.mean(ops.square(outputs[:, 2])) / scale_r
        return data_weight * data + weight * physics + fit_weight * fit

    return loss


def _term_metrics(scale_y: float, scale_r: float, name: str) -> list:
    def data_loss(y_true, outputs):
        return ops.mean(ops.square(outputs[:, 0] - y_true[:, 0])) / scale_y

    def physics_loss(y_true, outputs):
        return ops.mean(ops.square(outputs[:, 1])) / scale_r

    def physics_fit_loss(y_true, outputs):
        return ops.mean(ops.square(outputs[:, 2])) / scale_r

    def rmse(y_true, outputs):
        return ops.sqrt(ops.mean(ops.square(outputs[:, 0] - y_true[:, 0])))

    physics_loss.__name__ = f"{name}_loss"
    physics_fit_loss.__name__ = f"{name}_fit_loss"
    return [data_loss, physics_loss, physics_fit_loss, rmse]


def fit_with_physics(model, td, settings, physics: PhysicsResidual, mode: str = "calibrate",
                     physics_learning_rate: float | None = None) -> dict:
    """
    Trains a single output keras model on the data and a physics residual.

    The residual is evaluated twice with shared parameters theta: with the prediction (r_hat, the physics
    term of the network) and with the measured target (r_y, the fit of the physics model to the data).

    - 'calibrate' (default): 1. theta is fitted to the data on mean(r_y^2) (grey-box identification, the network
      is frozen), 2. theta is frozen and the network is trained on the data and w * mean(r_hat^2). The
      parameters cannot drift to follow the network.
    - 'joint': network and theta together on data + w * r_hat + r_y.
    - 'fixed': theta stays at its initial values, the network is trained on data + w * r_hat.

    Args:
        model: The keras model (one output). It is trained in place.
        td: Single step training data holding the loss columns of the residual.
        settings: The physXAI model (epochs, batch_size, learning_rate, early_stopping_epochs, random_seed).
        physics: The residual.
        mode: 'calibrate', 'joint' or 'fixed'.
        physics_learning_rate: Learning rate of the calibration. Defaults to 10 x the learning rate.

    Returns:
        dict: 'parameters' (trained values by name), 'fit' (RMSE of the physics term and of the physics model
              per split), 'calibration_record' (history of the calibration or None) and 'record' (keras
              history of the training of the network).
    """
    import numpy as np
    import tensorflow as tf

    from physXAI.expr import BuildContext, freeze
    from physXAI.io.run import keras_verbosity, keras_verbosity_int

    if mode not in ("calibrate", "joint", "fixed"):
        raise ValueError(f"Unknown physics_fit '{mode}', use 'calibrate', 'joint' or 'fixed'.")
    columns = physics.columns
    n_inputs, n_columns = len(td.inputs), len(columns)

    x_in = keras.Input(shape=(n_inputs,), name="inputs")
    columns_in = keras.Input(shape=(n_columns,), name="loss_columns")
    y_in = keras.Input(shape=(1,), name="target")
    prediction = model(x_in)
    shared: dict = dict()
    r_hat = BuildContext(columns_in, columns, prediction=prediction, layers=shared).tensor(physics.residual)
    r_y = BuildContext(columns_in, columns, prediction=y_in, layers=shared).tensor(physics.residual)
    wrapper = keras.Model([x_in, columns_in, y_in], keras.layers.Concatenate()([prediction, r_hat, r_y]))
    parameter_layers = [shared[id(p)] for p in physics.parameters() if id(p) in shared]

    scale_y = max(float(np.var(td.y_train_single[:, 0])), 1e-12)
    scale_r = physics.scale if physics.scale is not None else scale_y

    def data(split):
        if not td.has_split(split):
            return None
        y = getattr(td, f"y_{split}_single")[:, :1]
        return (getattr(td, f"X_{split}_single"), td.values(split, columns), y), y

    train, val = data("train"), data("val")

    def dataset(d, shuffle):
        ds = tf.data.Dataset.from_tensor_slices(d)
        if shuffle:
            ds = ds.shuffle(min(10000, len(d[1])), seed=settings.random_seed, reshuffle_each_iteration=True)
        return ds.batch(settings.batch_size).prefetch(tf.data.AUTOTUNE)

    def fit(loss, learning_rate, monitor):
        wrapper.compile(keras.optimizers.Adam(learning_rate=learning_rate), loss=loss,
                        metrics=_term_metrics(scale_y, scale_r, physics.name))
        callbacks = list()
        if settings.early_stopping_epochs is not None and val is not None:
            callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, mode="min",
                                                           patience=settings.early_stopping_epochs,
                                                           restore_best_weights=True,
                                                           verbose=keras_verbosity_int()))
        return wrapper.fit(dataset(train, True), validation_data=dataset(val, False) if val is not None else None,
                           epochs=settings.epochs, callbacks=callbacks, verbose=keras_verbosity(), shuffle=False)

    calibration = None
    if mode == "calibrate" and parameter_layers:
        model.trainable = False
        calibration = fit(_residual_loss(scale_y, scale_r, 0.0, 1.0, data_weight=0.0),
                          physics_learning_rate or 10 * settings.learning_rate, f"val_{physics.name}_fit_loss")
        model.trainable = True
    if mode in ("calibrate", "fixed"):
        for layer in parameter_layers:
            layer.trainable = False
        record = fit(_residual_loss(scale_y, scale_r, physics.weight, 0.0), settings.learning_rate, "val_data_loss")
    else:
        record = fit(_residual_loss(scale_y, scale_r, physics.weight, 1.0), settings.learning_rate, "val_data_loss")

    # trained parameters (also the named ones, e.g. px.positive) and the fit of the physics model
    values = {p.name: float(np.asarray(shared[id(p)].get_weights()[0]).reshape(-1)[0])
              for p in physics.parameters() if id(p) in shared}
    parameters = dict(values)
    for node in physics.named_values():
        parameters[node.name] = float(np.asarray(freeze(node, values).evaluate({})).reshape(-1)[0])
    fit_metrics = dict()
    for split in ("train", "val", "test"):
        d = data(split)
        if d is None:
            continue
        outputs = wrapper.predict(d[0], verbose=0, batch_size=1024)
        fit_metrics[split] = {
            "physics_rmse": float(np.sqrt(np.mean(outputs[:, 1] ** 2))),
            "fit_rmse": float(np.sqrt(np.mean(outputs[:, 2] ** 2))),
        }
    return {
        "name": physics.name,
        "parameters": parameters,
        "fit": fit_metrics,
        "calibration_record": calibration.history if calibration is not None else None,
        "record": record,
    }
