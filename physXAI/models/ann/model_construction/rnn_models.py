"""
Construction of recurrent (multi step) models.

Two keras models are built:

- The **step model** `rnn_step` maps `[sequence, *states] -> [prediction, *states]` for a sequence of
  any length. It is the model that is saved and used in AgentLib-MPC, which applies it one step at a
  time and warms its states up on the measured past.
- The **sequence model** wraps the step model for training and evaluation. It starts from zero
  states, warms up on the first `warmup` steps of a sequence (not part of the loss) and predicts the
  following `horizon` steps:

  - An output 'Change(T)' is accumulated to the trajectory of T, starting from the measured value
    T(t0), so the model can be trained on the trajectory.
  - If a predicted feature is also an input of the model, it is fed back during the prediction
    (closed loop), as in an MPC. During the warmup the measured values are used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from physXAI.expr import Expr
from physXAI.expr.core import BuildContext
from physXAI.models.ann.configs.ann_model_configs import RNNModelConstruction_config

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402

STEP_MODEL_NAME = "rnn_step"
RNN_LAYERS = {"RNN": keras.layers.SimpleRNN, "GRU": keras.layers.GRU, "LSTM": keras.layers.LSTM}


def number_of_states(rnn_layer: str) -> int:
    """Number of state tensors a recurrent layer carries between two time steps (2 for an LSTM)."""
    return 2 if rnn_layer == "LSTM" else 1


@dataclass
class SequenceSpec:
    """
    Describes the in- and outputs of a recurrent model.

    Attributes:
        n_inputs: Number of input features.
        difference: Per output, whether it is the change of its feature.
        feedback: Pairs (input index, output index) of predicted features which are also inputs.
        absolute: Whether the start values of the features are known (trajectories are absolute).
    """

    n_inputs: int
    difference: list[bool]
    feedback: list[tuple[int, int]]
    absolute: bool

    @property
    def n_outputs(self) -> int:
        return len(self.difference)

    @classmethod
    def from_training_data(cls, td) -> SequenceSpec:
        feedback = [(td.inputs.index(f), j) for j, f in enumerate(td.features) if f in td.inputs]
        return cls(len(td.inputs), list(td.difference), feedback, bool(td.absolute))


def build_step_model(
    config: dict, td, feature_architecture: list[Expr] | None = None, name: str = STEP_MODEL_NAME
) -> keras.Model:
    """
    Builds the step model `[sequence, *states] -> [prediction, *states]`.

    Args:
        config: Settings, validated by `RNNModelConstruction_config` (rnn_units, rnn_layer).
        td: `TrainingDataMultiStep`, used for the normalization of in- and outputs.
        feature_architecture: Expressions (`physXAI.expr`) building the inputs of the recurrent layer
                              out of the input features. Defaults to the normalized inputs.
        name: Name of the model.
    """
    config = RNNModelConstruction_config.model_validate(config).model_dump()
    units, rnn_layer = config["rnn_units"], config["rnn_layer"]
    n_inputs, n_outputs = len(td.inputs), len(td.outputs)
    X = td.X_train_single
    targets = td.targets("train").reshape(-1, n_outputs)

    # AgentLib-MPC recognizes input layers by 'input' in their name
    sequence = keras.Input(shape=(None, n_inputs), name="sequence_input")
    states = [keras.Input(shape=(units,), name=f"state_input_{i}") for i in range(number_of_states(rnn_layer))]

    if feature_architecture is None:
        normalization = keras.layers.Normalization(mean=X.mean(axis=0), variance=np.maximum(X.var(axis=0), 1e-12))
        features = normalization(sequence)
    else:
        # the expressions slice the features they need out of the last axis, so they work on
        # sequences as well as on single time steps
        ctx = BuildContext(sequence, td.inputs, {c: X[:, i] for i, c in enumerate(td.inputs)})
        channels = [ctx.tensor(expression) for expression in feature_architecture]
        features = channels[0] if len(channels) == 1 else keras.layers.Concatenate()(channels)

    rnn = RNN_LAYERS[rnn_layer](units, return_sequences=True, return_state=True)
    hidden, *new_states = rnn(features, initial_state=states)
    prediction = keras.layers.Dense(n_outputs)(hidden)
    # plain floats: a Rescaling layer holding tensors cannot be serialized
    scale = [float(v) for v in np.maximum(targets.std(axis=0), 1e-12)]
    offset = [float(v) for v in targets.mean(axis=0)]
    prediction = keras.layers.Rescaling(
        scale=scale if n_outputs > 1 else scale[0], offset=offset if n_outputs > 1 else offset[0]
    )(prediction)
    return keras.Model([sequence, *states], [prediction, *new_states], name=name)


#: Kinds of terms of the physics-inspired module of a PCNN and the sign of their effect
PCNN_TERMS = {"heating": 1.0, "cooling": -1.0, "loss": 1.0}
#: Name of the dense layer holding the (relative) physical parameters of a PCNN
PCNN_PHYSICS_LAYER = "dense_pcnn_physics"


@dataclass
class PCNNTerm:
    """
    One term of the energy accumulator of a PCNN.

    Attributes:
        kind: 'heating' (+ s * u), 'cooling' (- s * u) or 'loss' (+ s * (T_x - T), e.g. the outside temperature).
        input: Index of the input feature in the sequence.
        initial: Initial value of the parameter s per time step (> 0). The trained parameter is
                 `initial * weight`, the weight starts at 1 (s = s0 * s~ as in Di Natale et al. 2022).
    """

    kind: str
    input: int
    initial: float


def build_pcnn_step_model(config: dict, td, terms: list[PCNNTerm], name: str = STEP_MODEL_NAME) -> keras.Model:
    """
    Builds the step model of a physically consistent neural network (PCNN, Di Natale et al. 2022,
    Applied Energy 325, 119806), `[sequence, E, *states of f] -> [change of T, E, *states of f]`.

    The predicted feature T (input and output `Change(T)`) is the sum of the unforced dynamics D and the
    energy accumulator E:

        E(k+1) = E(k) + sum_heating a * u(k) - sum_cooling d * u(k) + sum_losses b * (T_x(k) - T(k))
        D(k+1) = D(k) + f(D(k), x(k)),   D(k) = T(k) - E(k)

    f is a recurrent network on D and the remaining inputs x, which excludes the controls and the
    temperatures of the losses. With a, b, d > 0 and 1 - sum b > 0, T increases with heating, decreases
    with cooling and increases with the temperatures T_x over the whole horizon, for any f. The parameters
    work in physical units, f on normalized values.

    Only keras layers translated by AgentLib-MPC are used (input slices, constants, subtract, multiply,
    concatenate, dense, simple_rnn, normalization, rescaling, add), the layer names carry their types.

    Args:
        config: Settings, validated by `RNNModelConstruction_config` (rnn_units, rnn_layer of f).
        td: `TrainingDataMultiStep` with one output `Change(T)` and T as input.
        terms: The terms of the energy accumulator.
        name: Name of the model.
    """
    from physXAI.models.ann.keras_models.keras_models import ConstantLayer, InputSliceLayer

    config = RNNModelConstruction_config.model_validate(config).model_dump()
    units, rnn_layer = config["rnn_units"], config["rnn_layer"]
    n_inputs = len(td.inputs)
    X = td.X_train_single
    feature = td.inputs.index(td.features[0])
    change = td.targets("train").reshape(-1)

    sequence = keras.Input(shape=(None, n_inputs), name="sequence_input")
    energy_state = keras.Input(shape=(1,), name="state_input_0")
    states = [keras.Input(shape=(units,), name=f"state_input_{i + 1}") for i in range(number_of_states(rnn_layer))]

    # --- physics-inspired module: increment of the energy accumulator in physical units -----------
    temperature = InputSliceLayer([feature], name="input_slice_pcnn_feature")(sequence)
    scaled = list()
    for j, term in enumerate(terms):
        column = InputSliceLayer([term.input], name=f"input_slice_pcnn_term_{j}")(sequence)
        if term.kind == "loss":
            column = keras.layers.Subtract(name=f"subtract_pcnn_gradient_{j}")([column, temperature])
        initial = keras.layers.Multiply(name=f"multiply_pcnn_scale_{j}")
        constant = ConstantLayer(value=PCNN_TERMS[term.kind] * term.initial, shape=(1,), name=f"constant_pcnn_scale_{j}")
        scaled.append(initial([column, constant(column)]))
    physics = scaled[0] if len(scaled) == 1 else keras.layers.Concatenate(name="concatenate_pcnn_physics")(scaled)
    increment = keras.layers.Dense(
        1, use_bias=False, kernel_initializer="ones", kernel_constraint=keras.constraints.NonNeg(),
        name=PCNN_PHYSICS_LAYER,
    )(physics)
    # E(k+1) = E(k) + increment(k): a linear recurrent layer with fixed weights 1
    accumulator = keras.layers.SimpleRNN(
        1, activation="linear", use_bias=False, kernel_initializer="ones", recurrent_initializer="ones",
        return_sequences=True, return_state=True, trainable=False, name="simple_rnn_pcnn_energy",
    )
    energy_next, new_energy_state = accumulator(increment, initial_state=[energy_state])
    energy = keras.layers.Subtract(name="subtract_pcnn_energy")([energy_next, increment])
    unforced = keras.layers.Subtract(name="subtract_pcnn_unforced")([temperature, energy])

    # --- black-box module f(D, x) ----------------------------------------------------------------
    used = {feature, *(term.input for term in terms)}
    others = [i for i in range(n_inputs) if i not in used]
    columns = [feature, *others]
    if others:
        disturbances = InputSliceLayer(others, name="input_slice_pcnn_disturbances")(sequence)
        features = keras.layers.Concatenate(name="concatenate_pcnn_unforced")([unforced, disturbances])
    else:
        features = unforced
    normalized = keras.layers.Normalization(
        mean=X[:, columns].mean(axis=0), variance=np.maximum(X[:, columns].var(axis=0), 1e-12),
        name="normalization_pcnn",
    )(features)
    prefix = {"RNN": "simple_rnn", "GRU": "gru", "LSTM": "lstm"}[rnn_layer]
    rnn = RNN_LAYERS[rnn_layer](units, return_sequences=True, return_state=True, name=f"{prefix}_pcnn_unforced")
    hidden, *new_states = rnn(normalized, initial_state=states)
    unforced_change = keras.layers.Dense(1, name="dense_pcnn_unforced")(hidden)
    unforced_change = keras.layers.Rescaling(
        scale=float(max(change.std(), 1e-12)), offset=0.0, name="rescaling_pcnn"
    )(unforced_change)
    prediction = keras.layers.Add(name="add_pcnn_change")([unforced_change, increment])
    return keras.Model([sequence, energy_state, *states], [prediction, new_energy_state, *new_states], name=name)


def pcnn_initial_values(step_model: keras.Model) -> list[float]:
    """
    Initial values of the parameters of the energy accumulator per time step, as built into the model (the constants
    `constant_pcnn_scale_<j>`). They are read from the model, since the initial rates of a loaded model may have been
    estimated from other data than the current one.
    """
    n_terms = int(step_model.get_layer(PCNN_PHYSICS_LAYER).get_weights()[0].shape[0])
    return [abs(float(np.ravel(step_model.get_layer(f"constant_pcnn_scale_{j}").get_weights()[0])[0]))
            for j in range(n_terms)]


def pcnn_parameters(step_model: keras.Model, terms: list[PCNNTerm] | None = None) -> list[float]:
    """
    Trained parameters of the energy accumulator per time step: the trained factor times the initial value built into
    the model, in the order of the terms of the model.

    Args:
        step_model: The PCNN step model.
        terms: Optional terms of the model, only to check their number.
    """
    weights = step_model.get_layer(PCNN_PHYSICS_LAYER).get_weights()[0][:, 0]
    initial = pcnn_initial_values(step_model)
    if terms is not None and len(terms) != len(weights):
        raise ValueError(f"The PCNN holds {len(weights)} terms, the model settings define {len(terms)}.")
    return [float(w * s) for w, s in zip(weights, initial, strict=True)]


def state_sizes(step_model: keras.Model) -> list[int]:
    """Sizes of the states of a step model."""
    return [int(t.shape[-1]) for t in step_model.inputs[1:]]


@keras.saving.register_keras_serializable(package="physXAI")
class ZeroStates(keras.layers.Layer):
    """Zero states (batch, size) for every size, with the batch size of the input."""

    def __init__(self, sizes: list[int], **kwargs):
        super().__init__(**kwargs)
        self.sizes = [int(s) for s in sizes]

    def call(self, inputs):
        batch = keras.ops.shape(inputs)[0]
        return [keras.ops.zeros((batch, size), dtype=inputs.dtype) for size in self.sizes]

    def get_config(self):
        return {**super().get_config(), "sizes": self.sizes}


@keras.saving.register_keras_serializable(package="physXAI")
class TimeSlice(keras.layers.Layer):
    """Steps [start, stop) of a sequence (batch, steps, features)."""

    def __init__(self, start: int = 0, stop: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.start, self.stop = int(start), None if stop is None else int(stop)

    def call(self, inputs):
        return inputs[:, self.start : self.stop, :]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


@keras.saving.register_keras_serializable(package="physXAI")
class Accumulate(keras.layers.Layer):
    """Trajectory of the features: start value plus the accumulated changes (absolute outputs as they are)."""

    def __init__(self, difference: list[bool], **kwargs):
        super().__init__(**kwargs)
        self.difference = [bool(d) for d in difference]

    def call(self, prediction, initial):
        mask = keras.ops.convert_to_tensor(np.asarray(self.difference, dtype="float32").reshape(1, 1, -1))
        accumulated = initial[:, None, :] + keras.ops.cumsum(prediction, axis=1)
        return mask * accumulated + (1.0 - mask) * prediction

    def get_config(self):
        return {**super().get_config(), "difference": self.difference}


class ClosedLoopCell(keras.layers.Layer):
    """
    RNN cell applying the step model one step at a time and feeding predicted features back.

    State: `[values of the predicted features, *states of the step model]`.
    Input: one row of input features; the fed back features are replaced by the predicted values.
    Output: [trajectory of the predicted features, outputs of the step model].
    """

    def __init__(self, step_model: keras.Model, spec: SequenceSpec, **kwargs):
        super().__init__(**kwargs)
        self.step_model = step_model
        slots = np.zeros((spec.n_outputs, spec.n_inputs), dtype="float32")
        for i, j in spec.feedback:
            slots[j, i] = 1.0
        self.slots = slots
        self.mask = slots.sum(axis=0, keepdims=True)
        self.difference = np.asarray(spec.difference, dtype="float32").reshape(1, -1)
        self.state_size = [spec.n_outputs, *state_sizes(step_model)]
        self.output_size = 2 * spec.n_outputs

    def call(self, inputs, states):
        values, *rnn_states = states
        row = inputs * (1.0 - self.mask) + keras.ops.matmul(values, self.slots)
        prediction, *rnn_states = self.step_model([row[:, None, :], *rnn_states])
        prediction = prediction[:, 0, :]
        values = self.difference * (values + prediction) + (1.0 - self.difference) * prediction
        return keras.ops.concatenate([values, prediction], axis=-1), [values, *rnn_states]


@keras.saving.register_keras_serializable(package="physXAI")
class TakeFeatures(keras.layers.Layer):
    """Features [start, stop) of the last axis."""

    def __init__(self, start: int, stop: int, **kwargs):
        super().__init__(**kwargs)
        self.start, self.stop = int(start), int(stop)

    def call(self, inputs):
        return inputs[..., self.start : self.stop]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "stop": self.stop}


def needs_initial(spec: SequenceSpec, trajectory: bool) -> bool:
    """Whether the sequence model needs the start values of the predicted features."""
    return bool(spec.feedback) or (trajectory and spec.absolute and any(spec.difference))


def build_sequence_model(
    step_model: keras.Model, spec: SequenceSpec, warmup: int, horizon: int, trajectory: bool = True
) -> keras.Model:
    """
    Wraps the step model for the prediction of sequences (see the module docstring).

    Args:
        step_model: The step model.
        spec: In- and outputs of the model.
        warmup: Number of warmup steps.
        horizon: Number of predicted steps.
        trajectory: If True (and the start values are known), the model returns the trajectories of
                    the predicted features, else the outputs of the step model (e.g. the changes).

    Returns:
        keras.Model: [sequence (batch, warmup + horizon, inputs), start values (batch, outputs) if
        `needs_initial`] -> (batch, horizon, outputs).
    """
    trajectory = trajectory and spec.absolute
    sequence = keras.Input(shape=(warmup + horizon, spec.n_inputs), name="sequence")
    inputs = [sequence]
    initial = None
    if needs_initial(spec, trajectory):
        initial = keras.Input(shape=(spec.n_outputs,), name="initial")
        inputs.append(initial)
    states = ZeroStates(state_sizes(step_model))(sequence)

    if not spec.feedback:
        # open loop: the whole sequence in one pass, starting from zero states
        prediction, *_ = step_model([sequence, *states])
        if warmup:
            prediction = TimeSlice(warmup)(prediction)
        if trajectory and any(spec.difference):
            prediction = Accumulate(spec.difference)(prediction, initial)
        output = prediction
    else:
        # warmup on the measured past, then closed loop over the horizon
        if warmup:
            _, *states = step_model([TimeSlice(0, warmup)(sequence), *states])
        future = TimeSlice(warmup)(sequence) if warmup else sequence
        cell = ClosedLoopCell(step_model, spec)
        result = keras.layers.RNN(cell, return_sequences=True)(future, initial_state=[initial, *states])
        start = 0 if trajectory else spec.n_outputs
        output = TakeFeatures(start, start + spec.n_outputs)(result)
    return keras.Model(inputs, output, name="rnn_sequence")


def find_step_model(model: keras.Model) -> keras.Model:
    """The step model of a sequence model (or the model itself, if it is a step model)."""
    if model.name == STEP_MODEL_NAME:
        return model
    try:
        return model.get_layer(STEP_MODEL_NAME)
    except ValueError:
        pass
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN) and isinstance(layer.cell, ClosedLoopCell):
            return layer.cell.step_model
    raise ValueError(f"The model '{model.name}' holds no step model '{STEP_MODEL_NAME}'.")


def sequence_inputs(td, split: str, spec: SequenceSpec, trajectory: bool, horizon=None, stride=None):
    """Inputs of the sequence model and the matching truth for a split (None if the split is empty)."""
    data = td.sequences(split, horizon, stride)
    if data is None:
        return None
    X, targets, initial = data
    trajectory = trajectory and spec.absolute
    x = [X, initial] if needs_initial(spec, trajectory) else X
    y = td.trajectory(targets, initial) if trajectory else targets
    return x, y


def predict_sequences(model: keras.Model, td, horizon: int, stride: int) -> dict:
    """
    Predicts the trajectories of all sequences of `horizon` steps.

    Returns:
        dict: {split: (prediction, truth)}, both of shape (sequences, horizon, outputs).
    """
    from physXAI.preprocessing.splitting import SPLITS

    step_model = find_step_model(model)
    spec = SequenceSpec.from_training_data(td)
    network = build_sequence_model(step_model, spec, td.warmup_width, horizon, trajectory=True)
    predictions = dict()
    for split in SPLITS:
        data = sequence_inputs(td, split, spec, True, horizon, stride)
        if data is not None:
            predictions[split] = (network.predict(data[0], verbose=0, batch_size=1024), data[1])
    return predictions


def evaluate_sequences(model: keras.Model, td) -> None:
    """
    Evaluates a recurrent model on the sequences of `td` as it is applied in an MPC: warmup on the
    measured past, closed loop for fed back features, accumulated changes. Stores the predictions,
    the metrics (`MetricsMultiStep`) and the errors over the evaluation horizon (`HorizonMetrics`).
    """
    from physXAI.evaluation.metrics import MetricsMultiStep
    from physXAI.evaluation.rollout import HorizonMetrics

    label = predict_sequences(model, td, td.label_width, td.stride)

    def prediction(split):
        return label[split][0] if split in label else None

    td.add_predictions(prediction("train"), prediction("val"), prediction("test"))
    td.add_metrics(MetricsMultiStep(td))

    if td.eval_horizon == td.label_width and td.eval_stride == td.stride:
        horizon = label
    else:
        horizon = predict_sequences(model, td, td.eval_horizon, td.eval_stride)
    td.horizon_predictions = horizon
    if horizon:
        features = td.features if td.absolute else td.outputs
        td.horizon_metrics = HorizonMetrics(horizon, td.time_step, features)
