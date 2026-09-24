import os
import re
from typing import Optional, Union
import numpy as np
from physXAI.preprocessing.training_data import TrainingDataMultiStep
from physXAI.utils.logging import Logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


change_pattern = r"Change\((.*)\)"  # Naming convention for outputs holding a difference


def derive_feedback(td: TrainingDataMultiStep) -> dict[str, str]:
    """
    Determines which input features a multi step model predicts itself.

    An input feature is fed back if there is an output for it, either directly or as a
    difference following the 'Change(feature)' naming convention.

    Args:
        td (TrainingDataMultiStep): The multi step training data.

    Returns:
        dict[str, str]: Maps the name of an input feature to the name of the output that
                        updates it.
    """

    feedback = dict()
    for output in td.output:
        match = re.match(change_pattern, output)
        feature = match.group(1).strip() if match else output
        if feature in td.columns:
            feedback[feature] = output
    return feedback


def closed_loop_predictions(model: keras.Model, td: TrainingDataMultiStep,
                            feedback: Optional[dict[str, str]] = None,
                            warmup: str = 'model') -> Optional[tuple]:
    """
    Predicts the multi step data closed loop.

    During training a multi step model is teacher forced: at every time step it sees the
    measured value of the features it predicts itself. Herem the prediction of a time step
    overwrites the corresponding input feature of the next one, so the resulting error
    shows how the model actually behaves in an MPC.

    Args:
        model (keras.Model): The trained multi step model.
        td (TrainingDataMultiStep): The multi step training data.
        feedback (dict[str, str], optional): Maps the name of an input feature to the
                        name of the output that updates it. Derived from the naming of
                        the outputs if None.
        warmup (str): How the hidden states are initialized. 'model' initializes them
                        the way the model was trained, i.e. with its initialization model
                        or, if it has none, by applying the model itself to the warmup
                        sequence, exactly as an MPC does. 'zeros' skips the warmup and
                        starts from zeros.

    Returns:
        The closed loop predictions for the training, validation and test set, in the
        same shape as the teacher forced ones. None, if the model does not predict any
        of its own input features.
    """

    if feedback is None:
        feedback = derive_feedback(td)
    if not feedback:
        return None

    slots = _feedback_slots(td, feedback)
    step_model = _step_model(model)

    predictions = list()
    for X, y in ((td.X_train, td.y_train), (td.X_val, td.y_val), (td.X_test, td.y_test)):
        if X is None or y is None:
            predictions.append(None)
        else:
            predictions.append(_roll_out(model, step_model, X, y.shape[1], slots, warmup))
    return tuple(predictions)


def _feedback_slots(td: TrainingDataMultiStep, feedback: dict[str, str]) -> list[tuple[int, int, bool]]:
    """
    Translates the feedback into indices.

    Returns:
        list[tuple[int, int, bool]]: For every fed back feature its index in the inputs,
                    the index of the output that updates it, and whether that output is
                    a difference rather than the value itself.
    """

    slots = list()
    for feature, output in feedback.items():
        if feature not in td.columns:
            raise ValueError(f"Closed loop error: The fed back feature '{feature}' is not an input "
                             f"of the model. Inputs are {td.columns}.")
        if output not in td.output:
            raise ValueError(f"Closed loop error: The output '{output}' updating '{feature}' is not "
                             f"an output of the model. Outputs are {td.output}.")
        slots.append((td.columns.index(feature), td.output.index(output), output != feature))
    return slots


def _step_model(model: keras.Model) -> keras.Model:
    """Returns the part of the model that performs a single recurrent step."""
    try:
        return model.get_layer('out_model')
    except ValueError:
        return model


def _initial_states(model: keras.Model, step_model: keras.Model, X: Union[np.ndarray, tuple],
                    samples: int, warmup: str) -> list[np.ndarray]:
    """Determines the hidden states at the beginning of the prediction."""

    units = [int(state.shape[-1]) for state in step_model.inputs[1:]]
    zeros = [np.zeros((samples, unit), dtype='float32') for unit in units]
    if warmup == 'zeros':
        return zeros

    try:
        init = model.get_layer('init_model')
    except ValueError:
        init = None

    if init is not None:
        states = init(X[1] if isinstance(X, tuple) else X, training=False)
        if not isinstance(states, (list, tuple)):
            states = [states]
        return [keras.ops.convert_to_numpy(state) for state in states]

    if not isinstance(X, tuple) or X[1].shape[-1] != X[0].shape[-1]:
        # the model was trained without a warmup, so it starts from zeros anyway
        return zeros

    # the model was trained to warm its own states up on the measured past, which is
    # what an MPC does. The warmup itself stays teacher forced, since the past is known.
    states = zeros
    for step in range(X[1].shape[1]):
        state_input = states[0] if len(states) == 1 else list(states)
        result = step_model([X[1][:, step:step + 1, :], state_input], training=False)
        states = [keras.ops.convert_to_numpy(state) for state in result[1:]]
    return states


def _roll_out(model: keras.Model, step_model: keras.Model, X: Union[np.ndarray, tuple], steps: int,
              slots: list[tuple[int, int, bool]], warmup: str) -> np.ndarray:
    """Applies the model one step at a time, feeding its predictions back into its inputs."""

    inputs = X[0] if isinstance(X, tuple) else X
    states = _initial_states(model, step_model, X, inputs.shape[0], warmup)

    # the first step still uses the measurement, just like the MPC does at the beginning
    # of its horizon
    current = {feature: inputs[:, 0, feature].copy() for feature, _, _ in slots}

    predictions = list()
    for step in range(steps):
        step_input = inputs[:, step:step + 1, :].copy()
        for feature, _, _ in slots:
            step_input[:, 0, feature] = current[feature]

        # a model with a single state does not expect it wrapped in a list
        state_input = states[0] if len(states) == 1 else list(states)
        result = step_model([step_input, state_input], training=False)
        prediction = keras.ops.convert_to_numpy(result[0])
        states = [keras.ops.convert_to_numpy(state) for state in result[1:]]
        predictions.append(prediction)

        for feature, output, difference in slots:
            value = prediction[:, 0, output]
            current[feature] = current[feature] + value if difference else value

    return np.concatenate(predictions, axis=1)


def evaluate_closed_loop(model: keras.Model, td: TrainingDataMultiStep, warmup: str = 'model'):
    """
    Predicts the multi step data closed loop and stores the predictions and the
    resulting metrics in the training data.

    Args:
        model (keras.Model): The trained multi step model.
        td (TrainingDataMultiStep): The multi step training data.
        warmup (str): How the hidden states are initialized, see closed_loop_predictions.
    """

    from physXAI.evaluation.metrics import MetricsMultiStep

    predictions = closed_loop_predictions(model, td, warmup=warmup)
    if predictions is None:
        return

    td.add_closed_loop_predictions(*predictions)
    Logger.print('Closed loop metrics:', 'info')
    td.add_closed_loop_metrics(MetricsMultiStep(td, predictions=predictions))
