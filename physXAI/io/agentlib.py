"""
Export of physXAI models for AgentLib-MPC.

AgentLib-MPC (`agentlib_mpc.machine_learning_plugins.physXAI`) executes a physXAI training
script and reads the following files from the run directory:

- `{name}_preprocessing.json`  (inputs, output, shift, time_step, split info)
- `{name}_model.json`          (at least `__class_name__`)
- `{name}_training_data.json`  (optional, stored as training info)
- `{name}.keras` or `{name}.joblib` (the trained model)

The input and output names follow two conventions, which AgentLib-MPC parses:
`Change(X)` denotes an output predicting the difference of `X`, and `X_lagN` the value of
`X` N time steps in the past. `check_agentlib_compatibility` validates the same rules as
AgentLib-MPC before a model is trained, so a run fails early instead of after training.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from physXAI.io.run import Run, logger

if TYPE_CHECKING:  # pragma: no cover
    from physXAI.preprocessing.preprocessing import PreprocessingData

# The name rules below mirror the parser of AgentLib-MPC and are independent of `physXAI.preprocessing` on
# purpose: the check has to decide as AgentLib-MPC does, also if the rules of physXAI change.
CHANGE_PATTERN = re.compile(r"Change\((.*)\)")
LAG_PATTERN = re.compile(r"_lag(\d+)$")
MULTI_STEP_CLASS = "PreprocessingMultiStep"
#: Branch of AgentLib-MPC supporting recurrent models (AgentLib-MPC releases up to 1.1.1 have no support)
AGENTLIB_RNN_BRANCH = "143-feature-multistep-model-support"


class AgentLibCompatibilityError(ValueError):
    """Raised if a preprocessing config cannot be translated by AgentLib-MPC."""


def split_lag(name: str) -> tuple[str, int]:
    """Splits 'X_lag2' into ('X', 2) and 'X' into ('X', 0)."""
    match = LAG_PATTERN.search(name)
    if match is None:
        return name, 0
    return name[: match.start()], int(match.group(1))


def output_feature(name: str) -> tuple[str, str]:
    """Returns the name of the predicted feature and the output type ('absolute' or 'difference')."""
    match = CHANGE_PATTERN.fullmatch(name)
    if match is None:
        return name, "absolute"
    return match.group(1).strip(), "difference"


def agentlib_supports_recurrent_models() -> bool | None:
    """Whether the installed AgentLib-MPC loads recurrent models (`RNNModel`, `PCNNModel`), None if not installed."""
    try:
        from agentlib_mpc.models import casadi_predictor
    except ImportError:
        return None
    return hasattr(casadi_predictor, "CasadiRNN")


def _check_recurrent_support(config: dict) -> None:
    """Recurrent models need an AgentLib-MPC with support for them: warns if the installed one has none."""
    if config.get("__class_name__") != MULTI_STEP_CLASS:
        return
    supported = agentlib_supports_recurrent_models()
    message = (
        "Recurrent models (RNNModel, PCNNModel) need AgentLib-MPC with support for recurrent models (branch "
        f"'{AGENTLIB_RNN_BRANCH}'); AgentLib-MPC releases up to 1.1.1 cannot load them."
    )
    if supported is False:
        logger.warning("%s The installed AgentLib-MPC has no support.", message)
    elif supported is None:
        logger.info(message)


def check_agentlib_compatibility(preprocessing: PreprocessingData | dict) -> None:
    """
    Checks whether AgentLib-MPC can translate a preprocessing config.

    Args:
        preprocessing: A physXAI preprocessing object or its config dict.

    Raises:
        AgentLibCompatibilityError: Listing all violated rules.
    """
    config = preprocessing if isinstance(preprocessing, dict) else preprocessing.get_config()
    errors = list()

    if config.get("shift", 1) != 1:
        errors.append(
            f"'shift' has to be 1, but is {config.get('shift')} (AgentLib-MPC evaluates outputs over the interval "
            "[t, t+1) of the inputs, as trained with shift=1)."
        )
    inputs = list(config.get("inputs") or [])
    outputs = config.get("output")
    if isinstance(outputs, str):
        outputs = [outputs]
    if not inputs:
        errors.append("'inputs' must not be empty.")

    if config.get("__class_name__") == MULTI_STEP_CLASS:
        # recurrent models carry their memory in hidden states, see AgentLib-MPC 'KerasRNN'
        lagged = [name for name in inputs if LAG_PATTERN.search(name)]
        if lagged:
            errors.append(f"Multi step models must not have lagged inputs, increase 'warmup_width' instead: {lagged}")
        if not outputs:
            errors.append("'output' must not be empty.")
    else:
        if not isinstance(outputs, list) or len(outputs) != 1:
            errors.append(f"Single step models need exactly one output, but got {outputs}.")
        else:
            errors.extend(_check_lag_order(inputs, output_feature(outputs[0])[0]))

    if errors:
        raise AgentLibCompatibilityError("Config cannot be used in AgentLib-MPC:\n- " + "\n- ".join(errors))
    _warn_derived_inputs(config, inputs)
    _check_recurrent_support(config)


def _warn_derived_inputs(config: dict, inputs: list[str]) -> None:
    """Inputs computed by the preprocessing are no variables of the MPC: AgentLib-MPC has to compute them itself."""
    derived = set(config.get("columns") or {})
    computed = [
        name for name in inputs if split_lag(name)[0] in derived or CHANGE_PATTERN.fullmatch(split_lag(name)[0])
    ]
    if computed:
        logger.warning(
            "The inputs %s are computed by the preprocessing, AgentLib-MPC needs them as variables of the MPC model. "
            "Expressions inside the model (physXAI.expr, ModularANN) need only the raw inputs.",
            computed,
        )
    if config.get("transform"):
        logger.warning(
            "The preprocessing uses the transform '%s': inputs it creates have to be variables of the MPC model.",
            config["transform"],
        )


def _check_lag_order(inputs: list[str], recursive_feature: str) -> list[str]:
    """
    Lags of a feature have to start at the feature itself (lag 0), be consecutive and ascending, and the recursive
    feature has to be last.

    AgentLib-MPC only counts the inputs of a feature and passes its values from lag 0 on: for the inputs
    ['u', 'T_lag1'] it would pass T(k) instead of T(k-1), without an error. AgentLib-MPC does not check this itself,
    so this rule is stricter than its parser. physXAI itself allows isolated lags (e.g. a dead time 'u_lag3').
    """
    errors = list()
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, name in enumerate(inputs):
        base, lag = split_lag(name)
        groups[base].append((index, lag))

    for base, items in groups.items():
        first = min(lag for _, lag in items)
        if first != 0:
            missing = [base] + [f"{base}_lag{k}" for k in range(1, first)]
            errors.append(
                f"The inputs of '{base}' start at lag {first}: AgentLib-MPC passes the values of '{base}' from lag 0 on, "
                f"add {missing} as inputs."
            )
        for (i0, lag0), (i1, lag1) in zip(items, items[1:], strict=False):
            if i1 != i0 + 1:
                errors.append(
                    f"Inputs of '{base}' are not grouped consecutively: {inputs[i0]} at {i0}, {inputs[i1]} at {i1}."
                )
            if lag1 != lag0 + 1:
                errors.append(f"Lags of '{base}' are not ascending by one: {inputs[i0]}, {inputs[i1]}.")

    if recursive_feature in groups:
        indices = [index for index, _ in groups[recursive_feature]]
        expected = list(range(len(inputs) - len(indices), len(inputs)))
        if indices != expected:
            errors.append(f"The recursive feature '{recursive_feature}' and its lags have to be the last inputs.")
    return errors


def save_for_agentlib(run: Run, name: str, preprocessing, model, training_data=None, trained_model=None) -> str:
    """
    Writes all files AgentLib-MPC reads into the run directory.

    Args:
        run: The run directory. AgentLib-MPC passes `base_path` and `folder_name` to the training
             script, which correspond to `Run(base_path, folder_name)`.
        name: Name of the model, i.e. the prefix of all files.
        preprocessing: The physXAI preprocessing object.
        model: The physXAI model object.
        training_data: Optional training data, stored as training info.
        trained_model: The trained (keras or scikit-learn) model returned by the training pipeline.
                       Defaults to `model.trained_model` if the model object holds it.

    Returns:
        str: `name`, the return value expected from a training script.
    """
    check_agentlib_compatibility(preprocessing)
    if trained_model is None:
        trained_model = getattr(model, "trained_model", None)
    if trained_model is None:
        raise ValueError("No trained model given. Pass the model returned by the training pipeline as 'trained_model'.")
    run.save(name, preprocessing, model, training_data, trained_model, plot=False)
    logger.info("Saved model '%s' for AgentLib-MPC in %s", name, run.path)
    return name
