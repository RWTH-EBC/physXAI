"""`check_agentlib_compatibility` has to accept and reject exactly what AgentLib-MPC accepts and rejects."""

import pytest

pytest.importorskip("agentlib_mpc")
from agentlib_mpc.machine_learning_plugins.physXAI.model_config_creation import physXAI_2_agentlib_json  # noqa: E402

from physXAI.io import AgentLibCompatibilityError, check_agentlib_compatibility  # noqa: E402

pytestmark = pytest.mark.agentlib


def _config(inputs, output="Change(T)", shift=1):
    return {
        "__class_name__": "PreprocessingSingleStep",
        "inputs": inputs,
        "output": [output] if isinstance(output, str) else output,
        "shift": shift,
        "time_step": 300,
        "test_size": 0.1,
        "val_size": 0.1,
        "random_state": 1,
    }


CONFIGS = {
    "plain": _config(["u", "T"]),
    "lags": _config(["u", "u_lag1", "u_lag2", "T", "T_lag1"]),
    "absolute_output": _config(["u", "T"], output="T"),
    "non_recursive": _config(["u", "d"], output="P"),
    "recursive_not_last": _config(["T", "u"]),
    "lags_not_grouped": _config(["u", "T", "u_lag1"]),
    "lags_not_ascending": _config(["u", "u_lag2", "T"]),
    "two_outputs": _config(["u", "T"], output=["Change(T)", "P"]),
    "shift_2": _config(["u", "T"], shift=2),
    "shift_0": _config(["u", "d"], output="P", shift=0),
}


@pytest.mark.parametrize("name", CONFIGS)
def test_same_decision_as_agentlib(name):
    config = CONFIGS[name]
    try:
        physXAI_2_agentlib_json("001", dict(config))
        agentlib_accepts = True
    except ValueError:
        agentlib_accepts = False
    try:
        check_agentlib_compatibility(config)
        physxai_accepts = True
    except AgentLibCompatibilityError:
        physxai_accepts = False
    assert physxai_accepts == agentlib_accepts, name


def test_multi_step_rejects_lags():
    config = {**_config(["u", "u_lag1"]), "__class_name__": "PreprocessingMultiStep", "warmup_width": 4}
    with pytest.raises(AgentLibCompatibilityError, match="lagged"):
        check_agentlib_compatibility(config)
