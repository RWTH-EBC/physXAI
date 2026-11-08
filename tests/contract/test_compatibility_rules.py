"""
`check_agentlib_compatibility` has to reject everything AgentLib-MPC rejects and accept what AgentLib-MPC translates
correctly. It is stricter for lags without the feature itself, which AgentLib-MPC accepts but translates wrongly.
"""

import pytest
from conftest import import_agentlib

import_agentlib("agentlib_mpc")
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


@pytest.mark.parametrize("inputs", [["u", "T_lag1"], ["u_lag1", "T"]])
def test_isolated_lags_are_stricter_than_agentlib(inputs):
    """
    AgentLib-MPC accepts lags without the feature itself, but passes the values from lag 0 on: for 'T_lag1' it passes
    T(k) (one recursive input), for 'u_lag1' the values u(k) and u(k-1). physXAI rejects these configs.
    """
    translated = physXAI_2_agentlib_json("001", _config(inputs))
    lags = {**{n: f["lag"] for n, f in translated["input"].items()}, **{n: f["lag"] for n, f in translated["output"].items()}}
    assert lags == ({"u": 1, "T": 1} if inputs[1] == "T_lag1" else {"u": 2, "T": 1})  # no trace of the lag 1 offset
    with pytest.raises(AgentLibCompatibilityError, match="lag 0 on"):
        check_agentlib_compatibility(_config(inputs))


def test_multi_step_rejects_lags():
    config = {**_config(["u", "u_lag1"]), "__class_name__": "PreprocessingMultiStep", "warmup_width": 4}
    with pytest.raises(AgentLibCompatibilityError, match="lagged"):
        check_agentlib_compatibility(config)
