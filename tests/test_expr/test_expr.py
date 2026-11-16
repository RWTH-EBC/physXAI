import json

import numpy as np
import pandas as pd
import pytest

from physXAI import expr as px


@pytest.fixture
def v():
    return px.Vars()


def test_arithmetic_and_names(v):
    e = (v.a * (v.b - v.c) + 2) / v.a**2
    values = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 3.0]), "c": np.array([1.0, 2.0])}
    np.testing.assert_allclose(e.evaluate(values), [(1 * 2 + 2) / 1, (2 * 1 + 2) / 4])
    assert e.name == "(((a*(b-c))+2)/(a**2))"
    assert e.free_vars() == ["a", "b", "c"]
    assert not e.trainable
    assert e.named("x").name == "x"
    assert v["Func(logistic)"].column == "Func(logistic)"


def test_reversed_operators_and_negation(v):
    values = pd.DataFrame({"a": [2.0, 4.0]})
    np.testing.assert_allclose((1 - v.a).evaluate(values), [-1, -3])
    np.testing.assert_allclose((8 / v.a).evaluate(values), [4, 2])
    np.testing.assert_allclose((2**v.a).evaluate(values), [4, 16])
    np.testing.assert_allclose((-v.a).evaluate(values), [-2, -4])


def test_lag_shifts_all_variables(v):
    e = v.m * (v.T_in - v.T.lag(1))
    assert e.lag(2).free_vars() == ["m_lag2", "T_in_lag2", "T_lag3"]
    assert v.T.lag(0) is not None and v.T.lag(0).column == "T"


def test_activations(v):
    x = np.linspace(-2, 2, 5)
    for fn, ref in [
        (px.softplus, np.log1p(np.exp(x))),
        (px.tanh, np.tanh(x)),
        (px.relu, np.maximum(x, 0)),
        (px.sigmoid, 1 / (1 + np.exp(-x))),
        (px.exp, np.exp(x)),
    ]:
        np.testing.assert_allclose(fn(v.x).evaluate({"x": x}), ref, rtol=1e-12)
    with pytest.raises(ValueError):
        px.Activation(v.x, "sin")


def test_casadi_backend(v):
    import casadi as ca

    e = px.softplus(v.a * v.b) - v.a / 2
    syms = {"a": ca.MX.sym("a"), "b": ca.MX.sym("b")}
    f = ca.Function("f", [syms["a"], syms["b"]], [e.to_casadi(syms)])
    assert float(f(1.5, -2.0)) == pytest.approx(float(e.evaluate({"a": 1.5, "b": -2.0})))


def test_shared_nodes_stay_shared(v):
    k = px.Param(0.5, name="k")
    e = px.Linear([k * v.a, k * v.b])
    restored = px.from_config(json.loads(json.dumps(px.to_config(e))))
    params = [n for n in restored.nodes() if isinstance(n, px.Param)]
    assert len(params) == 1
    assert restored.trainable


def _all_node_types(v):
    k = px.Param(1.0, trainable=False, name="k")
    heat = (v.m * (v.T_in - v.T)).named("heat")
    return [
        px.Linear([v.m, heat, v.T], monotonicities={"heat": 1}, nominal_range=(-1, 1)),
        px.MLP([v.m, v.T], n_layers=2, n_neurons=[4, 3], activation=["tanh", "relu"]),
        px.MonotoneMLP([v.m, heat], monotonicities={"m": -1}, n_neurons=6, activation_split=[1, 0, 1]),
        px.Polynomial([v.m, v.T], degree=3, interaction_degree=2),
        px.RBF([v.m, v.T], units=4),
        px.Mean([v.m, v.T, k]),
        px.Normalize(heat),
        px.Linear([px.sigmoid(v.m), px.Const(2.0) ** v.m, v.m / v.T]),
    ]


def test_serialization_roundtrip_of_all_node_types(v):
    for e in _all_node_types(v):
        config = json.loads(json.dumps(px.to_config(e)))
        restored = px.from_config(config)
        assert type(restored) is type(e)
        assert restored.name == e.name
        assert px.to_config(restored) == config, e.name


def test_list_serialization(v):
    exprs = [v.a + v.b, px.Normalize(v.a)]
    restored = px.from_config(px.to_config(exprs))
    assert [e.name for e in restored] == [e.name for e in exprs]


def test_block_validation(v):
    with pytest.raises(ValueError, match="unknown inputs"):
        px.Linear([v.a], monotonicities={"b": 1})
    with pytest.raises(ValueError):
        px.MLP([v.a], n_layers=2, n_neurons=[3])
    with pytest.raises(ValueError):
        px.Linear([])
    with pytest.raises(NotImplementedError):
        px.Linear([v.a]).evaluate({"a": 1.0})
