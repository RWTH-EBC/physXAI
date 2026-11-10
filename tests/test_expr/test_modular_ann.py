import json

import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.models import AbstractModel, ModularANN, ShapeConstraints
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

EPOCHS = 2  # keep trainings in tests short


@pytest.fixture(scope="module")
def one_room(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


def _data(model, csv):
    prep = PreprocessingSingleStep(output=px.Change("T"), random_state=0)
    return prep, prep.pipeline(csv, model=model)


def _architectures():
    v = px.Vars()
    heat = (v.mDot * (v.T_in - v.T)).named("heat")
    k = px.Param(0.1, name="k")
    return {
        "linear": px.Linear([v.mDot, v.load, heat, v.T], monotonicities={"load": 1}),
        "mlp": px.MLP([v.mDot, v.load, v.T], n_neurons=4),
        "monotone": px.MonotoneMLP([v.mDot, v.load, heat], monotonicities={"load": 1}, n_neurons=6),
        "polynomial": px.Polynomial([v.mDot, v.T], degree=2, interaction_degree=2),
        "rbf": px.RBF([v.mDot, v.T], units=4),
        "nested": px.Linear(
            [px.MLP([v.mDot, v.T], n_neurons=3), k * heat, px.Mean([v.load, v.load.lag(1)]), px.Normalize(heat)]
        ),
    }


@pytest.mark.parametrize("name", list(_architectures()))
def test_train_save_load(name, one_room, tmp_path):
    architecture = _architectures()[name]
    m = ModularANN(architecture, epochs=EPOCHS, early_stopping_epochs=None)
    prep, td = _data(m, one_room)
    assert set(m.required_inputs) <= set(prep.inputs)
    assert prep.inputs[-1].startswith("T")  # the recursive feature is last (AgentLib-MPC)

    model = m.pipeline(td, plot=False, save_model=False)
    m.save_model(model, str(tmp_path / "m"))
    import keras

    loaded = keras.saving.load_model(tmp_path / "m.keras")
    x = td.X_test[:5]
    np.testing.assert_allclose(loaded.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6)

    config = json.loads(json.dumps(m.get_config()))
    restored = AbstractModel.model_from_config(config)
    assert isinstance(restored, ModularANN)
    assert json.loads(json.dumps(restored.get_config())) == config


def test_two_models_in_one_process(one_room):
    """There is no global state: the same expression can be built into several models."""
    v = px.Vars()
    shared = (v.mDot * (v.T_in - v.T)).named("heat")
    m1 = ModularANN(px.Linear([shared, v.T]), epochs=EPOCHS, early_stopping_epochs=None)
    m2 = ModularANN(px.MLP([shared, v.load]), epochs=EPOCHS, early_stopping_epochs=None)
    for m in (m1, m2, m1):
        _, td = _data(m, one_room)
        m.pipeline(td, plot=False, save_model=False)


def test_ols_recovers_the_physics(one_room):
    v = px.Vars()
    m = ModularANN(px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T]), solver="ols")
    _, td = _data(m, one_room)
    model = m.pipeline(td, plot=False, save_model=False)
    kernel, bias = model.get_layer(index=-1).get_weights()
    # dT = dt/C * (load + cp * mDot * (T_in - T))
    np.testing.assert_allclose(kernel.reshape(-1), [0, 300 / 1e5, 300 / 1e5 * 1000, 0], atol=1e-5)
    assert td.metrics.kpi("test", "r2") > 0.999


@pytest.mark.parametrize(
    "architecture", [px.MLP(["a"]), px.Linear(["a"], monotonicities={"a": 1}), px.Linear([px.Param(1.0) * px.Var("a")])]
)
def test_ols_requirements(architecture):
    with pytest.raises(ValueError, match="ols"):
        ModularANN(architecture, solver="ols")


def test_missing_inputs_are_reported(one_room):
    v = px.Vars()
    m = ModularANN(px.Linear([v.mDot, v.unknown]))
    with pytest.raises(ValueError, match="unknown"):
        PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)").pipeline(one_room, model=m)


def test_constrained_regression(one_room):
    """More cold supply air cools the room: only where T_in < T, so the constraint holds on the data, not the box."""
    v = px.Vars()
    heat = (v.mDot * (v.T_in - v.T)).named("heat")
    shape = ShapeConstraints(monotonicities={"load": 1, "mDot": -1}, domain="data")
    m = ModularANN(px.Linear([v.load, heat, v.T]), solver="qp", shape=shape)
    _, td = _data(m, one_room)
    m.pipeline(td, plot=False, save_model=False)
    assert td.metrics.kpi("test", "r2") > 0.99
    assert [c["certificate"] for c in td.physics["shape_constraints"]["constraints"]] == ["data", "data"]
    config = json.loads(json.dumps(m.get_config()))
    loaded = AbstractModel.model_from_config(config)
    assert [e.name for e in loaded.architecture.inputs] == ["load", "heat", "T"]
    assert loaded.solver == "qp" and loaded.shape.monotonicities == {"load": 1, "mDot": -1}
