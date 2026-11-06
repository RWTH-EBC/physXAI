"""Physics residuals with trainable physical parameters (`physics=PhysicsResidual(...)`)."""

import json

import keras
import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.models.ann.ann_design import ClassicalANNModel, PINNModel
from physXAI.models.ann.pinn.physics import PhysicsResidual
from physXAI.models.models import AbstractModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.training import RecursiveTraining

#: The simulated room: T(t+1) - T(t) = K1 * mDot * (T_in - T) + K2 * load (dt = 300 s, C = 1e5 J/K, cp = 1000 J/kgK)
K1, K2 = 3.0, 0.003


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1000).to_csv(path, sep=";")
    return str(path)


def _residual(k1=2.0, k2=0.002, weight=0.5):
    """Disagreement of the prediction of Change(T) with the grey-box model, parameters start off by 1.5."""
    v = px.Vars()
    physics = px.positive(k1, "k1") * v.mDot * (v.T_in - v.T) + px.positive(k2, "k2") * v.load
    return PhysicsResidual(px.Prediction() - physics, weight=weight, name="room")


def _fit(csv, inputs=("mDot", "T_in", "load", "T"), **kwargs):
    kwargs = {"n_neurons": 4, "epochs": 40, "learning_rate": 0.003, "early_stopping_epochs": None,
              "physics": _residual(), **kwargs}
    m = ClassicalANNModel(**kwargs)
    prep = PreprocessingSingleStep(inputs=list(inputs), output="Change(T)", random_state=0)
    td = prep.pipeline(csv, model=m)
    model = m.pipeline(td, plot=False, save_model=False)
    return prep, m, td, model


def test_prediction_and_parameter_expressions():
    residual = px.Prediction() - px.positive(2.0, "k") * px.Var("T")
    assert residual.free_vars() == ["T"]
    restored = px.from_config(json.loads(json.dumps(px.to_config(residual))))
    assert restored.name == residual.name
    assert float(px.freeze(px.positive(2.0, "k"), {"log_k": np.log(1.5)}).evaluate({})) == pytest.approx(3.0)
    assert px.Target("Change(T)").column == "y:Change(T)"
    with pytest.raises(ValueError, match="positive"):
        px.positive(0.0, "k")
    ctx = px.BuildContext(keras.Input(shape=(1,)), ["T"])
    with pytest.raises(ValueError, match="physics residuals"):
        ctx.tensor(px.Prediction())


def test_calibration_recovers_the_physical_parameters(csv, tmp_path):
    prep, m, td, model = _fit(csv)
    parameters = m.physics_parameters
    assert parameters["k1"] == pytest.approx(K1, rel=0.1)
    assert parameters["k2"] == pytest.approx(K2, rel=0.1)
    assert set(td.physics) == {"name", "parameters", "fit", "calibration_record"}
    assert td.physics["fit"]["test"]["fit_rmse"] < 0.2 * float(np.std(td.y_test))
    history = td.training_record.history
    assert {"data_loss", "room_loss", "room_fit_loss", "rmse", "val_data_loss"} <= set(history)
    record = td.physics["calibration_record"]  # deterministic least squares, not gradient descent
    assert record["method"] == "least_squares" and record["success"]
    assert record["fit_loss"] < record["fit_loss_initial"]
    json.dumps(td.get_config())

    # only the network is saved and exported: inputs of the model, one output, no physics layers
    assert len(model.inputs) == 1 and model.inputs[0].shape[-1] == len(td.inputs)
    assert all("log_k" not in w.path for w in model.weights)
    run = Run(tmp_path, "physics")
    run.save_for_agentlib("T", prep, m, td)
    loaded = keras.saving.load_model(run.path / "T.keras")
    np.testing.assert_allclose(loaded.predict(td.X_test, verbose=0), model.predict(td.X_test, verbose=0), atol=1e-6)

    restored = AbstractModel.model_from_config(json.loads(json.dumps(m.get_config())))
    assert restored.physics.weight == 0.5 and restored.physics_fit == "calibrate"
    assert restored.physics.columns == m.physics.columns


@pytest.mark.parametrize("mode", ["fixed", "joint"])
def test_physics_fit_modes(csv, mode):
    _, m, _, _ = _fit(csv, physics_fit=mode, epochs=3)
    if mode == "fixed":
        assert m.physics_parameters["k1"] == pytest.approx(2.0) and m.physics_parameters["k2"] == pytest.approx(0.002)
    else:
        assert m.physics_parameters["k1"] != pytest.approx(2.0)


def test_residual_columns_need_not_be_inputs(csv):
    """The residual may use columns the model does not see (e.g. 'load'), they become loss columns."""
    prep, m, td, model = _fit(csv, inputs=("mDot", "T_in", "T"), epochs=2)
    assert "load" not in td.inputs and "load" in td.loss_columns and prep.loss_columns == m.physics.columns
    assert model.inputs[0].shape[-1] == 3
    # without the model the preprocessing does not know the columns
    td = PreprocessingSingleStep(inputs=["mDot", "T_in", "T"], output="Change(T)").pipeline(csv)
    with pytest.raises(ValueError, match="model=model"):
        ClassicalANNModel(physics=_residual(), epochs=1).pipeline(td, plot=False, save_model=False)


def test_recursive_training_after_the_physics(csv):
    _, m, td, _ = _fit(csv, epochs=2, recursive_training=RecursiveTraining(horizon=3, epochs=1))
    assert td.recursive_training_record is not None and m.physics_parameters is not None


def test_invalid_residuals():
    v = px.Vars()
    with pytest.raises(ValueError, match="Prediction"):
        PhysicsResidual(v.T - v.T_in)
    with pytest.raises(ValueError, match="blocks"):
        PhysicsResidual(px.Prediction() - px.MLP([v.T], n_neurons=2))
    with pytest.raises(ValueError, match="physics_fit"):
        ClassicalANNModel(physics=_residual(), physics_fit="sometimes")
    with pytest.raises(ValueError, match="ols"):
        ModularANN(px.Linear([v.T]), solver="ols", physics=_residual())
    with pytest.raises(ValueError, match="physics targets"):
        PINNModel(physics_targets={"a": 1.0}, physics=_residual())


def test_calibration_is_deterministic(csv):
    """The calibration does not depend on the seed of the network (batch order, initialization)."""
    parameters = [_fit(csv, epochs=1, random_seed=seed)[1].physics_parameters for seed in (0, 1)]
    assert parameters[0]["k1"] == parameters[1]["k1"] and parameters[0]["k2"] == parameters[1]["k2"]
    assert parameters[0]["k1"] == pytest.approx(K1, rel=1e-3)
    assert parameters[0]["k2"] == pytest.approx(K2, rel=1e-3)


def test_calibration_by_gradient_descent_without_casadi_form(csv):
    """A parameter without initial value has no least squares start, the calibration falls back to gradients."""
    v = px.Vars()
    residual = PhysicsResidual(px.Prediction() - px.Param(None, name="k") * v.load, name="room")
    _, _, td, _ = _fit(csv, physics=residual, epochs=2)
    assert "val_room_fit_loss" in td.physics["calibration_record"]
