"""Physics-informed models (PINN) and models trained on data without validation or test split."""

import json
from unittest.mock import patch

import keras
import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.evaluation.comparison import ModelComparison
from physXAI.io import Run, check_agentlib_compatibility
from physXAI.models import LinearRegressionModel, ModularANN
from physXAI.models.ann.ann_design import ClassicalANNModel, PINNModel, RNNModel
from physXAI.models.ann.pinn.physics import PhysicsLoss, TermLoss
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep
from physXAI.training import RecursiveTraining

DT, CAPACITY, CP = 300.0, 100000.0, 1000.0


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=800).to_csv(path, sep=";")
    return str(path)


def _physics_step():
    """Exact step of the simulated room computed from the data at time t (K per step)."""
    v = px.Vars()
    return DT / CAPACITY * (v.mDot * CP * (v.T_in - v.T) + v.load)


def _pinn_data(csv, **prep_kwargs):
    # the physics prediction of T(t+1) - T(t) is computed from the data at t; physics targets are taken
    # at the predicted time t+1 like the output, so the column is shifted back by one step
    prep = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)",
                                   columns={"dT_rc": _physics_step().lag(1)}, random_state=0, **prep_kwargs)
    m = PINNModel(physics_targets={"dT_rc": 0.5}, n_neurons=4, epochs=2, early_stopping_epochs=1)
    td = prep.pipeline(csv, model=m)
    return prep, m, td


def test_physics_targets_are_aligned_with_the_output(csv):
    prep, _, td = _pinn_data(csv)
    assert prep.loss_columns == ["y:dT_rc"] and td.loss_columns == ["y:dT_rc"]
    np.testing.assert_allclose(td.values("train", ["y:dT_rc"]), td.y_train, atol=1e-6)


def test_loss_matches_a_hand_computation():
    y_true = np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 1.0]], dtype="float32")
    y_pred = np.array([[1.5], [2.0]], dtype="float32")
    loss = PhysicsLoss([0.5, 2.0], scale=4.0)
    data = ((1.5 - 1) ** 2 + (2 - 3) ** 2) / 2
    first = ((1.5 - 2) ** 2 + (2 - 1) ** 2) / 2
    second = ((1.5 - 0) ** 2 + (2 - 1) ** 2) / 2
    assert float(loss(y_true, y_pred)) == pytest.approx((data + 0.5 * first + 2.0 * second) / 4.0)
    metric = TermLoss(1, scale=4.0, name="a_loss")
    metric.update_state(y_true, y_pred)
    assert float(metric.result()) == pytest.approx(first / 4.0)
    # serializable, so saved models load in AgentLib-MPC
    restored = keras.losses.deserialize(keras.losses.serialize(loss))
    assert restored.weights == [0.5, 2.0] and restored.scale == 4.0


def test_pinn_training_and_metrics(csv, tmp_path):
    prep, m, td = _pinn_data(csv)
    with patch("keras.callbacks.EarlyStopping", wraps=keras.callbacks.EarlyStopping) as early_stopping:
        model = m.pipeline(td, plot=False, save_model=False)
    assert early_stopping.call_args.kwargs["monitor"] == "val_data_loss"
    # the scale is the variance of the training targets only
    assert m._scale == pytest.approx(float(np.var(td.y_train[:, 0])), rel=1e-5)
    history = td.training_record.history
    assert {"data_loss", "dT_rc_loss", "rmse", "val_data_loss", "val_dT_rc_loss", "val_rmse"} <= set(history)

    assert model.outputs[0].shape[-1] == 1
    assert list(td.metrics.by_output) == ["Change(T)"]
    physics = td.metrics.physics["dT_rc"]
    assert set(physics) == {"train", "val", "test"}
    assert physics["test"]["normalized"] == pytest.approx(physics["test"]["mse"] / m._scale)
    json.dumps(td.get_config())

    # one output: exportable to AgentLib-MPC
    check_agentlib_compatibility(prep)
    run = Run(tmp_path, "pinn")
    run.save_for_agentlib("T", prep, m, td)
    config = json.loads((run.path / "T_preprocessing.json").read_text())
    assert config["output"] == ["Change(T)"] and config["loss_columns"] == ["y:dT_rc"]
    loaded = keras.saving.load_model(run.path / "T.keras")
    np.testing.assert_allclose(loaded.predict(td.X_test, verbose=0), model.predict(td.X_test, verbose=0), atol=1e-6)


def test_pinn_options_and_errors(csv):
    with pytest.raises(ValueError, match="physics targets"):
        PINNModel(physics_targets={})
    with pytest.raises(ValueError, match="early_stopping_on"):
        PINNModel(physics_targets={"a": 1}, early_stopping_on="physics")
    m = PINNModel(physics_targets={"dT_rc": 1.0}, normalize_terms=False, early_stopping_on="total")
    assert m._early_stopping_monitor() == "val_loss"
    # without the model, the preprocessing does not know the physics targets
    prep = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)", columns={"dT_rc": _physics_step()})
    with pytest.raises(ValueError, match="model=model"):
        m.pipeline(prep.pipeline(csv), plot=False, save_model=False)
    restored = PINNModel.from_config(json.loads(json.dumps(m.get_config())))
    assert restored.physics_targets == {"dT_rc": 1.0} and not restored.normalize_terms


@pytest.mark.parametrize("test_size, val_size", [(0.0, 0.2), (0.2, 0.0), (0.0, 0.0)])
def test_missing_splits(csv, tmp_path, test_size, val_size):
    """Models train, evaluate, plot and save without test or validation data (test_size=0, val_size=0)."""
    sizes = dict(test_size=test_size, val_size=val_size, random_state=0)
    single = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", eval_horizon=4,
                                     **sizes)
    v = px.Vars()
    models = {
        "linreg": LinearRegressionModel(),
        "ann": ClassicalANNModel(n_neurons=4, epochs=1),
        "modular": ModularANN(px.Linear([v.mDot * (v.T_in - v.T), v.load, v.T]), epochs=1,
                              recursive_training=RecursiveTraining(horizon=3, epochs=1)),
    }
    comparison = ModelComparison()
    run = Run(tmp_path, "run")
    for name, m in models.items():
        td = single.pipeline(csv, model=m if name == "modular" else None)
        assert (td.X_test is None) == (test_size == 0) and (td.X_val is None) == (val_size == 0)
        m.pipeline(td, plot=False, save_model=False)
        present = {"train"} | ({"val"} if val_size else set()) | ({"test"} if test_size else set())
        assert set(td.metrics.by_output["Change(T)"]) == present
        assert set(td.horizon_metrics.by_feature["T"]) <= present
        run.save(name, single, m, td)
        comparison.add(name, td)

    _, pinn, td = _pinn_data(csv, **{k: v for k, v in sizes.items() if k != "random_state"})
    pinn.pipeline(td, plot=False, save_model=False)
    assert set(td.metrics.physics["dT_rc"]) == set(td.metrics.by_output["Change(T)"])
    run.save("pinn", None, pinn, td)

    multi = PreprocessingMultiStep(inputs=["mDot", "load", "T_in"], output=px.Change("T"), label_width=3,
                                   warmup_width=1, **sizes)
    td = multi.pipeline(csv)
    RNNModel(rnn_units=2, epochs=1, early_stopping_epochs=None).pipeline(td, plot=False, save_model=False)
    assert ("test" in td.metrics.by_output["T"]) == bool(test_size)
    comparison.add("rnn", td)

    scores = comparison.scores("test")
    assert len(scores) == 4
    comparison.plot("train")
