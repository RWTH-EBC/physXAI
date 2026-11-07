"""Recurrent (multi step) models: warmup from a zero state, change outputs, closed loop and export."""

import json
import pickle

import keras
import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.io import Run
from physXAI.io.agentlib import check_agentlib_compatibility
from physXAI.models import LinearRegressionModel
from physXAI.models.ann.ann_design import RNNModel
from physXAI.models.models import AbstractModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep

WARMUP, LABEL = 3, 4


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


def _data(csv, inputs, output=None, **kwargs):
    kwargs = {"label_width": LABEL, "warmup_width": WARMUP, "random_state": 0, **kwargs}
    prep = PreprocessingMultiStep(inputs=inputs, output=output or px.Change("T"), **kwargs)
    return prep, prep.pipeline(csv)


def _train(td, **kwargs):
    kwargs = {"rnn_units": 4, "epochs": 1, "early_stopping_epochs": None, **kwargs}
    m = RNNModel(**kwargs)
    return m, m.pipeline(td, plot=False, save_model=False)


def _mpc_rollout(step_model, X, initial, feedback_column=None, difference=True):
    """
    Applies the step model as AgentLib-MPC does: zero states, warmup on the measured rows, then one
    step at a time, feeding the predicted feature back if it is an input.
    """
    units = [int(t.shape[-1]) for t in step_model.inputs[1:]]
    states = [np.zeros((len(X), u), dtype="float32") for u in units]
    value, trajectory = initial.copy(), list()
    for k in range(X.shape[1]):
        row = X[:, k : k + 1, :].copy()
        predicting = k >= WARMUP
        if predicting and feedback_column is not None:
            row[:, 0, feedback_column] = value[:, 0]
        prediction, *states = step_model([row, *states], training=False)
        states = [np.asarray(s) for s in states]
        if predicting:
            prediction = np.asarray(prediction)[:, 0, :]
            value = value + prediction if difference else prediction
            trajectory.append(value)
    return np.stack(trajectory, axis=1)


@pytest.mark.parametrize("rnn_layer", ["RNN", "GRU", "LSTM"])
def test_trajectory_equals_mpc_rollout(csv, rnn_layer, tmp_path):
    """The prediction of the evaluation is what the saved step model gives in an MPC."""
    _, td = _data(csv, ["mDot", "load", "T_in"])
    m, model = _train(td, rnn_layer=rnn_layer)

    path = str(tmp_path / "model.keras")
    m.save_model(model, path)
    step_model = keras.saving.load_model(path)
    assert step_model.name == "rnn_step"
    # signature expected by AgentLib-MPC: [sequence of any length, *states] -> [prediction, *states]
    assert step_model.inputs[0].shape[1] is None
    assert len(step_model.inputs) == len(step_model.outputs) == (3 if rnn_layer == "LSTM" else 2)

    X, initial = td.X("test")[:5], td.initial("test")[:5]
    expected = _mpc_rollout(step_model, X, initial)
    np.testing.assert_allclose(td.y_test_pred[:5], expected, rtol=1e-4, atol=1e-4)
    assert td.y_test.shape == td.y_test_pred.shape == (len(td.X_test), LABEL, 1)


def test_closed_loop_feeds_the_prediction_back(csv):
    """If the predicted feature is an input, the prediction replaces the measurement (as in an MPC)."""
    _, td = _data(csv, ["mDot", "load", "T_in", "T"])
    m, model = _train(td, rnn_layer="GRU")
    step_model = model.get_layer("rnn_step")
    X, initial = td.X("test")[:5], td.initial("test")[:5]

    expected = _mpc_rollout(step_model, X, initial, feedback_column=3)
    np.testing.assert_allclose(td.y_test_pred[:5], expected, rtol=1e-4, atol=1e-4)
    # with the measured values (teacher forcing) the result differs
    teacher_forced = _mpc_rollout(step_model, X, initial)
    assert not np.allclose(teacher_forced, expected, atol=1e-6)


def test_absolute_output_with_feedback(csv):
    _, td = _data(csv, ["mDot", "load", "T_in", "T"], output="T")
    assert td.difference == [False]
    m, model = _train(td, rnn_layer="GRU")
    expected = _mpc_rollout(
        model.get_layer("rnn_step"), td.X("test")[:3], td.initial("test")[:3], feedback_column=3, difference=False
    )
    np.testing.assert_allclose(td.y_test_pred[:3], expected, rtol=1e-4, atol=1e-4)


def test_training_reduces_the_trajectory_error(csv):
    _, td = _data(csv, ["mDot", "load", "T_in"])
    _train(td, rnn_layer="GRU", epochs=5, learning_rate=0.01, first_step_weight=0.5)
    history = td.training_record.history
    assert history["loss"][-1] < history["loss"][0]
    assert "val_loss" in history


def test_loss_on_outputs(csv):
    _, td = _data(csv, ["mDot", "load", "T_in"])
    m, model = _train(td, loss_on="outputs")
    # trained on the changes, the model still predicts and is evaluated on trajectories
    assert model.outputs[0].shape[1:] == (LABEL, 1)
    assert td.y_test_pred.shape == td.y_test.shape
    with pytest.raises(ValueError, match="loss_on"):
        RNNModel(loss_on="state")


def test_evaluation_horizon_beyond_training(csv):
    _, td = _data(csv, ["mDot", "load", "T_in"], eval_horizon=10, eval_stride=5)
    _train(td)
    kpis = td.horizon_metrics.kpis
    assert len(kpis["test"]["rmse"]) == 10 and len(kpis["test"]["bias"]) == 10
    assert td.horizon_metrics.hours[-1] == pytest.approx(10 * 300 / 3600)
    config = json.loads(json.dumps(td.get_config()))
    assert config["horizon_metrics"]["feature"] == "T"
    assert config["summary"]["sequences"]["train"] == len(td.X_train)
    pickle.dumps(td)


def test_feature_architecture(csv):
    """Expressions in front of the recurrent layer are part of the step model, also in the closed loop."""
    v = px.Vars()
    heat_flow = (v.mDot * (v.T_in - v.T)).named("heat_flow")
    architecture = [px.Normalize(v.load), px.Normalize(heat_flow)]
    _, td = _data(csv, ["mDot", "load", "T_in", "T"])
    m, model = _train(td, rnn_layer="GRU", feature_architecture=architecture)
    layers = [layer.name for layer in model.get_layer("rnn_step").layers]
    assert any("multiply" in name for name in layers), layers
    assert any("subtract" in name for name in layers), layers
    assert m.required_inputs == ["load", "mDot", "T_in", "T"]

    restored = AbstractModel.model_from_config(json.loads(json.dumps(m.get_config())))
    assert restored.feature_architecture[-1].inputs[0].name == "heat_flow"


def test_online_pipeline(csv, tmp_path):
    _, td = _data(csv, ["mDot", "load", "T_in"])
    m, model = _train(td)
    path = str(tmp_path / "rnn.keras")
    m.save_model(model, path)
    weights = model.get_layer("rnn_step").get_weights()
    updated = m.online_pipeline(td, path, plot=True, save_model=False)
    assert any(not np.allclose(a, b) for a, b in zip(weights, updated.get_layer("rnn_step").get_weights(), strict=True))


def test_export_for_agentlib(csv, tmp_path):
    prep, td = _data(csv, ["mDot", "load", "T_in", "T"])
    m, model = _train(td, rnn_layer="LSTM")
    check_agentlib_compatibility(prep)
    run = Run(tmp_path, "rnn")
    assert run.save_for_agentlib("T", prep, m, td, trained_model=model) == "T"

    with open(run.path / "T_preprocessing.json", encoding="utf-8") as f:
        config = json.load(f)
    assert config["__class_name__"] == "PreprocessingMultiStep"
    assert config["warmup_width"] == WARMUP and config["output"] == ["Change(T)"]
    assert "init_features" not in config
    step_model = keras.saving.load_model(run.path / "T.keras")
    assert step_model.name == "rnn_step"
    assert [tuple(t.shape) for t in step_model.outputs] == [(None, None, 1), (None, 4), (None, 4)]
    # AgentLib-MPC translates the layers to CasADi by their names
    supported = ("input", "normalization", "lstm", "gru", "simple_rnn", "dense", "rescaling")
    assert all(any(s in layer.name for s in supported) for layer in step_model.layers), step_model.layers


def test_wrong_models_and_data(csv):
    _, td = _data(csv, ["mDot", "load", "T_in"])
    with pytest.raises(TypeError, match="eval_horizon"):
        LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    single = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)").pipeline(csv)
    with pytest.raises(TypeError, match="PreprocessingMultiStep"):
        RNNModel(epochs=1).pipeline(single, plot=False, save_model=False)


@pytest.mark.parametrize("inputs, columns", [
    (["mDot", "load", "T_in", "T_lag1"], None),
    (["mDot", "load", "T_in", "T", "Change(T)"], None),
    (["mDot", "load", "dT", "T"], "dT"),
])
def test_inputs_holding_the_predicted_feature_are_rejected(csv, inputs, columns):
    """Regression: lags, changes and derived columns of a predicted feature would leak its measured values."""
    kwargs = {"columns": {"dT": px.Var("T_in") - px.Var("T")}} if columns else {}
    with pytest.raises(ValueError, match="leakage"):
        _data(csv, inputs, **kwargs)


def test_lags_of_exogenous_inputs_are_allowed(csv):
    _, td = _data(csv, ["mDot", "load", "load_lag1", "T_in", "T"])
    assert "load_lag1" in td.inputs
