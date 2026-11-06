"""Recursive (multi step) evaluation and training of single step models."""

import json

import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.evaluation.rollout import evaluate_horizon, recursion_spec, rollout_data
from physXAI.models import AbstractModel, ClassicalANNModel, LinearRegressionModel, ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.training import RecursiveTraining

HORIZON = 12


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=2000).to_csv(path, sep=";")
    return str(path)


def _exact_linear(csv, horizon=HORIZON):
    v = px.Vars()
    m = ModularANN(px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T]), solver="ols")
    td = PreprocessingSingleStep(output=px.Change("T"), eval_horizon=horizon, random_state=0).pipeline(csv, model=m)
    model = m.pipeline(td, plot=False, save_model=False)
    return m, model, td


def test_exact_model_has_no_recursive_error(csv):
    _, _, td = _exact_linear(csv)
    for split in ("train", "val", "test"):
        kpis = td.horizon_metrics.kpis[split]
        assert len(kpis["rmse"]) == HORIZON
        assert max(kpis["rmse"]) < 1e-3
        assert max(np.abs(kpis["bias"])) < 1e-3
    config = json.loads(json.dumps(td.get_config()))
    assert len(config["horizon_metrics"]["test"]["bias"]) == HORIZON


def test_truth_is_the_measured_future(csv):
    _, _, td = _exact_linear(csv)
    spec = recursion_spec(td)
    rows, history, truth = rollout_data(td, spec, "test", HORIZON)
    start = td.window_starts("test", HORIZON)[0]
    T = td.column("T")
    np.testing.assert_allclose(truth[0, :, 0], T[start + 1 : start + 1 + HORIZON], rtol=1e-6)
    np.testing.assert_allclose(history[0, 0], T[start], rtol=1e-6)


def test_derived_columns_are_recomputed(csv):
    """A data column depending on the recursive state (vT) is recomputed in every step."""
    v = px.Vars()
    prep = PreprocessingSingleStep(
        inputs=["mDot", "load", "vT", "T"],
        output="Change(T)",
        columns={"vT": v.mDot * (v.T_in - v.T)},
        eval_horizon=HORIZON,
        random_state=0,
    )
    td = prep.pipeline(csv)
    model = LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    spec = recursion_spec(td)
    assert spec.derived == ["vT"]
    assert spec.columns == ["mDot", "load", "vT", "T", "T_in"]
    assert max(td.horizon_metrics.kpis["test"]["rmse"]) < 1e-3

    # reference: recursion by hand with a perturbed model, recomputing vT from the predicted T
    model.intercept_ = model.intercept_ + 0.02
    evaluate_horizon(model, td)
    prediction, _ = td.horizon_predictions["test"]
    rows, history, _ = rollout_data(td, spec, "test", HORIZON)
    m_dot, load, _, _, t_in = rows[0].T
    T, expected = history[0, 0], []
    for k in range(HORIZON):
        x = np.array([[m_dot[k], load[k], m_dot[k] * (t_in[k] - T), T]])
        T = T + model.predict(x).item()
        expected.append(T)
    np.testing.assert_allclose(prediction[0, :, 0], expected, rtol=1e-6)


def test_offset_shows_up_as_bias(csv):
    m, model, td = _exact_linear(csv)
    dense = model.layers[-1]
    kernel, bias = dense.get_weights()
    dense.set_weights([kernel, bias + 0.01])
    metrics = evaluate_horizon(model, td)
    test_bias = np.array(metrics.kpis["test"]["bias"])
    assert test_bias[0] == pytest.approx(0.01, abs=1e-3)
    assert np.all(np.diff(test_bias) > 0)  # the bias accumulates over the horizon
    assert test_bias[-1] > 5 * test_bias[0]


def test_lags_and_absolute_outputs(csv):
    prep = PreprocessingSingleStep(
        inputs=["mDot", "load", "T_in", "T"], lags={"T": 2}, output="T", eval_horizon=HORIZON, random_state=0
    )
    td = prep.pipeline(csv)
    LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    spec = recursion_spec(td)
    assert spec.max_lag == 2
    assert not spec.difference
    assert td.horizon_metrics.kpis["test"]["rmse"][-1] < 0.5


def test_no_recursion_without_feedback(csv):
    td = PreprocessingSingleStep(inputs=["mDot", "load"], output="T_in", eval_horizon=HORIZON).pipeline(csv)
    LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    assert recursion_spec(td) is None
    assert td.horizon_metrics is None


def test_step_weights():
    assert np.allclose(RecursiveTraining(first_step_weight=0.25).weights(4), [0.25, 0.25, 0.25, 0.25])
    weights = RecursiveTraining(first_step_weight=0.5).weights(6)
    assert weights[0] == pytest.approx(0.5)
    assert np.allclose(weights[1:], 0.1)
    assert np.allclose(RecursiveTraining(step_weights=[2, 1, 1]).weights(3), [0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        RecursiveTraining(step_weights=[1, 1]).weights(3)
    with pytest.raises(ValueError):
        RecursiveTraining(mode="other")


@pytest.mark.parametrize("mode", ["single_then_multi", "multi_step"])
def test_recursive_training(csv, mode, tmp_path):
    rt = RecursiveTraining(horizon=6, epochs=2, mode=mode, stride=3)
    m = ClassicalANNModel(n_neurons=4, epochs=2, early_stopping_epochs=None, recursive_training=rt)
    td = PreprocessingSingleStep(
        inputs=["mDot", "load", "T_in", "T"], output="Change(T)", eval_horizon=6, random_state=0
    ).pipeline(csv)
    model = m.pipeline(td, plot=True, save_model=False)
    assert td.recursive_training_record is not None
    assert len(td.recursive_training_record.history["loss"]) == 2
    # the trained model stays a single step model, as AgentLib-MPC expects it
    assert tuple(model.input_shape) == (None, 4)

    config = json.loads(json.dumps(m.get_config()))
    assert config["recursive_training"]["mode"] == mode
    restored = AbstractModel.model_from_config(config)
    assert restored.recursive_training.horizon == 6


def test_ols_initializes_recursive_training(csv):
    v = px.Vars()
    rt = RecursiveTraining(horizon=4, epochs=1, stride=4, learning_rate=1e-7)
    m = ModularANN(
        px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T]),
        solver="ols",
        recursive_training=rt,
        early_stopping_epochs=None,
    )
    td = PreprocessingSingleStep(output=px.Change("T"), eval_horizon=4, random_state=0).pipeline(csv, model=m)
    m.pipeline(td, plot=False, save_model=False)
    assert max(td.horizon_metrics.kpis["test"]["rmse"]) < 0.05


def test_multi_step_training_needs_feedback(csv):
    m = ClassicalANNModel(n_neurons=4, epochs=1, recursive_training=RecursiveTraining(horizon=4))
    td = PreprocessingSingleStep(inputs=["mDot", "load"], output="T_in").pipeline(csv)
    with pytest.raises(ValueError, match="predicts one of its own inputs"):
        m.pipeline(td, plot=False, save_model=False)
