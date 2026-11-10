"""Physically consistent neural network (PCNN, Di Natale et al. 2022)."""

import json

import keras
import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.io import Run
from physXAI.io.agentlib import check_agentlib_compatibility
from physXAI.models.ann.ann_design import PCNNModel
from physXAI.models.models import AbstractModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

WARMUP, LABEL = 3, 6
INPUTS = ["load", "T_in", "mDot", "T"]  # T is fed back
#: CasADi translation of AgentLib-MPC recognizes the layer types by these parts of their names
SUPPORTED = (
    "input",
    "dense",
    "normalization",
    "concatenate",
    "constant",
    "add",
    "subtract",
    "multiply",
    "simple_rnn",
    "lstm",
    "gru",
    "rescaling",
)


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1200).to_csv(path, sep=";")
    return str(path)


@pytest.fixture(scope="module")
def data(csv):
    prep = PreprocessingMultiStep(
        inputs=INPUTS, output=px.Change("T"), label_width=LABEL, warmup_width=WARMUP, random_state=0
    )
    return prep, prep.pipeline(csv)


def _train(td, rnn_layer="LSTM", **kwargs):
    # heating by the internal load (K/h per W), losses to the supply temperature (1/h); mDot is a disturbance of f
    kwargs = {
        "heating": {"load": 0.03},
        "losses": {"T_in": 0.5},
        "rnn_units": 4,
        "rnn_layer": rnn_layer,
        "epochs": 1,
        "early_stopping_epochs": None,
        **kwargs,
    }
    m = PCNNModel(**kwargs)
    return m, m.pipeline(td, plot=False, save_model=False)


def _rollout(step_model, X, initial):
    """Step by step from zero states, warmup on the measurements, then the predicted T fed back (as in an MPC)."""
    states = [np.zeros((len(X), int(t.shape[-1])), dtype="float32") for t in step_model.inputs[1:]]
    value, trajectory = initial.copy(), list()
    for k in range(X.shape[1]):
        row = X[:, k : k + 1, :].copy()
        if k >= WARMUP:
            row[:, 0, INPUTS.index("T")] = value[:, 0]
        prediction, *states = step_model([row, *states], training=False)
        states = [np.asarray(s) for s in states]
        if k >= WARMUP:
            value = value + np.asarray(prediction)[:, 0, :]
            trajectory.append(value)
    return np.stack(trajectory, axis=1)


@pytest.mark.parametrize("rnn_layer", ["RNN", "GRU", "LSTM"])
def test_step_model_and_closed_loop(data, rnn_layer, tmp_path):
    _, td = data
    m, model = _train(td, rnn_layer)
    path = str(tmp_path / "pcnn.keras")
    m.save_model(model, path)
    step_model = keras.saving.load_model(path)
    assert step_model.name == "rnn_step"
    # states: energy accumulator E (1) and the states of f
    assert [int(t.shape[-1]) for t in step_model.inputs[1:]] == [1] + [4] * (2 if rnn_layer == "LSTM" else 1)
    assert all(any(s in layer.name for s in SUPPORTED) for layer in step_model.layers), [
        layer.name for layer in step_model.layers
    ]
    X, initial = td.X("test")[:4], td.initial("test")[:4]
    np.testing.assert_allclose(td.y_test_pred[:4], _rollout(step_model, X, initial), rtol=1e-4, atol=1e-4)


def test_physical_consistency(data):
    """More heating or a warmer T_x raises T at all later steps, more cooling lowers it, for any network f."""
    _, td = data
    m, model = _train(td, cooling={"mDot": 1.0}, losses={"T_in": 0.5}, heating={"load": 0.03})
    step_model = model.get_layer("rnn_step")
    X, initial = td.X("test")[:8].copy(), td.initial("test")[:8]
    base = _rollout(step_model, X, initial)
    j = WARMUP + 1  # change of the input at the second predicted step
    for name, delta, sign in (("load", 50.0, 1), ("T_in", 1.0, 1), ("mDot", 0.01, -1)):
        changed = X.copy()
        changed[:, j, INPUTS.index(name)] += delta
        difference = sign * (_rollout(step_model, changed, initial) - base)
        np.testing.assert_allclose(difference[:, : j - WARMUP], 0, atol=1e-5)  # no effect on earlier steps
        assert (difference[:, j - WARMUP :] > -1e-5).all(), name
        assert (difference[:, -1] > 1e-6).all(), name  # and an effect until the end of the horizon


def test_parameters_and_config(data, tmp_path):
    prep, td = data
    m, model = _train(td, epochs=3, learning_rate=0.01)
    parameters = td.physics["parameters"]
    assert set(parameters["heating"]) == {"load"} and set(parameters["losses"]) == {"T_in"}
    assert parameters["heating"]["load"] > 0 and parameters["losses"]["T_in"] > 0
    assert 0 < parameters["consistency_margin"] < 1
    json.dumps(td.get_config())

    config = json.loads(json.dumps(m.get_config()))
    restored = AbstractModel.model_from_config(config)
    assert isinstance(restored, PCNNModel) and restored.losses == {"T_in": 0.5}

    check_agentlib_compatibility(prep)
    run = Run(tmp_path, "pcnn")
    files = run.save("T", prep, m, td)
    assert files["model"].name == "T.keras" and files["plot"].exists()
    assert json.loads(files["training_data"].read_text())["physics"]["parameters"]["heating"]["load"] > 0


def test_parameters_are_read_from_the_model_after_online_training(csv, data, tmp_path):
    """
    Regression: after online_pipeline on other data, the parameters were reported with the initial rates estimated
    from the new data instead of the initial rates built into the loaded model.
    """
    _, td = data
    m, _ = _train(td, heating={"load": None}, losses={"T_in": None})
    m.save_model(m.trained_model, str(tmp_path / "pcnn"))
    initial = dict(td.physics["initial_rates"])

    # other data: a third of the load gives the same temperatures, so the heating rate estimated from them triples
    other = simulate_one_room(n_steps=1200)
    other["load"] = other["load"] / 3
    other.to_csv(tmp_path / "other.csv", sep=";")
    td_other = PreprocessingMultiStep(
        inputs=INPUTS, output=px.Change("T"), label_width=LABEL, warmup_width=WARMUP, random_state=0
    ).pipeline(str(tmp_path / "other.csv"))
    online = PCNNModel(heating={"load": None}, losses={"T_in": None}, rnn_units=4, epochs=1, early_stopping_epochs=None)
    assert online.initial_rates(td_other)["load"] == pytest.approx(3 * initial["load"], rel=0.1)
    model = online.online_pipeline(td_other, str(tmp_path / "pcnn.keras"), plot=False, save_model=False)

    from physXAI.models.ann.model_construction.rnn_models import PCNN_PHYSICS_LAYER, find_step_model

    step = find_step_model(model)
    factors = step.get_layer(PCNN_PHYSICS_LAYER).get_weights()[0][:, 0]
    hours = td_other.time_step / 3600
    assert td_other.physics["initial_rates"] == pytest.approx(initial, rel=1e-5)  # those of the loaded model
    parameters = td_other.physics["parameters"]
    assert parameters["heating"]["load"] == pytest.approx(factors[0] * initial["load"], rel=1e-5)
    assert parameters["losses"]["T_in"] == pytest.approx(factors[1] * initial["T_in"], rel=1e-5)
    assert parameters["consistency_margin"] == pytest.approx(1 - factors[1] * initial["T_in"] * hours, rel=1e-5)


def test_invalid_configurations(csv, data):
    _, td = data
    with pytest.raises(ValueError, match="at least one term"):
        PCNNModel()
    with pytest.raises(ValueError, match="positive"):
        PCNNModel(heating={"load": -1})
    with pytest.raises(ValueError, match="one term"):
        PCNNModel(heating={"load": 1}, losses={"load": 1})
    with pytest.raises(ValueError, match="feature_architecture"):
        PCNNModel(heating={"load": 1}, feature_architecture=px.Var("load"))
    with pytest.raises(ValueError, match="not an input"):
        _train(td, heating={"unknown": 1.0})
    with pytest.raises(ValueError, match="cannot be a"):
        _train(td, losses={"T": 1.0})
    open_loop = PreprocessingMultiStep(inputs=["load", "T_in"], output=px.Change("T"), label_width=3).pipeline(csv)
    with pytest.raises(ValueError, match="has to be an input"):
        _train(open_loop)
    absolute = PreprocessingMultiStep(inputs=INPUTS, output="T", label_width=3).pipeline(csv)
    with pytest.raises(ValueError, match="change of one feature"):
        _train(absolute)


def test_initial_rates_from_the_data(data, caplog):
    """Rates given as None are estimated from the training steps: the load heats by dt / C = 0.003 K/W per step."""
    _, td = data
    m, _ = _train(td, heating={"load": None}, losses={"T_in": None})
    rates = td.physics["initial_rates"]
    assert rates["load"] == pytest.approx(300.0 / 1e5 * 12, rel=0.05)  # 0.036 K/h per W
    assert rates["T_in"] > 0
    config = json.loads(json.dumps(m.get_config()))
    assert config["heating"] == {"load": None}
    assert AbstractModel.model_from_config(config).losses == {"T_in": None}

    # mDot cools the room (T_in < T): no positive heating effect, the term starts at a weak rate with a warning
    caplog.clear()
    m = PCNNModel(heating={"mDot": None}, losses={"T_in": 0.5}, rnn_units=4, epochs=1, early_stopping_epochs=None)
    assert m.initial_rates(td)["mDot"] > 0
    assert "no positive effect of 'mDot'" in caplog.text


def test_loss_scaled_by_a_flow(data, tmp_path):
    """
    flows: the loss to T_in is scaled by mDot, b * mDot * (T_in - T), the true structure of the data. The estimated
    rate is dt * cp / C = 3 per step and kg/s (36 per hour), mDot leaves f, the step model is closed loop exact.
    """
    _, td = data
    m, model = _train(td, heating={"load": None}, losses={"T_in": None}, flows={"T_in": "mDot"})
    assert td.physics["initial_rates"]["T_in"] == pytest.approx(300.0 / 1e5 * 1000.0 * 12, rel=0.05)
    step_model = model.get_layer("rnn_step")
    names = [layer.name for layer in step_model.layers]
    assert "multiply_pcnn_flow_1" in names and all(any(s in n for s in SUPPORTED) for n in names), names
    # f sees only T: the normalization covers D alone, mDot is part of the physics
    assert step_model.get_layer("normalization_pcnn").mean.shape[-1] == 1
    X, initial = td.X("test")[:4], td.initial("test")[:4]
    np.testing.assert_allclose(td.y_test_pred[:4], _rollout(step_model, X, initial), rtol=1e-4, atol=1e-4)

    # consistency margin with the largest flow of the training data
    parameters = td.physics["parameters"]
    per_step = parameters["losses"]["T_in"] * td.time_step / 3600
    assert parameters["consistency_margin"] == pytest.approx(1 - per_step * td.X_train_single[:, 2].max(), rel=1e-5)

    config = json.loads(json.dumps(m.get_config()))
    assert AbstractModel.model_from_config(config).flows == {"T_in": "mDot"}
    assert "flows" not in PCNNModel(losses={"T_in": None}).get_config()  # unchanged configs without the option

    with pytest.raises(ValueError, match="flows scale losses"):
        PCNNModel(heating={"load": 1}, flows={"load": "mDot"})
    with pytest.raises(ValueError, match="one term"):
        PCNNModel(heating={"mDot": 1}, losses={"T_in": 1}, flows={"T_in": "mDot"})
    with pytest.raises(ValueError, match="not an input"):
        _train(td, losses={"T_in": 1.0}, flows={"T_in": "unknown"})


def test_estimated_rates_are_not_taken_from_another_series(csv):
    """Regression: the cache of the estimated initial rates belongs to one series object (not to a reused id)."""
    import pickle

    m = PCNNModel(heating={"load": None}, losses={"T_in": None})
    first = PreprocessingMultiStep(
        inputs=INPUTS, output=px.Change("T"), label_width=LABEL, warmup_width=WARMUP, random_state=0
    ).pipeline(csv)
    rates = m.initial_rates(first)
    m._estimated = (m._estimated[0], {k: 123.0 for k in rates})  # mark the cache
    assert m.initial_rates(first)["load"] == 123.0
    second = PreprocessingMultiStep(
        inputs=INPUTS, output=px.Change("T"), label_width=LABEL, warmup_width=WARMUP, random_state=5
    ).pipeline(csv)
    assert m.initial_rates(second)["load"] != 123.0
    assert pickle.loads(pickle.dumps(m))._estimated is None
