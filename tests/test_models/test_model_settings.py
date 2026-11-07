"""Settings of the models: configs, seeds, unknown arguments, outputs, monotonicity and saved models."""

import logging

import numpy as np
import pytest

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import (
    ClassicalANNModel,
    CMNNModel,
    LinANNModel,
    LinearRegressionModel,
    ModularANN,
    PCNNModel,
    PINNModel,
    RBFModel,
    RNNModel,
)
from physXAI.models.models import AbstractModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

FAST = dict(epochs=1, early_stopping_epochs=None)
INPUTS = ["mDot", "load", "T_in", "T"]


@pytest.fixture
def td(one_room_csv):
    prep = PreprocessingSingleStep(inputs=INPUTS, output="Change(T)", csv_delimiter=",", random_state=0)
    return prep.pipeline(one_room_csv)


@pytest.fixture
def no_global_seed():
    yield
    physXAI.set_seed(None)


def _all_models():
    v = px.Vars()
    return [
        LinearRegressionModel(),
        ClassicalANNModel(n_neurons=4),
        LinANNModel(n_neurons=4),
        RBFModel(n_neurons=4),
        CMNNModel(monotonies={"mDot": 1}),
        PINNModel(physics_targets={"p": 0.5}),
        RNNModel(rnn_layer="GRU", step_weights=None),
        PCNNModel(heating={"u": 0.5}, losses={"T_amb": 0.01}),
        ModularANN(px.Linear([v.mDot, v.mDot * (v.T_in - v.T)]), solver="ols"),
    ]


@pytest.mark.parametrize("model", _all_models(), ids=lambda m: m.__class__.__name__)
def test_saved_config_creates_the_same_model(model):
    """A config as written by `Run.save` (with its meta keys) creates the model again."""
    saved = {**model.get_config(), **Run.version_info()}
    loaded = AbstractModel.model_from_config(saved)
    assert type(loaded) is type(model)
    assert loaded.get_config() == model.get_config()


@pytest.mark.parametrize(
    "create",
    [
        lambda: LinearRegressionModel(fit_intercept=False),
        lambda: ClassicalANNModel(hidden_layer_size=[4]),
        lambda: CMNNModel(monotonicities={"T": 1}),  # the argument is called 'monotonies'
        lambda: RNNModel(units=8),
        lambda: ModularANN(px.Linear(["T"]), epoch=3),
    ],
)
def test_unknown_arguments_are_rejected(create):
    with pytest.raises(TypeError, match="unknown arguments"):
        create()


def test_models_use_the_global_seed(no_global_seed):
    physXAI.set_seed(7)
    assert ClassicalANNModel().random_seed == 7
    assert RNNModel().random_seed == 7
    assert px.RBF(["T"]).random_state == 7
    assert ClassicalANNModel(random_seed=3).random_seed == 3  # an explicit seed takes precedence
    physXAI.set_seed(None)
    seeds = {ClassicalANNModel().random_seed for _ in range(3)}
    assert len(seeds) > 1  # without a global seed, every model draws (and stores) a new seed


def test_deterministic_operations():
    """`set_seed(..., deterministic=True)` enables the deterministic kernels of TensorFlow (in a new process, since
    TensorFlow cannot disable them again)."""
    import os
    import subprocess
    import sys

    code = (
        "import physXAI; physXAI.set_seed(3, deterministic=True)\n"
        "from tensorflow.python.framework import config\n"
        "assert config.is_op_determinism_enabled() and physXAI.get_seed() == 3\n"
    )
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, timeout=300)


def test_global_seed_makes_the_training_reproducible(td, no_global_seed):
    def weights(seed):
        physXAI.set_seed(seed)
        model = ClassicalANNModel(n_neurons=4, **FAST).pipeline(td, plot=False)
        return np.concatenate([w.ravel() for w in model.get_weights()])

    np.testing.assert_array_equal(weights(7), weights(7))
    assert not np.allclose(weights(7), weights(8))


def test_single_output_networks_reject_several_outputs(one_room_csv):
    prep = PreprocessingSingleStep(inputs=INPUTS, output=["Change(T)", "load"], csv_delimiter=",", random_state=0)
    td = prep.pipeline(one_room_csv)
    with pytest.raises(ValueError, match="predicts one output"):
        ClassicalANNModel(**FAST).pipeline(td, plot=False)
    # a linear regression predicts several outputs
    LinearRegressionModel().pipeline(td, plot=False)
    assert set(td.metrics.by_output) == {"Change(T)", "load"}


def _gradients(keras_model, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    x = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = keras_model(x)
    return tape.gradient(y, x).numpy()


def _test_points(td) -> np.ndarray:
    """Random points within and far outside the range of the training data."""
    X = td.X_train
    rng = np.random.default_rng(0)
    low, high = X.min(axis=0), X.max(axis=0)
    span = high - low
    return rng.uniform(low - 2 * span, high + 2 * span, size=(500, X.shape[1]))


def test_cmnn_is_monotonic(td):
    m = CMNNModel(monotonies={"load": 1, "T": -1}, n_neurons=6, epochs=3, early_stopping_epochs=None)
    model = m.pipeline(td, plot=False)
    gradients = _gradients(model, _test_points(td))
    assert np.all(gradients[:, INPUTS.index("load")] >= -1e-7)
    assert np.all(gradients[:, INPUTS.index("T")] <= 1e-7)


def test_monotone_mlp_block_is_monotonic(td):
    v = px.Vars()
    block = px.MonotoneMLP([v.mDot, v.load, v.T_in, v.T], monotonicities={"load": 1, "T": -1}, n_neurons=6)
    model = ModularANN(block, epochs=3, early_stopping_epochs=None).pipeline(td, plot=False)
    gradients = _gradients(model, _test_points(td))
    assert np.all(gradients[:, INPUTS.index("load")] >= -1e-7)
    assert np.all(gradients[:, INPUTS.index("T")] <= 1e-7)


@pytest.mark.parametrize(
    "model",
    [CMNNModel(monotonies={"load": 1}, n_neurons=6, **FAST), RBFModel(n_neurons=6, **FAST),
     LinANNModel(n_neurons=6, **FAST)],
    ids=lambda m: m.__class__.__name__,
)
def test_saved_model_predicts_the_same(model, td, tmp_path):
    trained = model.pipeline(td, plot=False)
    model.save_model(trained, tmp_path / "model")
    loaded = model.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(loaded.predict(td.X_test, verbose=0), trained.predict(td.X_test, verbose=0),
                               rtol=1e-6, atol=1e-7)
    # the results in td are those of the trained model (LinANN must not leave its linear regression in td)
    np.testing.assert_allclose(td.y_test_pred, trained.predict(td.X_test, verbose=0), rtol=1e-6, atol=1e-7)


def test_rbf_without_normalization(td):
    from physXAI.models.ann.model_construction.rbf_models import RBFModelConstruction

    model = RBFModelConstruction({"n_layers": 1, "n_neurons": 4, "normalize": False}, td)
    assert model.predict(td.X_test, verbose=0).shape == (len(td.X_test), 1)


def test_recurrent_export_reports_missing_agentlib_support(caplog, monkeypatch):
    from physXAI.io import agentlib
    from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

    monkeypatch.setattr(agentlib, "agentlib_supports_recurrent_models", lambda: False)
    prep = PreprocessingMultiStep(inputs=["mDot", "T"], output="Change(T)", label_width=3)
    with caplog.at_level(logging.WARNING, logger="physXAI"):
        agentlib.check_agentlib_compatibility(prep)
    assert agentlib.AGENTLIB_RNN_BRANCH in caplog.text


def test_derived_inputs_are_reported_for_agentlib(caplog):
    from physXAI.io import check_agentlib_compatibility

    v = px.Vars()
    prep = PreprocessingSingleStep(inputs=["mDot", "q", "T"], output="Change(T)", columns={"q": v.mDot * v.T})
    with caplog.at_level(logging.WARNING, logger="physXAI"):
        check_agentlib_compatibility(prep)
    assert "['q']" in caplog.text
