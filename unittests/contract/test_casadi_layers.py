"""
Every keras layer physXAI places into a model that is exported to AgentLib-MPC has to be
translatable by the CasADi converter of AgentLib-MPC (`agentlib_mpc.models.casadi_predictor`).
The converter identifies layers by sub-strings of their names, so these tests also guard
the names and attributes of the custom layers.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("agentlib_mpc")
from agentlib_mpc.models.casadi_predictor import FunctionalWrapper  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402

from physXAI import models  # noqa: E402,F401  (registers the custom keras objects)
from physXAI.models.ann.keras_models.keras_models import (  # noqa: E402
    ConstantLayer,
    DivideLayer,
    InputSliceLayer,
    PowerLayer,
    RBFLayer,
)

pytestmark = pytest.mark.agentlib

LEGACY_MODEL = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "AgentLib-MPC"
    / "tests"
    / "test_physXAI_plugin"
    / "models"
    / "model.keras"
)


def _save_and_load(model: keras.Model, tmp_path: Path) -> keras.Model:
    """AgentLib-MPC always loads models from file. Only then do input tensors carry the input layer names,
    which the converter relies on."""
    path = tmp_path / "model.keras"
    model.save(path)
    return keras.saving.load_model(path)


def _assert_casadi_matches(model: keras.Model, x: np.ndarray, tmp_path: Path = None):
    if tmp_path is not None:
        model = _save_and_load(model, tmp_path)
    casadi_model = FunctionalWrapper(model)
    reference = model.predict(x, verbose=0).reshape(len(x), -1)
    translated = np.vstack([np.array(casadi_model.forward(row.reshape(1, -1))).reshape(1, -1) for row in x])
    np.testing.assert_allclose(translated, reference, rtol=1e-4, atol=1e-5)


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    return rng.uniform(0.5, 2.0, size=(8, 3)).astype("float32")


def test_custom_and_arithmetic_layers(x, tmp_path):
    inp = keras.Input(shape=(3,))
    a = InputSliceLayer([0])(inp)
    b = InputSliceLayer([1])(inp)
    c = InputSliceLayer([2])(inp)
    norm = keras.layers.Normalization()
    norm.adapt(x[:, :1])
    a_n = norm(a)
    k = ConstantLayer(value=0.7, trainable=True, weight_name="k")(inp)
    terms = [
        keras.layers.Add()([a_n, b]),
        keras.layers.Subtract()([b, c]),
        keras.layers.Multiply()([k, c]),
        DivideLayer()([a, b]),
        PowerLayer()([b, k]),
        keras.layers.Average()([a, c]),
    ]
    rbf = RBFLayer(units=4)(keras.layers.Concatenate()([a, b]))
    hidden = keras.layers.Dense(5, activation="softplus")(keras.layers.Concatenate()([*terms, rbf]))
    out = keras.layers.Dense(1)(hidden)
    out = keras.layers.Rescaling(scale=2.0, offset=0.5)(out)
    model = keras.Model(inp, out)
    _assert_casadi_matches(model, x, tmp_path)


def test_nested_sequential(x, tmp_path):
    inner = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Dense(4, activation="tanh"), keras.layers.Dense(1)])
    inp = keras.Input(shape=(3,))
    model = keras.Model(inp, inner(inp))
    _assert_casadi_matches(model, x, tmp_path)


@pytest.mark.skipif(not LEGACY_MODEL.exists(), reason="AgentLib-MPC test model not available")
def test_legacy_model_still_loads():
    """Models saved with physXAI < 1.0 have to keep loading (registered custom object names)."""
    model = keras.saving.load_model(LEGACY_MODEL)
    _assert_casadi_matches(model, np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 10.0]], dtype="float32"))
