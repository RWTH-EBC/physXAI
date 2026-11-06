"""
Contract tests between physXAI and AgentLib-MPC.

These tests run the real AgentLib-MPC entry point (`generate_physxai_model`) on physXAI
training scripts and check that the resulting model config can be loaded by AgentLib-MPC
and translated into a CasADi function that reproduces the trained physXAI model.

They guard the only interface that must stay stable across the physXAI 1.0.0 refactoring.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

agentlib_mpc = pytest.importorskip("agentlib_mpc")
model_generation = pytest.importorskip("agentlib_mpc.machine_learning_plugins.physXAI.model_generation")
from agentlib_mpc.models.casadi_predictor import CasadiPredictor  # noqa: E402
from agentlib_mpc.models.serialized_ml_model import SerializedMLModel  # noqa: E402

import physXAI.agentlib_mpc_plugin as physxai_plugin  # noqa: E402

pytestmark = pytest.mark.agentlib

SCRIPTS = Path(__file__).parent / "scripts"
# Keys of the agentlib config which AgentLib-MPC relies on
REQUIRED_KEYS = {"dt", "input", "output", "agentlib_mpc_hash", "training_info", "model_type"}


def _generate(models, scripts_path, csv, run_id="001"):
    return model_generation.generate_physxai_model(
        models=models, physXAI_scripts_path=str(scripts_path), training_data_path=csv, run_id=run_id, time_step=300
    )


def _casadi_vs_physxai(serialized: SerializedMLModel, x: np.ndarray, reference: np.ndarray):
    predictor = CasadiPredictor.from_serialized_model(serialized)
    casadi_prediction = np.array([float(predictor.predict(row.reshape(1, -1))) for row in x])
    np.testing.assert_allclose(casadi_prediction, reference.reshape(-1), rtol=1e-4, atol=1e-5)


def test_ann_example_ci(tmp_path, monkeypatch, one_room_csv):
    """The example shipped with physXAI (used in the AgentLib-MPC CI) yields a valid KerasANN config."""
    monkeypatch.chdir(tmp_path)
    files = _generate(["example_ci"], os.path.dirname(physxai_plugin.__file__), one_room_csv)

    assert len(files) == 1
    config = json.loads(Path(files[0]).read_text())
    assert REQUIRED_KEYS <= config.keys()
    assert config["model_type"] == "KerasANN"
    assert config["dt"] == 300
    assert config["input"] == {"mDot": {"name": "mDot", "lag": 1}}
    assert config["output"] == {"T": {"name": "T", "lag": 1, "output_type": "difference", "recursive": True}}
    # intermediate physXAI files are cleaned up by AgentLib-MPC, only the model remains
    run_dir = Path(files[0]).parent
    assert (run_dir / "T.keras").exists()
    assert not (run_dir / "T_preprocessing.json").exists()

    serialized = SerializedMLModel.load_serialized_model(Path(files[0]))
    keras_model = serialized.deserialize()
    x = np.array([[0.01, 293.0], [0.04, 291.5], [0.0, 295.0]], dtype=np.float32)
    _casadi_vs_physxai(serialized, x, keras_model.predict(x, verbose=0))


def test_linreg_change_with_derived_input(tmp_path, monkeypatch, one_room_csv):
    """A linear regression on Change(T) is exported as LinReg with its coefficients."""
    monkeypatch.chdir(tmp_path)
    files = _generate({"T": "linreg_change"}, SCRIPTS, one_room_csv)

    config = json.loads(Path(files[0]).read_text())
    assert REQUIRED_KEYS <= config.keys()
    assert config["model_type"] == "LinReg"
    assert list(config["input"]) == ["mDot", "load", "vT"]
    assert all(feature["lag"] == 1 for feature in config["input"].values())
    assert config["output"]["T"]["output_type"] == "difference"
    assert config["output"]["T"]["recursive"] is True
    assert config["training_info"]["preprocessing"].keys() == {"test_size", "val_size", "random_state"}
    assert len(config["parameters"]["coef"][0]) == 4

    serialized = SerializedMLModel.load_serialized_model(Path(files[0]))
    linear_model = serialized.deserialize()
    x = np.array([[0.01, 100.0, -0.03, 293.0], [0.04, 0.0, -0.1, 291.5]])
    _casadi_vs_physxai(serialized, x, linear_model.predict(x))


def test_derived_input_inside_the_model(tmp_path, monkeypatch, one_room_csv):
    """With the derived input inside the model, AgentLib-MPC only sees raw inputs (KerasANN)."""
    monkeypatch.chdir(tmp_path)
    files = _generate({"T": "modular_linreg"}, SCRIPTS, one_room_csv)

    config = json.loads(Path(files[0]).read_text())
    assert config["model_type"] == "KerasANN"
    assert list(config["input"]) == ["mDot", "load", "T_in"]
    assert config["output"] == {"T": {"name": "T", "lag": 1, "output_type": "difference", "recursive": True}}

    serialized = SerializedMLModel.load_serialized_model(Path(files[0]))
    keras_model = serialized.deserialize()
    x = np.array([[0.01, 100.0, 290.0, 293.0], [0.04, 0.0, 291.0, 291.5]], dtype=np.float32)
    prediction = keras_model.predict(x, verbose=0)
    _casadi_vs_physxai(serialized, x, prediction)
    # the simulated system is linear in these features: dT = dt/C * (cp * mDot * (T_in - T) + load)
    m_dot, load, t_in, t = x.T
    expected = 300 / 1e5 * (1000 * m_dot * (t_in - t) + load)
    np.testing.assert_allclose(prediction.reshape(-1), expected, rtol=1e-3, atol=1e-4)


def test_existing_models_are_copied(tmp_path, monkeypatch, one_room_csv):
    """A string as model spec reuses an existing run folder."""
    monkeypatch.chdir(tmp_path)
    _generate({"T": "linreg_change"}, SCRIPTS, one_room_csv, run_id="001")
    files = model_generation.generate_physxai_model(
        models="001", physXAI_scripts_path=str(SCRIPTS), training_data_path=one_room_csv, run_id="002"
    )
    assert [Path(f).name for f in files] == ["T.json"]
