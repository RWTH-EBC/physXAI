"""
Contract of recurrent (multi step) models with AgentLib-MPC.

AgentLib-MPC (branch '143-feature-multistep-model-support') evaluates the saved step model of an RNN
one step at a time in CasADi, after warming its states up from zero on the measured past. These
tests run as soon as the installed AgentLib-MPC supports recurrent models and are skipped otherwise.
"""

import json

import numpy as np
import pytest
from conftest import import_agentlib, simulate_one_room

from physXAI.io.agentlib import AGENTLIB_RNN_BRANCH, agentlib_supports_recurrent_models

casadi_predictor = import_agentlib("agentlib_mpc.models.casadi_predictor")
if not agentlib_supports_recurrent_models():
    # allowed also in the CI until AgentLib-MPC releases the support of recurrent models
    pytest.skip(
        f"The installed AgentLib-MPC does not support recurrent models (branch {AGENTLIB_RNN_BRANCH}).",
        allow_module_level=True,
    )
import casadi as ca  # noqa: E402
import keras  # noqa: E402
from agentlib_mpc.machine_learning_plugins.physXAI.model_config_creation import physXAI_2_agentlib_json  # noqa: E402
from agentlib_mpc.models.serialized_ml_model import SerializedKerasRNN  # noqa: E402

from physXAI import expr as px  # noqa: E402
from physXAI.io import Run  # noqa: E402
from physXAI.models.ann.ann_design import PCNNModel, RNNModel  # noqa: E402
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep  # noqa: E402

pytestmark = pytest.mark.agentlib
WARMUP = 3


@pytest.mark.parametrize(
    "rnn_layer, pcnn",
    [("RNN", False), ("GRU", False), ("LSTM", False), ("LSTM", True), ("LSTM", "flow")],
)
def test_casadi_step_function_reproduces_the_rnn(rnn_layer, pcnn, tmp_path):
    csv = tmp_path / "one_room.csv"
    simulate_one_room(n_steps=800).to_csv(csv, sep=";")
    prep = PreprocessingMultiStep(
        inputs=["T_in", "load", "mDot", "T"], output=px.Change("T"), label_width=4, warmup_width=WARMUP, random_state=0
    )
    td = prep.pipeline(csv)
    settings = dict(rnn_units=4, rnn_layer=rnn_layer, epochs=1, early_stopping_epochs=None)
    # the PCNN step model has the energy accumulator as additional state
    # pcnn 'flow': the loss to T_in scaled by mDot (PCNNModel flows)
    flows = {"flows": {"T_in": "mDot"}} if pcnn == "flow" else {}
    m = PCNNModel(heating={"load": 0.03}, losses={"T_in": 0.5}, **flows, **settings) if pcnn else RNNModel(**settings)
    model = m.pipeline(td, plot=False, save_model=False)
    run = Run(tmp_path, "rnn")
    run.save_for_agentlib("T", prep, m, td, trained_model=model)

    with open(run.path / "T_preprocessing.json", encoding="utf-8") as f:
        config = physXAI_2_agentlib_json("run", json.load(f))
    assert config["model_type"] == "KerasRNN"
    assert config["warmup_steps"] == WARMUP and config["rnn_inputs"] == prep.inputs
    assert config["output"]["T"]["output_type"] == "difference"
    config.update(model_path=str(run.path / "T.keras"), dt=300)
    predictor = casadi_predictor.CasadiRNN(SerializedKerasRNN(**{k: v for k, v in config.items() if k != "model_type"}))

    # one sequence (warmup and prediction) of measured inputs, step by step from zero states
    sequence = td.X("test")[0]
    step_model = keras.saving.load_model(run.path / "T.keras")
    expected = np.asarray(
        step_model.predict(
            [sequence[None], *[np.zeros((1, int(t.shape[-1]))) for t in step_model.inputs[1:]]], verbose=0
        )[0]
    ).reshape(-1)
    states = ca.DM.zeros(predictor.state_dimension, 1)
    predictions = list()
    for row in sequence:
        prediction, states = predictor.predict(ca.DM(row), states)
        predictions.append(float(prediction))
    np.testing.assert_allclose(predictions, expected, atol=1e-4)
