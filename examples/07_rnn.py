"""
07 - Recurrent neural networks
==============================

A recurrent model carries its memory in hidden states instead of lagged inputs and predicts a trajectory.

- The RNN predicts the change of the air temperature. Its trajectory is the measured start value plus the
  accumulated changes, and the model is trained on that trajectory (`loss_on='trajectory'`).
- The air temperature is also an input: during the prediction the predicted temperature is fed back (closed loop),
  as in an MPC.
- There is no initialization model. Every sequence starts from zero states and warms up on the measured past
  (`warmup_width`), which is not part of the loss. AgentLib-MPC warms the model up the same way.
- `label_width` (training horizon) should be longer than the MPC horizon, `eval_horizon` shows the error beyond it.
- The saved `.keras` file is the step model `[sequence, *states] -> [prediction, *states]` used by AgentLib-MPC.
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

run = Run(ROOT / "results", "07_rnn", overwrite=True, log_level="warning")
physXAI.set_seed(0)

# Multi step data: sequences of warmup and prediction, each cut out of one split (no leakage). No lagged inputs.
prep = PreprocessingMultiStep(
    inputs=["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "oveHeaPumY_u", "reaTZon_y"],
    output=px.Change("reaTZon_y"),
    label_width="12h",  # predicted and trained horizon
    warmup_width="6h",  # measured past the model warms up on
    eval_horizon="24h",  # evaluation beyond the training horizon
    time_index_col="time",
)
td = prep.pipeline(DATA)

model = RNNModel(rnn_layer="LSTM", rnn_units=16, epochs=100, early_stopping_epochs=20)  # or 'GRU', 'RNN'
model.pipeline(td)
run.save("model", prep, model, td)

rmse = td.horizon_metrics.kpis["test"]["rmse"]
print(f"RNN test RMSE of T: {rmse[0]:.3f} K after 15 min, {rmse[47]:.3f} K after 12 h, {rmse[-1]:.3f} K after 24 h")
