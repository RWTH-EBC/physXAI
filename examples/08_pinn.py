"""
08 - Physics-informed neural network (PINN)
===========================================

A simplified physical model of the heat pump is a physics target: the network is trained on the data and, with a
weight, on the agreement with the physical model. This regularizes the network where the data is sparse.

- The model has one output, the electrical power, and is used like any other model (also in AgentLib-MPC).
- The physics target is an expression of data columns, added to the data via `columns`. Like the output, it is
  taken at the predicted time.
- Loss: mse(y_pred, y) / s + sum_i w_i * mse(y_pred, p_i) / s with s = var(y_train). The weight w_i is the
  relative importance of the physics, chosen by validation (see example 11).
- Early stopping monitors the validation error of the data, history and report show every term of the loss.

Example 09 adds physics with trainable parameters, example 10 a model whose physics holds by construction.
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import PINNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

run = Run(ROOT / "results", "08_pinn", overwrite=True, log_level="warning")
physXAI.set_seed(0)

# Physical model of the power: compressor power with a Carnot like efficiency plus auxiliaries when running
v = px.Vars()
u_hp, u_hp_on = v.oveHeaPumY_u, v["Func(logistic)"]
T_amb, T_room = v.weaSta_reaWeaTDryBul_y, v.reaTZon_y
physics = u_hp * 10000 * ((T_room + 15 - T_amb) / ((T_room + 15) * 0.55)) + (1110 + 500) * u_hp_on

prep = PreprocessingSingleStep(
    inputs=["oveHeaPumY_u", "Func(logistic)", "weaSta_reaWeaTDryBul_y", "reaTZon_y"],
    output="reaPHeaPum_y",
    columns={"pinn": physics},
    time_index_col="time",
)

# physics_targets maps the physics columns to their weight. The base architecture is a CMNN.
model = PINNModel(physics_targets={"pinn": 0.5}, epochs=200, early_stopping_epochs=20)
td = prep.pipeline(DATA, model=model)  # the model adds its physics targets to the loss columns
model.pipeline(td)
run.save("model", prep, model, td)

print(
    f"PINN test RMSE {td.metrics.kpi('test', 'rmse'):.1f} W, "
    f"disagreement with the physical model (test RMSE) {td.metrics.physics['pinn']['test']['rmse']:.1f} W"
)
