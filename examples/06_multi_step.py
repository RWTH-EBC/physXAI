"""
06 - Error over the prediction horizon and recursive training
=============================================================

The single step error says little about the performance in an MPC. There a single step model is applied
recursively over the prediction horizon, and small errors add up: a slight bias of Change(T) becomes a large
deviation of T after some hours.

- `eval_horizon` evaluates the model recursively on windows of the data, as in an MPC: the predicted temperature
  (and its lags) are fed back, all other inputs are taken from the data. RMSE, MAE, bias and standard deviation of
  the error of T are computed per step and shown in the report.
- `RecursiveTraining` fine-tunes the model on the error of the recursive prediction ('single_then_multi': single
  step training first). The loss weights the first step with `first_step_weight`, the remaining steps evenly.
  The saved model stays a single step model, AgentLib-MPC uses it unchanged.

The same architecture is trained without and with recursive training.
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.training import RecursiveTraining

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

run = Run(ROOT / "results", "06_multi_step", overwrite=True, log_level="warning")
physXAI.set_seed(0)


def architecture():
    v = px.Vars()
    T, T_amb, u_hp = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u
    return px.Linear(
        [T - T_amb, u_hp, u_hp.lag(1), v.weaSta_reaWeaHDirNor_y, px.MLP([u_hp, T_amb, T], n_neurons=8)], normalize=True
    )


models = {
    "single_step": ModularANN(architecture(), epochs=200, early_stopping_epochs=20),
    "recursive": ModularANN(
        architecture(),
        epochs=200,
        early_stopping_epochs=20,
        recursive_training=RecursiveTraining(horizon="6h", first_step_weight=0.25, mode="single_then_multi"),
    ),
}
prep = PreprocessingSingleStep(output=px.Change("reaTZon_y"), time_index_col="time", eval_horizon="12h")
for name, model in models.items():
    td = prep.pipeline(DATA, model=model)
    model.pipeline(td)
    run.save(name, prep, model, td)  # the report <name>.html shows RMSE and bias over the horizon
    test = td.horizon_metrics.kpis["test"]
    print(
        f"{name:12s} single step RMSE {td.metrics.kpi('test', 'rmse'):.4f} K | RMSE of T after 6 h "
        f"{test['rmse'][23]:.3f} K, after 12 h {test['rmse'][-1]:.3f} K, bias after 12 h {test['bias'][-1]:+.3f} K"
    )
