"""
Contract script: a constrained least squares fit (`ModularANN(solver='qp', shape=...)`). AgentLib-MPC receives the same
KerasANN as for `modular_linreg`, the constraints only change the weights.
"""

from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN, ShapeConstraints
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(
    base_path: str, folder_name: str, training_data_path: str, time_step: int, output_name: str = "T"
) -> str:
    run = Run(base_path, folder_name, overwrite=True, log_level="warning")
    v = px.Vars()
    model = ModularANN(
        px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T], monotonicities={"load": 1}),
        solver="qp",
        shape=ShapeConstraints(monotonicities={"load": 1, "mDot": -1}, domain="data"),
    )
    prep = PreprocessingSingleStep(output=px.Change("T"), csv_delimiter=",", time_step=time_step, random_state=1)
    td = prep.pipeline(training_data_path, model=model)
    trained = model.pipeline(td, plot=False, save_model=False)
    return run.save_for_agentlib(output_name, prep, model, td, trained_model=trained)
