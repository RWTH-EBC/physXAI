"""
Contract script of a typical user run: a linear regression predicting Change(T) with an input
derived from the recursive state, computed as data column (fallback: the MPC has to compute 'vT').
Called by AgentLib-MPC through `generate_physxai_model`.
"""

from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(
    base_path: str, folder_name: str, training_data_path: str, time_step: int, output_name: str = "T"
) -> str:
    run = Run(base_path, folder_name, overwrite=True, log_level="warning")
    v = px.Vars()
    prep = PreprocessingSingleStep(
        inputs=["mDot", "load", "vT", "T"],
        output=px.Change("T"),
        columns={"vT": v.mDot * (v.T_in - v.T)},
        csv_delimiter=",",
        time_step=time_step,
        random_state=1,
        ignore_nan=True,
    )
    td = prep.pipeline(training_data_path)
    m = LinearRegressionModel()
    model = m.pipeline(td, plot=False, save_model=False)
    return run.save_for_agentlib(output_name, prep, m, td, trained_model=model)
