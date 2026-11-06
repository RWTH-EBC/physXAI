"""
Contract script modelled after a typical user run (linear regression predicting Change(T)
with a derived input depending on the recursive state). Called by AgentLib-MPC through
`generate_physxai_model`. Only the produced files are part of the contract, the script
itself is migrated to the new physXAI API together with the library.
"""

from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.constructed import Feature
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(
    base_path: str, folder_name: str, training_data_path: str, time_step: int, output_name: str = "T"
) -> str:
    run = Run(base_path, folder_name, overwrite=True, log_level="warning")

    m_dot = Feature("mDot")
    t_in = Feature("T_in")
    t = Feature("T")
    change = t - t.lag(1)
    change.rename("Change(T)")
    vt = m_dot * (t_in - t)
    vt.rename("vT")

    prep = PreprocessingSingleStep(
        inputs=["mDot", "load", "vT", "T"],
        output="Change(T)",
        csv_delimiter=",",
        time_step=time_step,
        random_state=1,
        ignore_nan=True,
    )
    td = prep.pipeline(training_data_path)

    m = LinearRegressionModel()
    model = m.pipeline(td, plot=False, save_model=False)
    return run.save_for_agentlib(output_name, prep, m, td, trained_model=model)
