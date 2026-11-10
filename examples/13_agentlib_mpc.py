"""
13 - Export for AgentLib-MPC
============================

AgentLib-MPC (`agentlib_mpc.machine_learning_plugins.physXAI`) calls a training script of physXAI and loads the
trained model into the MPC. The script provides a function

    train_model(base_path, folder_name, training_data_path, time_step, output_name) -> str

that trains a model and saves it with `run.save_for_agentlib`. This writes {name}_preprocessing.json,
{name}_model.json, {name}_training_data.json and {name}.keras / {name}.joblib into base_path/folder_name.

AgentLib-MPC reads inputs and output from their names: Change(X) is integrated by the MPC, X_lagN is X N steps in
the past, the recursive feature and its lags are the last inputs. Inputs taken from the model are ordered like that
automatically, `check_agentlib_compatibility` lists all violations before a model is trained.

The script below trains on the BOPTEST data. `physXAI/agentlib_mpc_plugin/example.py` is the training script of the
example 'one_room_mpc' of AgentLib-MPC. Run this file directly to train the model once without AgentLib-MPC.
"""

from pathlib import Path

from physXAI import expr as px
from physXAI.io import AgentLibCompatibilityError, Run, check_agentlib_compatibility
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(
    base_path: str, folder_name: str, training_data_path: str, time_step: int, output_name: str = "reaTZon_y"
) -> str:
    """Training script called by AgentLib-MPC: trains the model of the air temperature and returns its name."""
    run = Run(base_path, folder_name, overwrite=True, log_level="warning")

    # Physical terms inside the model: the MPC only provides the raw inputs
    v = px.Vars()
    T, T_amb, u_hp = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u
    model = ModularANN(
        px.Linear(
            [T - T_amb, u_hp, u_hp.lag(1), v.weaSta_reaWeaHDirNor_y, px.MLP([u_hp, T_amb, T], n_neurons=8)],
            normalize=True,
        ),
        epochs=200,
        early_stopping_epochs=20,
    )

    prep = PreprocessingSingleStep(
        output=px.Change("reaTZon_y"), time_step=time_step, time_index_col="time", csv_delimiter=";"
    )
    td = prep.pipeline(training_data_path, model=model)
    model.pipeline(td)

    # Checks the rules of AgentLib-MPC and writes all files it reads
    return run.save_for_agentlib(output_name, prep, model, td)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

    # A configuration AgentLib-MPC cannot use: the recursive feature has to be the last input
    try:
        check_agentlib_compatibility(
            PreprocessingSingleStep(inputs=["reaTZon_y", "oveHeaPumY_u"], output="Change(reaTZon_y)")
        )
    except AgentLibCompatibilityError as error:
        print("Not compatible with AgentLib-MPC:", error)

    name = train_model(str(ROOT / "results"), "13_agentlib_mpc", str(DATA), time_step=900)
    print("Files for AgentLib-MPC:", *sorted(p.name for p in (ROOT / "results" / "13_agentlib_mpc").glob(f"{name}*")))
