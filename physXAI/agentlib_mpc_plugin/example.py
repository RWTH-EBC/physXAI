"""
Training script of the AgentLib-MPC example 'one_room_mpc': a classical ANN of the room temperature (1000 epochs).

`example.py` is used by the example, `example_ci.py` (the same with 10 epochs) by the CI of AgentLib-MPC. A tutorial
on the export is `examples/13_agentlib_mpc.py`.
"""
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(base_path: str, folder_name: str, training_data_path: str, time_step: int,
                   output_name: str = 'T') -> str:
    """Trains the model of the room temperature, saves it for AgentLib-MPC and returns its name."""

    run = Run(base_path, folder_name, overwrite=True)

    inputs = ['mDot', 'T']
    output = px.Change('T')  # 'Change(T)', derived by the preprocessing

    prep = PreprocessingSingleStep(inputs=inputs, output=output, csv_delimiter=',', time_step=time_step)  # a new seed per run, stored in the config
    td = prep.pipeline(training_data_path)

    m = ClassicalANNModel(epochs=1000)
    model = m.pipeline(td, plot=False, save_model=False)

    # Writes {output_name}_preprocessing.json, _model.json, _training_data.json and the model file
    return run.save_for_agentlib(output_name, prep, m, td, trained_model=model)
