"""
Example to demonstrate interplay of physXAI and agentlib_mpc to train a simple ANN for one_room_mpc
Keep synchronized with physXAI.agentlib_mpc_plugin.example
"""
import numpy as np
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.constructed import Feature
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.io import Run


def train_model(base_path: str, folder_name: str, training_data_path: str, time_step: int,
                   output_name: str = 'T') -> str:
    """
    Example to demonstrate interplay of physXAI and agentlib_mpc to train a simple ANN for one_room_mpc
    """

    run = Run(base_path, folder_name, overwrite=True)

    inputs = ['mDot',
              'T',
        ]
    output = 'Change(T)'

    x1 = Feature('T')
    x2 = x1.lag(1)
    x3 = x1 - x2
    x3.rename("Change(T)")

    random_state = int(np.random.default_rng().integers(0, 100000000))
    prep = PreprocessingSingleStep(inputs=inputs, output=output, csv_delimiter=',', time_step=time_step, random_state=random_state)
    td = prep.pipeline(training_data_path)

    m = ClassicalANNModel(epochs=1000)
    model = m.pipeline(td, plot=False, save_model=False)

    # Writes {output_name}_preprocessing.json, _model.json, _training_data.json and the model file
    return run.save_for_agentlib(output_name, prep, m, td, trained_model=model)