"""
Example to demonstrate interplay of physXAI and agentlib_mpc to train a simple ANN for one_room_mpc
Keep synchronized with physXAI.agentlib_mpc_plugin.example
"""
import numpy as np
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.constructed import Feature
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.utils.logging import Logger


def train_model(base_path: str, folder_name: str, training_data_path: str, time_step: int,
                   output_name: str = 'T') -> str:
    """
    Example to demonstrate interplay of physXAI and agentlib_mpc to train a simple ANN for one_room_mpc
    """

    Logger.setup_logger(base_path=base_path, folder_name=folder_name, override=True)

    inputs = ['mDot',
              'T',
        ]
    output = 'Change(T)'

    x1 = Feature('T')
    x2 = x1.lag(1)
    x3 = x1 - x2
    x3.rename("Change(T)")

    random_state = np.random.randint(low=0, high=100000000)
    prep = PreprocessingSingleStep(inputs=inputs, output=output, csv_delimiter=',', time_step=time_step, random_state=random_state)
    td = prep.pipeline(training_data_path)

    m = ClassicalANNModel(epochs=1000)
    Logger.save_name_model = output_name
    model = m.pipeline(td, plot=True)

    Logger.log_setup(prep, m, save_name_preprocessing=f'{output_name}_preprocessing.json',
                     save_name_model=f'{output_name}_model.json',
                     save_name_constructed=f'{output_name}_constructed.json')
    Logger.save_training_data(td, path=f'{output_name}_training_data')

    return output_name