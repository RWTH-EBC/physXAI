from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.constructed import Feature
from physXAI.utils.logging import Logger


"""
This script demonstrates the usage of different shifts. It is not physically meaningful.
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='Dummy_shifting_ann', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features and lagged inputs
inputs = ['reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']
# Output feature
output = 'Change(T_zone)'

""" 
The constructed features are automatically added to the data via 'physXAI.preprocessing.constructed.py' 
Lagged inputs can be added directly based on the feature
"""
x1 = Feature('reaTZon_y')
x1.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
x2 = Feature('weaSta_reaWeaTDryBul_y')
x2.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
x3 = Feature('oveHeaPumY_u')
x3.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2


"""
shift (Union[int, str, dict]): Time step of the input data used to predict the output.
    - If a single int or str is given, it applies to all inputs.
    - If a dict is provided, it can specify different shifts for individual inputs.
    - If not all inputs are specified in the dict, unspecified inputs will use a default value (autocomplete).
    Examples:
        - shift = 0 or shift = 'current': Current time step will be used for prediction.
        - shift = 1 or shift = 'previous': Previous values will be used for prediction.
        - shift = 'mean_over_interval': Mean between current and previous time step will be used.
        - shift = {
            'inp_1': 1,
            'inp_2': 'mean_over_interval',
            '_default': 0,  # current time step will be used for all inputs not specified in the dict
            # If no custom default value is given in dict, 'previous' will be used as default
        }
"""
shift = {
    'reaTZon_y': 'previous',  # for all lags of reaTZon_y, the shift will be set automatically
    'weaSta_reaWeaHDirNor_y': 'mean_over_interval',
    '_default': 0,
}

# Create Training data
# Time step defines target sampling: if original sampling of data is in 15min intervals, it is resampled to 1h intervals for time_step=4
# Hence, if the shift method of an input is defined as 'mean_over_interval', the mean over the last hour is taken as input
prep = PreprocessingSingleStep(inputs, output, shift=shift, time_step=4)

# Process Training data
td = prep.pipeline(file_path)

# Classical ANN
m = ClassicalANNModel(epochs=500)

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)