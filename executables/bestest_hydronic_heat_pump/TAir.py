from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import CMNNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.utils.logging import Logger


"""
Creates a constrained monotonic neural network to predict the air temperature using the Boptest data
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='TAir', override=True)

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

# Create Training data
prep = PreprocessingSingleStep(inputs, output)
# Process Training data
td = prep.pipeline(file_path)

# Constrained Monotonic Neural Network (CMNN)
m = CMNNModel(monotonies={
    'reaTZon_y': -1,
    'reaTZon_y_lag1': -1,
    'reaTZon_y_lag2': -1,
    'weaSta_reaWeaTDryBul_y': 1,
    'weaSta_reaWeaTDryBul_y_lag1': 1,
    'weaSta_reaWeaHDirNor_y': 1,
    'oveHeaPumY_u': 1,
    'oveHeaPumY_u_lag1': 1,
    'oveHeaPumY_u_lag2': 1,

}, activation_split=[1, 1, 1], epochs=100)

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
