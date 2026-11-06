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

# List of input features. Lagged inputs ('X_lag1') and changes ('Change(X)') are derived automatically
inputs = ['reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']
# Output feature
output = 'Change(T_zone)'

# Lagged inputs ('X_lag1', ...) are derived automatically from their names

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output)
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
