from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.preprossesing import PreprocessingMultiStep
from physXAI.utils.logging import Logger


"""
Creates a linear regression to predict the air temperature using the Boptest data
The model is trained as a single step model, but evaluated as a multi step model
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
"""
The model is trained as a single step model and evaluated as a multi step model
warmup_width should be 0 as initial values cannot be processed by single step models
init_features: There are two prediction modes for multi-step: 
(0) Predict change of original variable, (1) Predict original variable
init_features should contain original variable, can be None in mode (1)
overlapping_sequence should be False to avoid duplicate labels for single step prediction
batch_size should be 1 as batches are processes differently in single step models
"""
prep = PreprocessingMultiStep(inputs, output, 48, 0, init_features=['reaTZon_y'],
                              overlapping_sequences=False, batch_size=1)
# Process Training data
td = prep.pipeline(file_path)

# Linear Regression
m = LinearRegressionModel()

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
