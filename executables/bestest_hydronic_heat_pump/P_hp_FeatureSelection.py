from core_functions.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from core_functions.models.models import LinearRegressionModel
from core_functions.preprocessing.constructed import Feature
from core_functions.preprocessing.preprossesing import PreprocessingSingleStep
from core_functions.utils.logging import Logger


"""
Feature selection for linear regression to predict the power of the heat pump using the Boptest data
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='P_hp_feature', override=True)

# File path to data
file_path = r"data/Boptest/pid_data.csv"

# List of all possible input features candidates.
inputs = ['oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2', 'Func(logistic)', 'Func(logistic)_lag1',
          'Func(logistic)_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          'weaSta_reaWeaTDryBul_y_lag2', 'reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2']
# Output feature
output = 'reaPHeaPum_y'

# Create lags
x1 = Feature('oveHeaPumY_u')
x1.lag(2)
x2 = Feature('Func(logistic)')
x2.lag(2)
x3 = Feature('weaSta_reaWeaTDryBul_y')
x3.lag(2)
x4 = Feature('reaTZon_y')
x4.lag(2)

# Generic Preprocessing Pipeline
# Model is output model, so single step evaluation is choosen
prep = PreprocessingSingleStep(inputs, output)

# Generic Model
m = LinearRegressionModel()

# Feature Selection
fs = recursive_feature_elimination_pipeline(file_path, prep, m, ascending_lag_order=True)
