from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.constructed import Feature
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep
from physXAI.utils.logging import Logger


"""
Feature selection for linear regression to predict the air temperature using the Boptest data
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='TAir_feature', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of all possible input features candidates.
inputs = ['reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          'weaSta_reaWeaTDryBul_y_lag2', 'weaSta_reaWeaHDirNor_y', 'weaSta_reaWeaHDirNor_y_lag1',
          'weaSta_reaWeaHDirNor_y_lag2', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']
# Output feature
output = 'Change(T_zone)'

# Create lags
x1 = Feature('reaTZon_y')
x1.lag(2)
x2 = Feature('weaSta_reaWeaTDryBul_y')
x2.lag(2)
x3 = Feature('weaSta_reaWeaHDirNor_y')
x3.lag(2)
x4 = Feature('oveHeaPumY_u')
x4.lag(2)

# Generic Preprocessing Pipeline
# Model is state model, so multi-step evaluation is choosen
# See example TAir_evaluateMultiStep.py for more information
prep = PreprocessingMultiStep(inputs=inputs, output=output, label_width=48, warmup_width=0, init_features=['reaTZon_y'],
                              overlapping_sequences=False, batch_size=1)

# Generic Model
m = LinearRegressionModel()

# Feature Selection
fs = recursive_feature_elimination_pipeline(file_path, prep, m, use_multi_step_error=True, ascending_lag_order=True)
