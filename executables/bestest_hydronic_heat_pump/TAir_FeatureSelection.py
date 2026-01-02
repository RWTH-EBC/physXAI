from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep
from configuration import *


"""
Feature selection for linear regression to predict the air temperature using the Boptest data
"""

# List of all possible input features candidates (overwrite default set in configuration.py)
bestest_tair.inputs = ['reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2', 'weaSta_reaWeaTDryBul_y',
                       'weaSta_reaWeaTDryBul_y_lag1', 'weaSta_reaWeaTDryBul_y_lag2', 'weaSta_reaWeaHDirNor_y',
                       'weaSta_reaWeaHDirNor_y_lag1', 'weaSta_reaWeaHDirNor_y_lag2', 'oveHeaPumY_u',
                       'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']

# Create lags
TAirRoom.lag(2)
t_amb.lag(2)
q_sol = Feature('weaSta_reaWeaHDirNor_y')
q_sol.lag(2)
u_hp.lag(2)

# Generic Preprocessing Pipeline
# Model is state model, so multi-step evaluation is choosen
# See example TAir_evaluateMultiStep.py for more information
bestest_tair.warmup_width = 0  # overwrite default set in configuration.py
prep = bestest_tair.get_preprocessing_multi_step(init_features=['reaTZon_y'], overlapping_sequences=False, batch_size=1)

# Generic Model
m = LinearRegressionModel()

# Feature Selection
fs = bestest_tair.recursive_feature_elimination_pipeline(m, prep=prep, use_multi_step_error=True,
                                                         ascending_lag_order=True)
