from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.models.models import LinearRegressionModel
from configuration import *

"""
Feature selection for linear regression to predict the power of the heat pump using the Boptest data
"""

# List of all possible input features candidates (overwrite default set in configuration.py)
bestest_php.inputs.extend(['oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2', 'Func(logistic)_lag1', 'Func(logistic)_lag2',
                           'weaSta_reaWeaTDryBul_y_lag1', 'weaSta_reaWeaTDryBul_y_lag2', 'reaTZon_y_lag1',
                           'reaTZon_y_lag2'])
# Create lags
u_hp.lag(2)
u_hp_logistic.lag(2)
t_amb.lag(2)
TAirRoom.lag(2)

# Generic Model
m = LinearRegressionModel()

# Feature Selection
fs = bestest_php.recursive_feature_elimination_pipeline(m, ascending_lag_order=True)
