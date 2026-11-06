from physXAI.feature_selection.selection import FeatureSelection
from physXAI.io import Run
from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Feature selection for linear regression to predict the power of the heat pump using the Boptest data
"""
# Directory of the run (stored_data/P_hp_feature)
run = Run('stored_data', 'P_hp_feature', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of all possible input features candidates.
inputs = ['oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2', 'Func(logistic)', 'Func(logistic)_lag1',
          'Func(logistic)_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          'weaSta_reaWeaTDryBul_y_lag2', 'reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2']
# Output feature
output = 'reaPHeaPum_y'

# Lagged inputs ('X_lag1', ...) are derived automatically from their names

# Generic Preprocessing Pipeline
# Model is output model, so single step evaluation is choosen
prep = PreprocessingSingleStep(inputs=inputs, output=output)

# Generic Model
m = LinearRegressionModel()

# Feature Selection (experimental). The power has to increase with the heat pump signal (plausibility check)
fs = FeatureSelection(prep, m, controls={'oveHeaPumY_u': 1}, n_repeats=3)
result = fs.run(file_path)
print('Selected inputs:', result.selected.inputs)

# Save the result (json) and its plot (html) into the run directory
result.save(run)
