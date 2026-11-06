from physXAI.feature_selection.recursive_feature_elimination import save_result
from physXAI.feature_selection.selection import FeatureSelection
from physXAI.io import Run
from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Feature selection (experimental) for a linear regression predicting the air temperature using the Boptest data.

- The single step error hardly correlates with the MPC performance, so the candidates are compared by the error
  of the recursive prediction over 12 hours (eval_horizon), weighted with the first step counting more.
- Every candidate is evaluated on 3 different block splits, the spread shows which differences are significant.
- The heat pump is the control of the MPC: a candidate whose prediction does not increase with the heat pump
  signal on (almost) all validation samples is marked implausible and not selected.
"""
# Directory of the run (stored_data/TAir_feature)
run = Run('stored_data', 'TAir_feature', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of all possible input features candidates. The recursive feature and its lags come last.
inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1', 'weaSta_reaWeaTDryBul_y_lag2',
          'weaSta_reaWeaHDirNor_y', 'weaSta_reaWeaHDirNor_y_lag1', 'weaSta_reaWeaHDirNor_y_lag2',
          'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2', 'reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2']
# Output feature
output = 'Change(reaTZon_y)'

# Generic Preprocessing Pipeline. The model is a state model, so it is evaluated recursively over 12 h (48 steps)
prep = PreprocessingSingleStep(inputs=inputs, output=output, eval_horizon=48)

# Generic Model
m = LinearRegressionModel()

# Feature Selection
fs = FeatureSelection(prep, m, first_step_weight=0.25, n_repeats=3, controls={'oveHeaPumY_u': 1}, tolerance=0.02)
result = fs.run(file_path)
print('Selected inputs:', result.selected.inputs)
print(result.table()[['n_inputs', 'score', 'score_std', 'single_step_rmse', 'plausible']].to_string())

# Save the result (json) and its plot (html) into the run directory
save_result(result, run)
