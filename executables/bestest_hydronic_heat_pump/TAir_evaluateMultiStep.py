from physXAI.io import Run
from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Creates a linear regression to predict the air temperature using the Boptest data.
The model is trained as a single step model, but also evaluated over a horizon of 48 steps: it is
applied recursively, as in an MPC, which shows how its error grows over the horizon.
"""
# Directory of the run (stored_data/TAir_evaluateMultiStep)
run = Run('stored_data', 'TAir_evaluateMultiStep', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Lagged inputs ('X_lag1') and changes ('Change(X)') are derived automatically
inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u',
          'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2', 'reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2']
# Output feature: the change of the air temperature, which is also an input (recursive feature)
output = 'Change(reaTZon_y)'

# Create Training data. eval_horizon adds the recursive evaluation over 48 steps
prep = PreprocessingSingleStep(inputs=inputs, output=output, eval_horizon=48)
# Process Training data
td = prep.pipeline(file_path)

# Linear Regression
m = LinearRegressionModel()

# Training pipeline
model = m.pipeline(td)

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, m, td)
