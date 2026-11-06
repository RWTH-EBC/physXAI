import os
import pathlib

from physXAI.io import Run
from physXAI.models.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Creates standard model to predict the power of the heat pump using the Boptest data
"""
base_path = os.path.join(pathlib.Path(__file__).resolve().parent.parent, 'results')

# Directory of the run
run = Run(base_path, 'tests/verify_installation', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Lagged inputs ('X_lag1') are derived automatically
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output)
# Process Training data
td = prep.pipeline(file_path)

# Linear Regression
m = LinearRegressionModel()

# Training pipeline
model = m.pipeline(td)

# Save configs, metrics, model and report
run.save('model', prep, m, td)
