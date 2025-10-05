from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.models.models import LinearRegressionModel
from physXAI.utils.logging import Logger

"""
Creates standard model to predict the power of the heat pump using the Boptest data
"""

# Setup up logger for saving
Logger.setup_logger(folder_name='unittests\\verify_installation', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

# Create Training data
prep = PreprocessingSingleStep(inputs, output)
# Process Training data
td = prep.pipeline(file_path)

# Linear Regression
m = LinearRegressionModel()

# Training pipeline
model = m.pipeline(td, plot=False, save_model=False)
