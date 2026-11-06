import os

from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI import expr as px
from physXAI.models.models import LinearRegressionModel
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel
from physXAI.utils.logging import Logger


"""
Creates standard models to predict the power of the heat pump using the Boptest data.
"""

# Setup up logger for saving
Logger.setup_logger(folder_name='P_hp', override=True, print_level='warning')

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Lagged inputs ('X_lag1') are derived automatically
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

"""
Derived inputs: lagged inputs ('weaSta_reaWeaTDryBul_y_lag1') are derived from their names.
Other derived quantities are best placed inside the model (see P_hp_modular.py). If a derived
quantity has to be a data column, it can be added as expression:
"""
v = px.Vars()
columns = {'T_sum': v.weaSta_reaWeaTDryBul_y + v.reaTZon_y}  # not used as input in this example

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output, columns=columns)
# Process Training data
td = prep.pipeline(file_path)

"""Example usages of different models"""
# m = LinearRegressionModel()  # Linear Regression
m = ClassicalANNModel(epochs=50)  # Classical ANN
# m = CMNNModel(monotonies={  # (-1 for decreasing, 0 for no constraint, 1 for increasing)
#     'oveHeaPumY_u': 1,
#     'Func(logistic)': 1,
#     'weaSta_reaWeaTDryBul_y': -1,
#     'reaTZon_y': 1,
# }, activation_split=[1, 1, 1],  # Proportions for splitting neurons into convex, concave, and saturated activation
# epochs=50)  # Constrained Monotonic Neural Network (CMNN)
# m = LinANNModel(epochs=50)  # A hybrid model combining a Linear Regression model with an ANN (RBF)

# Training pipeline
model = m.pipeline(td)

"""Example usage of online learning"""
m.epochs = 5
model_ol = m.online_pipeline(td, os.path.join(Logger._logger, 'model.keras'))

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
