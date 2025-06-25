from core_functions.preprocessing.preprossesing import PreprocessingSingleStep
from core_functions.preprocessing.constructed import Feature
from core_functions.models.models import LinearRegressionModel
from core_functions.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel
from core_functions.utils.logging import Logger


"""
Creates standard models to predict the power of the heat pump using the Boptest data.
"""

# Setup up logger for saving
Logger.setup_logger(folder_name='P_hp', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

""" 
Example how to use constructed features. 
The constructed features are automatically added to the data via 'core_functions.preprocessing.constructed.py' 
The names of the constructed features should be added to the input list
"""
# x = Feature('weaSta_reaWeaTDryBul_y')  # Create Feature for calculation
# x.lag(1)  # Lags can be added directly based on the feature
# x.exp()
# y = Feature('reaTZon_y')
# z = x + y  # Arithmetic operations can be performed on Features to create constructed Features
# z.rename('test')  # Constructed features derive a name based on the arithmetic operation.
# It is recommended to rename features, so that they can be easily added to the input list

# Create Training data
prep = PreprocessingSingleStep(inputs, output)
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
# m.epochs = 5
# model_ol = m.online_pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
