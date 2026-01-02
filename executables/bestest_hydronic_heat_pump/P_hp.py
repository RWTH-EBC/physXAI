from physXAI.models.models import LinearRegressionModel
from physXAI.models.ann.ann_design import ClassicalANNModel, CMNNModel, LinANNModel
from configuration import *

"""
Creates standard models to predict the power of the heat pump using the Boptest data.
"""

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

bestest_php.pipeline(m, online_epochs=5)
