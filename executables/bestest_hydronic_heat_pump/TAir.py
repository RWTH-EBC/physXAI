from physXAI.models.ann.ann_design import CMNNModel
from configuration import *

"""
Creates a constrained monotonic neural network to predict the air temperature using the Boptest data
"""

# Constrained Monotonic Neural Network (CMNN)
m = CMNNModel(monotonies={
    'reaTZon_y': -1,
    'reaTZon_y_lag1': -1,
    'reaTZon_y_lag2': -1,
    'weaSta_reaWeaTDryBul_y': 1,
    'weaSta_reaWeaTDryBul_y_lag1': 1,
    'weaSta_reaWeaHDirNor_y': 1,
    'oveHeaPumY_u': 1,
    'oveHeaPumY_u_lag1': 1,
    'oveHeaPumY_u_lag2': 1,

}, activation_split=[1, 1, 1], epochs=100)

bestest_tair.pipeline(m)
