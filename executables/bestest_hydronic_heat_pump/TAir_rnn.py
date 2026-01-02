from physXAI.models.ann.ann_design import RNNModel
from configuration import *

"""
Creates a recurrent neural network (RNN) to predict the air temperature using the Boptest data
"""

"""
List of input features. Can include constructed features and lagged inputs.
As a RNN is applied, there is no need to add the recursive feature 'reaTZon_y' to the inputs.
RNNs need to be initialized. Therefore a initialization model is trained based on the 'inits' features.
The RNN directly predicts the air temperature instead of the air temperature change.
"""
bestest_tair.inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u']
bestest_tair.output = 'reaTZon_y'

# Create RNN model. Models needs a rnn type for the output (rnn_layer) and the initialization (init_layer) model
m = RNNModel(epochs=100, rnn_layer='LSTM', init_layer='LSTM')

bestest_tair.pipeline(m)
