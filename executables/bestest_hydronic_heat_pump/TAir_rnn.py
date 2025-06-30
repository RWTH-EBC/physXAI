from physXAI.models.ann.ann_design import RNNModel
from physXAI.preprocessing.preprossesing import PreprocessingMultiStep
from physXAI.utils.logging import Logger


"""
Creates a recurrent neural network (RNN) to predict the air temperature using the Boptest data
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='TAir_rnn', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"


"""
List of input features. Can include constructed features and lagged inputs.
As a RNN is applied, there is no need to add the recursive feature 'reaTZon_y' to the inputs.
RNNs need to be initialized. Therefore a initialization model is trained based on the 'inits' features.
The RNN directly predicts the air temperature instead of the air temperature change.
"""
inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u']
inits = ['reaTZon_y']
output = 'reaTZon_y'

# Number of time steps in the output (label) sequence.
label_width = 48
# Number of time steps in the warmup sequence (for RNN state initialization).
warmup_width = 48

# Create Training data. For RNNs MultiStep training data is required
prep = PreprocessingMultiStep(inputs, output, label_width, warmup_width, init_features=inits)
# Process Training data
td = prep.pipeline(file_path)

# Create RNN model. Models needs a rnn type for the output (rnn_layer) and the initialization (init_layer) model
m = RNNModel(epochs=100, rnn_layer='LSTM', init_layer='LSTM')

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
