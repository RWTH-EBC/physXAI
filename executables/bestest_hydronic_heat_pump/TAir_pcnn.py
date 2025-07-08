from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import ANNModel, ClassicalANNModel, PCNNModel
from physXAI.preprocessing.preprossesing import PreprocessingMultiStep
from physXAI.utils.logging import Logger


"""
Creates a Physically Consistent Neural Network (PCNN) as proposed by Di Natale et al., 2022, (see https://doi.org/10.1016/j.apenergy.2022.119806)
to predict the air temperature using the Boptest data.

The PCNN consists of a disturbance module and a physics module which explicitly covers parts of the underlying
system dynamics.
"""

# Setup up logger for saving
Logger.setup_logger(folder_name='TAirRoom', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features and lagged inputs
inputs = ['weaSta_reaWeaHDirNor_y', 'weaSta_reaWeaTDryBul_y - reaTZon_y', 'oveHeaPumY_u']
# Output feature
output = 'reaTZon_y'

""" 
The constructed features are automatically added to the data via 'core_functions.preprocessing.constructed.py' 
Lagged inputs can be added directly based on the feature
"""
x1 = Feature('reaTZon_y')
x2 = Feature('weaSta_reaWeaTDryBul_y')
x3 = x2-x1
x3.rename('weaSta_reaWeaTDryBul_y - reaTZon_y')

# Create Training data
prep = PreprocessingMultiStep(inputs, output, label_width=48, warmup_width=0, init_features=['reaTZon_y'])
# Process Training data
td = prep.pipeline(file_path)

disturbance_ann = ClassicalANNModel(rescale_output=False)

m = PCNNModel(disturbance_ann, dis_features=1, epochs=50)

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)