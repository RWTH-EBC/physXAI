from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.constructed import Feature, FeatureExp
from physXAI.utils.logging import Logger

"""
This script demonstrates the usage of different sampling methods. It is not physically meaningful.

When creating a Feature, a sampling method can be specified.
For constructed features, no sampling method is necessary. It is assigned based on their corresponding base feature(s)

sampling_method (Union[str, int]): Time step of the input data used to predict the output.
    - if None: Feature.get_default_sampling_method() is used
    - if 'current' or 0: Current time step will be used for prediction.
    - if 'previous' or 1: Previous time step will be used for prediction.
    - if 'mean_over_interval': Mean between current and previous time step will be used.
    
    Specify default sampling method using Feature.set_default_sampling_method(<your default sampling>).
    If no default sampling method is specified by the user, 'previous' is used as default.
"""
Feature.set_default_sampling_method(0)

# Setup up logger for saving
Logger.setup_logger(folder_name='different_sampling_methods_ann', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include names of constructed features and lagged inputs
inputs = ['reaTZon_y', 'reaTZon_y_lag1', 'reaTZon_y_lag2', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaTDryBul_y_lag1',
          Feature('weaSta_reaWeaHDirNor_y', sampling_method='mean_over_interval'), 'oveHeaPumY_u',
          'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']
# Output feature. Can include names of constructed features as well
output = ['Change(T_air)']

""" 
The constructed features are automatically added to the data via 'physXAI.preprocessing.constructed.py' 
Lagged inputs can be added directly based on the feature
"""
# create lags of reaTZon_y: reaTZon_y_lag1, reaTZon_y_lag2
x1 = Feature('reaTZon_y', sampling_method='previous')
lx1 = x1.lag(2)  # for all lags of reaTZon_y, the shift will be set automatically as 'previous'

# create lag of weaSta_reaWeaTDryBul_y: weaSta_reaWeaTDryBul_y_lag1
x2 = Feature('weaSta_reaWeaTDryBul_y')
lx2 = x2.lag(1)

# create lag of oveHeaPumY_u: oveHeaPumY_u_lag1, oveHeaPumY_u_lag2
x3 = Feature('oveHeaPumY_u')
x3.lag(2)

# dummy Features
y = x1 + lx1[0]
z = y + x1
z.rename('example_feature_two')  # since z is a constructed feature based on x1, its sampling_method will be previous
e = FeatureExp(x1-273.15, 'exp', sampling_method=1)  # reduce x1 by 273.15, otherwise values are too high
inputs.extend([z, e])  # add dummy features to inputs

# construct output
change_tair = x1 - lx1[0]
change_tair.rename('Change(T_air)')

# Create Training data
# Time step defines target sampling: if original sampling of data is in 15min intervals, it is resampled to 1h intervals
# for time_step=4. Hence, if the shift method of an input is defined as 'mean_over_interval', the mean over the last
# hour is taken as input
prep = PreprocessingSingleStep(inputs, output, time_step=4)

# Process Training data
td = prep.pipeline(file_path)

# Build & train Classical ANN
m = ClassicalANNModel(epochs=50)
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
