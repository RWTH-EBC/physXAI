from physXAI.preprocessing.constructed import Feature
from physXAI.utils.test_case import TestCase

# TODO: add Docstring

# File path to data
data_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

""" 
Example how to use constructed features. 
The constructed features are automatically added to the data via 'physXAI.preprocessing.constructed.py' 
The names of the constructed features should be added to the input list
"""
# x = Feature('weaSta_reaWeaTDryBul_y')  # Create Feature for calculation
# x.lag(1)  # Lags can be added directly based on the feature
# x.exp()
# y = Feature('reaTZon_y')
# z = x + y  # Arithmetic operations can be performed on Features to create constructed Features
# z.rename('test')  # Constructed features derive a name based on the arithmetic operation.
# It is recommended to rename features, so that they can be easily added to the input list

"""heat pump power prediction"""
# List of input features. Can include constructed features
u_hp = Feature('oveHeaPumY_u')
u_hp_logistic = Feature('Func(logistic)')
t_amb = Feature('weaSta_reaWeaTDryBul_y')
TAirRoom = Feature('reaTZon_y')
inputs_php = [u_hp, u_hp_logistic, t_amb, TAirRoom]

# Output feature
output_php = ['reaPHeaPum_y']

# define test case
bestest_php = TestCase(inputs_php, output_php, data_path)


"""room temperature prediciton"""
# The constructed features are automatically added to the data via 'physXAI.preprocessing.constructed.py'
# Lagged inputs can be added directly based on the feature
TAirRoom.lag(2)  # reaTZon_y_lag1, reaTZon_y_lag2
t_amb.lag(1)  # weaSta_reaWeaTDryBul_y_lag1
u_hp.lag(2)  # oveHeaPumY_u_lag1, oveHeaPumY_u_lag2

# List of input features. Can include constructed features and lagged inputs
inputs_tair = [TAirRoom, 'reaTZon_y_lag1', 'reaTZon_y_lag2', t_amb, 'weaSta_reaWeaTDryBul_y_lag1',
               'weaSta_reaWeaHDirNor_y', u_hp, 'oveHeaPumY_u_lag1', 'oveHeaPumY_u_lag2']

# Output feature
output_tair = 'Change(T_zone)'

# Define test case, including parameter for multistep models
# Number of time steps in the output (label) sequence.
label_width = 48
# Number of time steps in the warmup sequence (for RNN state initialization).
warmup_width = 48
bestest_tair = TestCase(inputs_tair, output_tair, data_path, label_width=label_width, warmup_width=warmup_width)
