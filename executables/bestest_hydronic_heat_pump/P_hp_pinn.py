from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import PINNModel
from physXAI.utils.logging import Logger


"""
Creates a physics-informed model (PINN) to predict the power of the heat pump using the Boptest data
"""
# Setup up logger for saving
Logger.setup_logger(folder_name='P_hp_pinn', override=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features. Can include constructed features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']

"""
Output feature(s). In addition to the standard data-driven loss, the PINN includes a physical loss function, 
which is created as a constructed feature
"""
output = ['reaPHeaPum_y', 'pinn']

""" 
The constructed features are automatically added to the data via 'physXAI.preprocessing.constructed.py' 
The names of the constructed features should be added to the input list
"""
u_hp = Feature('oveHeaPumY_u')
u_hp_logistic = Feature('Func(logistic)')
t_amb = Feature('weaSta_reaWeaTDryBul_y')
TAirRoom = Feature('reaTZon_y')

"""
Arithmetic operations can be performed on Features to create constructed Features
The pinn features serves as a physical loss function for the PINN
"""
pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)
pinn.rename('pinn')

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output)
# Process Training data
td = prep.pipeline(file_path)

"""
Create PINN model. 
pinn_weights are used to balance the individual loss therms of the PINN, their length should be `num_outputs - 1`
This ´PINN implementation uses a CMNN as its base architecture
"""
m = PINNModel(pinn_weights=[1], epochs=50)

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
