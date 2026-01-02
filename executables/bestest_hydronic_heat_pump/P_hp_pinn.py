from physXAI.models.ann.ann_design import PINNModel
from configuration import *


"""
Creates a physics-informed model (PINN) to predict the power of the heat pump using the Boptest data
"""


"""
Output feature(s). In addition to the standard data-driven loss, the PINN includes a physical loss function, 
which is created as a constructed feature
"""
bestest_php.outputs.append('pinn')

"""
Arithmetic operations can be performed on Features to create constructed Features
The pinn features serves as a physical loss function for the PINN
"""
pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)
pinn.rename('pinn')

"""
Create PINN model. 
pinn_weights are used to balance the individual loss therms of the PINN, their length should be `num_outputs - 1`
This ´PINN implementation uses a CMNN as its base architecture
"""
m = PINNModel(pinn_weights=[1], epochs=50)

bestest_php.pipeline(m)
