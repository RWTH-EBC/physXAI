from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import PINNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Creates a physics-informed model (PINN) to predict the power of the heat pump using the Boptest data
"""
# Directory of the run (stored_data/P_hp_pinn)
run = Run('stored_data', 'P_hp_pinn', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']

"""
Output feature(s). In addition to the standard data-driven loss, the PINN includes a physical loss function,
which is computed from the data as an expression
"""
output = ['reaPHeaPum_y', 'pinn']

"""
The physical target 'pinn' is an expression of data columns, added to the data via `columns`.
It serves as a physical loss function for the PINN.
"""
v = px.Vars()
u_hp, u_hp_logistic = v.oveHeaPumY_u, v['Func(logistic)']
t_amb, TAirRoom = v.weaSta_reaWeaTDryBul_y, v.reaTZon_y
pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output, columns={'pinn': pinn})
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

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, m, td)
