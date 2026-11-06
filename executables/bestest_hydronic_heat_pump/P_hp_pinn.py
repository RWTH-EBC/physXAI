from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import PINNModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Creates a physics-informed model (PINN) to predict the power of the heat pump using the Boptest data.

- The model has one output, the electrical power of the heat pump, and is used like any other model
  (also in AgentLib-MPC).
- A simplified physical model of the heat pump is a physics target: the network is trained on the data
  and, with a weight, on the agreement with the physical model. All terms of the loss are divided by the
  variance of the training targets, the weight is the relative importance of the physics.
"""
# Directory of the run (stored_data/P_hp_pinn)
run = Run('stored_data', 'P_hp_pinn', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

# List of input features
inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
# Output feature
output = 'reaPHeaPum_y'

"""
Physical model of the power: an expression of data columns, added to the data via `columns`.
Like the output, it is taken at the predicted time.
"""
v = px.Vars()
u_hp, u_hp_logistic = v.oveHeaPumY_u, v['Func(logistic)']
t_amb, TAirRoom = v.weaSta_reaWeaTDryBul_y, v.reaTZon_y
pinn = (u_hp * 10000 * ((TAirRoom + 15 - t_amb) / ((TAirRoom + 15) * 0.55)) + (1110 + 500) * u_hp_logistic)

# Create Training data
prep = PreprocessingSingleStep(inputs=inputs, output=output, columns={'pinn': pinn})

"""
Create the PINN (a CMNN as base architecture). physics_targets maps the physics columns to their weight.
"""
m = PINNModel(physics_targets={'pinn': 0.5}, epochs=50)

# Process Training data: the model adds its physics targets to the loss columns of the preprocessing
td = prep.pipeline(file_path, model=m)

# Training pipeline (early stopping on the validation error of the data)
model = m.pipeline(td)

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, m, td)
