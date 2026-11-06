from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Creates a modular model to predict the power of the heat pump using the Boptest data.

The architecture is an expression graph (physXAI.expr): physical arithmetic on raw data columns
combined with trainable blocks. Everything is part of the keras model, so an MPC only needs the
raw inputs. The inputs of the preprocessing are taken from the model.
"""

# Directory of the run (stored_data/P_hp_modular)
run = Run('stored_data', 'P_hp_modular', overwrite=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

v = px.Vars()
u_hp, logistic = v.oveHeaPumY_u, v['Func(logistic)']
architecture = px.MLP([u_hp / logistic, logistic ** 2, v.weaSta_reaWeaTDryBul_y, v.reaTZon_y])
m = ModularANN(architecture=architecture, rescale_output=True)

prep = PreprocessingSingleStep(output='reaPHeaPum_y')
td = prep.pipeline(file_path, model=m)

# Training pipeline
model = m.pipeline(td)

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, m, td)
