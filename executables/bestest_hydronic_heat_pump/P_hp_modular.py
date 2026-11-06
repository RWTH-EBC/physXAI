from physXAI import expr as px
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.utils.logging import Logger

"""
Creates a modular model to predict the power of the heat pump using the Boptest data.

The architecture is an expression graph (physXAI.expr): physical arithmetic on raw data columns
combined with trainable blocks. Everything is part of the keras model, so an MPC only needs the
raw inputs. The inputs of the preprocessing are taken from the model.
"""

Logger.setup_logger(folder_name='P_hp_modular', override=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

v = px.Vars()
u_hp, logistic = v.oveHeaPumY_u, v['Func(logistic)']
architecture = px.MLP([u_hp / logistic, logistic ** 2, v.weaSta_reaWeaTDryBul_y, v.reaTZon_y])
m = ModularANN(architecture=architecture, rescale_output=True)

prep = PreprocessingSingleStep(output='reaPHeaPum_y')
td = prep.pipeline(file_path, model=m)

# Training pipeline
model = m.pipeline(td)

# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
