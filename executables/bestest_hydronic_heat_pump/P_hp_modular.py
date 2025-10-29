from physXAI.models.modular.modular_ann import ModularANN, ModularModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.constructed import Feature
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.utils.logging import Logger


"""
Creates modular models to predict the power of the heat pump using the Boptest data.
"""

Logger.setup_logger(folder_name='P_hp_modular', override=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

inputs = ['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y']
output = 'reaPHeaPum_y'

oveHeaPumY_u = Feature('oveHeaPumY_u')
func_logistic = Feature('Func(logistic)')
TDryBul = Feature('weaSta_reaWeaTDryBul_y')
TZon = Feature('reaTZon_y')

prep = PreprocessingSingleStep(inputs=inputs, output=output)
td = prep.pipeline(file_path)

"""Example usages of modular models"""
y = ModularModel(
    model=ClassicalANNModel(),
    inputs=[oveHeaPumY_u.input() / func_logistic.input(), func_logistic.input() ** 2, TDryBul.input(), TZon.input()],
    rescale_output=True
)
m = ModularANN(architecture=y)

# Training pipeline
model = m.pipeline(td)


# Log setup of preprocessing and model as json
Logger.log_setup(prep, m)
# Log training data as pickle
Logger.save_training_data(td)
