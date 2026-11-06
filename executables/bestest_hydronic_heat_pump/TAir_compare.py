from physXAI import expr as px
from physXAI.evaluation.comparison import ModelComparison, repeat_training
from physXAI.io import Run
from physXAI.models import LinearRegressionModel, ModularANN
from physXAI.models.ann.ann_design import ClassicalANNModel, RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep
from physXAI.training import RecursiveTraining

"""
Compares models predicting the air temperature, each trained with three seeds.

- One training run says little: the split of the data and the initial weights change the errors. Every
  model is trained with the seeds 0, 1, 2. The same seeds give all models the same splits of the data.
- The models differ in their outputs (single step models predict the change of the temperature, the RNN
  its trajectory), so they are compared on the error over the prediction horizon: the temperature
  predicted recursively, as in an MPC, on the horizon all models are evaluated on (12 h).
- Every run is saved into stored_data/TAir_compare/<model>/seed_<seed>/, the comparison (html and csv)
  into stored_data/TAir_compare/. ModelComparison.load compares saved runs at any time later.
"""
run = Run('stored_data', 'TAir_compare', overwrite=True, log_level='warning')

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"
seeds = [0, 1, 2]

inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'reaTZon_y']
output = px.Change('reaTZon_y')
single_step = PreprocessingSingleStep(inputs=inputs, output=output, eval_horizon='12h', time_index_col='time')

# Linear regression and a classical ANN, trained on the single step error
repeat_training(file_path, single_step, LinearRegressionModel(), seeds=seeds, run=run, name='linreg')
repeat_training(file_path, single_step, ClassicalANNModel(epochs=100, early_stopping_epochs=20), seeds=seeds,
                run=run, name='ann')

# Physically motivated architecture, fine-tuned on the error over 6 h (inputs are taken from the model)
v = px.Vars()
T, T_amb, u_hp = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u
architecture = px.Linear([T - T_amb, u_hp, u_hp.lag(1), v.weaSta_reaWeaHDirNor_y,
                          px.MLP([u_hp, T_amb, T], n_neurons=8)], normalize=True)
modular = ModularANN(architecture, epochs=100, early_stopping_epochs=20,
                     recursive_training=RecursiveTraining(horizon='6h', first_step_weight=0.25))
recursive_prep = PreprocessingSingleStep(output=output, eval_horizon='12h', time_index_col='time')
repeat_training(file_path, recursive_prep, modular, seeds=seeds, run=run, name='modular_recursive')

# Recurrent neural network, trained on the trajectory over 12 h and evaluated over 24 h
multi_step = PreprocessingMultiStep(inputs=['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u',
                                            'reaTZon_y'], output=output, label_width='12h', warmup_width='6h',
                                    eval_horizon='24h', time_index_col='time')
repeat_training(file_path, multi_step, RNNModel(rnn_layer='LSTM', rnn_units=16, epochs=100, early_stopping_epochs=20),
                seeds=seeds, run=run, name='rnn')

# Compare all saved runs: one model per sub directory
comparison = ModelComparison.load(run.path)
print(comparison.summary(split='test').to_string(float_format='{:.4f}'.format))
comparison.save(run)  # comparison.html, comparison_summary.csv, comparison_scores.csv, ...
