"""
11 - Comparing models over several seeds
========================================

One training run says little about a model: the split of the data and the initial weights change the errors, often
more than the difference between two models. `repeat_training` trains a model with several seeds, `ModelComparison`
compares the distributions.

- With the same seeds and the same split settings, all models are trained and tested on the same splits. The
  block length depends on the window length (horizon, warmup of the RNN) and on the size of the data, so all
  preprocessings get the same `BlockSplit` here.
- `paired` compares every model with a baseline run by run on the same split: the spread caused by the split
  cancels out, which the distributions of `summary` cannot show.
- The models differ in their outputs (single step models predict Change(T), the RNN a trajectory), so they are
  compared on the error of T over the horizon all of them are evaluated on (12 h), predicted recursively as in an MPC.
- Every run is saved into results/11_compare_models/<model>/seed_<seed>/, the comparison (html and csv)
  into results/11_compare_models/.
- `ModelComparison.load` compares saved runs at any time later, also runs of physXAI < 1.0 and of AgentLib-MPC
  training scripts. From the command line:
      python -m physXAI.evaluation.comparison results/11_compare_models --split test --horizon 12 --baseline linreg
"""

from pathlib import Path

from physXAI import expr as px
from physXAI.evaluation.comparison import ModelComparison, repeat_training
from physXAI.io import Run
from physXAI.models import ClassicalANNModel, LinearRegressionModel, ModularANN, RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep
from physXAI.preprocessing.splitting import BlockSplit
from physXAI.training import RecursiveTraining

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '11_compare_models', overwrite=True, log_level='warning')
seeds = [0, 1, 2]  # more seeds give a better picture of the spread

output = px.Change('reaTZon_y')
inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1', 'reaTZon_y']
# One split for all models: blocks of 32 h hold the longest window (the RNN: 6 h warmup + 24 h evaluation), and
# min_blocks=10 keeps that length for the two weeks of data (by default the blocks are shortened to get 20 of them).
split = BlockSplit(test_size=0.1, val_size=0.1, block_length='32h', min_blocks=10)
single_step = PreprocessingSingleStep(inputs=inputs, output=output, eval_horizon='12h', time_index_col='time',
                                      split=split)

# Linear regression and a classical ANN, trained on the single step error
repeat_training(DATA, single_step, LinearRegressionModel(), seeds=seeds, run=run, name='linreg')
repeat_training(DATA, single_step, ClassicalANNModel(epochs=100, early_stopping_epochs=20), seeds=seeds, run=run,
                name='ann')

# Physically motivated architecture, fine-tuned on the error over 6 h (example 06)
v = px.Vars()
T, T_amb, u_hp = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u
architecture = px.Linear([T - T_amb, u_hp, u_hp.lag(1), v.weaSta_reaWeaHDirNor_y,
                          px.MLP([u_hp, T_amb, T], n_neurons=8)], normalize=True)
modular = ModularANN(architecture, epochs=100, early_stopping_epochs=20,
                     recursive_training=RecursiveTraining(horizon='6h', first_step_weight=0.25))
modular_prep = PreprocessingSingleStep(output=output, eval_horizon='12h', time_index_col='time', split=split)
repeat_training(DATA, modular_prep, modular, seeds=seeds, run=run, name='modular_recursive')

# Recurrent neural network, trained on the trajectory over 12 h (example 07)
multi_step = PreprocessingMultiStep(inputs=['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u',
                                            'reaTZon_y'], output=output, label_width='12h', warmup_width='6h',
                                    eval_horizon='24h', time_index_col='time', split=split)
repeat_training(DATA, multi_step, RNNModel(rnn_layer='LSTM', rnn_units=16, epochs=100, early_stopping_epochs=20),
                seeds=seeds, run=run, name='rnn')

# Compare all saved runs: every sub directory is one model
comparison = ModelComparison.load(run.path)
print(comparison.summary(split='test').to_string(float_format='{:.4f}'.format))
# Run by run on the same splits: mean difference to the linear regression with its confidence interval, share of the
# splits on which a model is better, p-value (Wilcoxon, needs at least 6 seeds for 0.05), same_data = 1: same splits
paired = comparison.paired('linreg', split='test')
print(paired[['pairs', 'mean_difference', 'ci_low', 'ci_high', 'better', 'same_data']]
      .to_string(float_format='{:.4f}'.format))
comparison.save(run, baseline='linreg')  # comparison.html, comparison_summary.csv, comparison_paired.csv, ...

# Only some of the models, with labels
subset = ModelComparison.load({'Linear regression': run.path / 'linreg', 'RNN': run.path / 'rnn'})
print(subset.summary(split='test')[['runs', 'mean', 'std']].to_string(float_format='{:.4f}'.format))
