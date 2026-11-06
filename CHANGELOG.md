# Changelog

All notable changes to physXAI are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The version is defined once in `pyproject.toml`
and has to match the git tag of a release.

## [Unreleased] – 1.0.0

Major release, not backwards compatible except for the AgentLib-MPC integration
(`agentlib_mpc.machine_learning_plugins.physXAI`). A migration guide is part of the documentation.

### Added
- `physXAI.__version__` (read from the installed package metadata).
- `physXAI.io.Run`: explicit output directory for a training run, replaces the global `Logger`.
- `physXAI.io.save_for_agentlib` and `check_agentlib_compatibility`: write and validate the files
  consumed by AgentLib-MPC.
- Contract tests against AgentLib-MPC (`unittests/contract`, optional extra `physXAI[agentlib]`).
- `physXAI.set_seed`: one global seed for Python, NumPy and Keras, default `random_state` of all objects.
- Leakage free `BlockSplit` (stratified blocks, embargo between splits) with the strategies `blocks`
  (default), `holdout_end` and `random_rows`; windows never cross a split boundary.
- Preprocessing derives `X_lagN` and `Change(X)` columns natively, `lags={'X': n}`, `transform` hook,
  date time indices, `eval_horizon` / `eval_stride` for the multi step evaluation of single step models.
- `TrainingData.save` / `TrainingData.load` (npz + json, no pickle).
- Expression language `physXAI.expr` (replaces constructed features and modular expressions): variables
  (`px.Vars()`), parameters, arithmetic, activations and trainable blocks (`Linear`, `MLP`, `MonotoneMLP`,
  `Polynomial`, `RBF`, `Mean`, `Normalize`, `Pretrained`) in one graph, evaluated with NumPy, Keras or CasADi.
- `ModularANN(architecture, solver='adam' | 'ols')`: the whole graph is part of the keras model, the MPC only
  needs raw inputs. `solver='ols'` solves linear regressions on expressions in closed form.
- Inputs of the preprocessing can be derived from the model: `prep.pipeline(path, model=model)`, ordered as
  AgentLib-MPC expects them (recursive feature and its lags last, missing intermediate lags added).
- `columns={'name': expression}` in the preprocessing for derived data columns (e.g. PINN targets).
- Contract tests: every block translates to CasADi; a linear model with a derived input inside the model
  is exported as KerasANN on raw inputs only.
- Recursive (multi step) evaluation of single step models (`physXAI.evaluation.rollout`): with
  `eval_horizon='24h'` every single step model that predicts one of its inputs (e.g. `Change(T)` with input
  `T` and its lags) is applied recursively on windows of the data, as in an MPC. RMSE, MAE and bias per step
  of the horizon are stored in `td.horizon_metrics` and the training data json, and plotted over the horizon.
  Derived data columns depending on the recursive feature are recomputed in every step.
- Training of single step models on the multi step error: `recursive_training=RecursiveTraining(horizon='24h',
  first_step_weight=0.25, mode='single_then_multi')` for all keras models (and `ModularANN(solver='ols')` as
  initialization). The saved model stays a single step model for AgentLib-MPC.
- Example `executables/bestest_hydronic_heat_pump/TAir_recursive.py`.
- Recurrent models (`RNNModel`) warm themselves up: a sequence of `warmup_width + label_width` steps starts
  from zero states, the warmup is not part of the loss. This is how AgentLib-MPC applies the model.
- `RNNModel` with `Change(T)` outputs: the trajectory of T (measured start value plus accumulated changes) is
  predicted and trained on (`loss_on='trajectory'`, or `'outputs'`), with optional step weights
  (`first_step_weight`, `step_weights`). If T is also an input, the prediction is fed back (closed loop) in
  training and evaluation, as in an MPC.
- The saved RNN (`.keras`) is the step model `[sequence, *states] -> [prediction, *states]` used by AgentLib-MPC
  (branch 143), verified by contract tests that run as soon as the installed AgentLib-MPC supports RNNs.
- `PreprocessingMultiStep(eval_horizon=..., eval_stride=...)`: evaluation over a horizon longer than the training
  horizon; `label_width` / `warmup_width` also as time delta strings ('12h'). RMSE, MAE and bias per step in
  `td.horizon_metrics` (per output in `by_feature`).
- Metrics per output: `td.metrics.by_output[output][split]` holds MSE, RMSE, MAE, R2 and bias; `HorizonMetrics`
  additionally the standard deviation of the error per step. Everything is part of the training data json.
- Plots: one report for all models (`physXAI.plotting.model_report`): per output the true vs. predicted values
  with small transparent points, the binned median per split as bold line and its interquartile band, the sorted
  predictions, the error over the horizon (RMSE, bias, bias ± σ), the training history (including a multi step
  training) and a metrics table per output. Large point clouds are thinned out for display.
- Feature selection `physXAI.feature_selection.selection.FeatureSelection` (experimental): backward elimination
  on the step weighted error of the recursive prediction (or the single step error), repeated on several block
  splits (`n_repeats`, mean and spread), absolute bias at the MPC horizon, plausibility check of the sign of the
  sensitivity to the controls (`controls={'u': 1}`), `tolerance` to prefer fewer inputs. Results as table, json
  and plot.

### Changed
- Python >= 3.11 is required (3.9 and 3.10 are end of life).
- CI runs on Linux and Windows.
- Static versioning in `pyproject.toml` instead of `setuptools-scm`.
- `casadi` is now a declared dependency (it was imported, but missing).
- Default run folder names no longer contain `:` (invalid on Windows).
- Single step data is split in blocks instead of random rows by default (random rows overestimate
  the test performance because of autocorrelation).
- Data is loaded onto a regular time grid: missing time stamps are missing values instead of being
  joined with their neighbours.
- `TrainingData` holds the whole time series as NumPy array with split labels; `TrainingDataMultiStep`
  holds sequences as views on it: `X_*` (warmup and prediction), `y_*` (true trajectories), `targets(split)`
  and `initial(split)`.
- The batch size of recurrent models is set at the model (`RNNModel(batch_size=...)`), not the preprocessing.
- Feature selection on the multi step error uses `PreprocessingSingleStep(eval_horizon=...)`; the recursive
  feature and the controls are never removed. `recursive_feature_elimination_pipeline` is a wrapper around
  `FeatureSelection`: it no longer asks for input, does not change the preprocessing and saves its result as
  json and html into the run directory.
- Plots are not opened anymore by default: `pipeline(..., plot=True)` saves them as `<model>.html` next to the
  model (or into the run directory of the Logger); `show_plot=True` opens them. `Model.plot(td)` returns the figure.
- `random_state=None` draws a seed (or uses the global one) and stores it in the config.
- `PreprocessingData.from_config` restores the class from `__class_name__` and all settings.

### Fixed
- Multi step training and test windows overlapped (data leakage).
- With `ignore_nan=True`, targets were taken across gaps in the data.
- `PreprocessingMultiStep` ignored `ignore_nan`; configs missed csv options, so `from_config` was not
  reproducible.
- `y_val_pred_single` failed without a validation set.
- Modular models: `ModularModel` with `LinearRegressionModel` crashed; loading a config twice failed
  (config mutated in place); references to not yet built expressions became None; global caches of keras
  tensors broke a second model in the same process; auto names depended on the process history.
- `ConstrainedRegression`: the layer name did not contain 'dense', so the model could not be translated to
  CasADi by AgentLib-MPC; a second fit added all constraints again.
- The training history plot failed for models trained without keras history.
- The multi step evaluation of single step models wrote its predictions into the training data.
- `BuildContext` cached tensors by `id()` of temporary expressions, which Python reuses.
- Early stopping on `val_loss` without validation data; random seeds were set when a model object was
  created instead of when the model is generated.
- RNN: the summary was always printed, the verbosity ignored, the `Rescaling` layer held tensors instead of
  floats, and the config validation of `init_layer` compared strings with `is not`.
- `PINNModel.pipeline` and `online_pipeline` returned None instead of the trained model.
- Metrics of several outputs were computed on the flattened outputs (mixing their units).
- Feature selection: `input()` prompts, the preprocessing was changed in place, lags were parsed from strings.

### Removed
- `physXAI.preprocessing.constructed` (`Feature`, `FeatureConstruction`): lags and `Change(X)` are derived
  natively, other quantities are expressions (`physXAI.expr`), preferably inside the model.
- `physXAI.models.modular.modular_expression` and the `Modular*` classes: replaced by `physXAI.expr`
  (`ModularModel` -> `px.MLP` / `px.MonotoneMLP`, `ModularLinear` / `ModularMonotoneLinear` -> `px.Linear`,
  `ModularTrainable` -> `px.Param`, `ModularAverage` -> `px.Mean`, `ModularNormalization` -> `px.Normalize`,
  `ModularExistingModel` -> `px.Pretrained`). The separate `modular_expression_config.json` is gone, the graph
  is part of the model config.
- RNN initialization models (`init_layer`) and the warmup features of the preprocessing (`init_features`,
  `warmup_columns_*`). Old configs without initialization model are still accepted.
- Multi step evaluation of single step models with `PreprocessingMultiStep` (`SingleStepModel.evaluate_multi`):
  use `PreprocessingSingleStep(eval_horizon=...)`. `physXAI.evaluation.closed_loop` is replaced by the
  closed loop of `RNNModel` and by `physXAI.evaluation.rollout`.
- `GitPython` dependency; paths are no longer resolved relative to the git root.
- `requirements.txt` with outdated pins (use `pip install -e .[dev]`).
