# Changelog

All notable changes to physXAI are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). The version is defined once in `pyproject.toml`
and has to match the git tag of a release.

## [1.0.0] – unreleased

Major release, not backwards compatible except for the AgentLib-MPC integration
(`agentlib_mpc.machine_learning_plugins.physXAI`). A migration guide is part of the documentation. The changes are
listed relative to 0.3.0 and to the development branch `26-integrate-modular-anns` (modular models, constrained
regression).

**Known limitation:** the export of recurrent models (`RNNModel`, `PCNNModel`) needs AgentLib-MPC with support for
recurrent models (branch `143-feature-multistep-model-support`), which is not released yet: AgentLib-MPC 1.1.1
loads single step models only. The export is verified against that branch, but not in the CI of physXAI.

### Added
- `physXAI.__version__` (read from the installed package metadata).
- `physXAI.io.Run`: explicit output directory for a training run, replaces the global `Logger`. The default
  directory of the runs is `results/` (before `stored_data/`).
- `physXAI.io.save_for_agentlib` and `check_agentlib_compatibility`: write and validate the files
  consumed by AgentLib-MPC. The check is stricter than AgentLib-MPC for lags without the feature itself (e.g.
  `['u', 'T_lag1']`): AgentLib-MPC accepts them, but passes the values from lag 0 on (T(k) instead of T(k-1)).
  physXAI itself trains with such isolated lags.
- Contract tests against AgentLib-MPC (`tests/contract`, optional extra `physXAI[agentlib]`).
- `physXAI.set_seed`: one global seed for Python, NumPy and Keras, default `random_state` / `random_seed` of all
  objects (splits, weight initialization, batch order, k-means of RBF networks).
- Leakage free `BlockSplit` (stratified blocks, embargo between splits) with the strategies `blocks`
  (default), `holdout_end` and `random_rows`; windows never cross a split boundary.
- Preprocessing derives `X_lagN` and `Change(X)` columns natively, `lags={'X': n}`, `transform` hook,
  date time indices, `eval_horizon` / `eval_stride` for the multi step evaluation of single step models.
- `TrainingData.save` / `TrainingData.load` (npz + json, no pickle).
- Expression language `physXAI.expr` (replaces constructed features and modular expressions): variables
  (`px.Vars()`), parameters, arithmetic, activations and trainable blocks (`Linear`, `MLP`, `MonotoneMLP`,
  `Polynomial`, `RBF`, `Mean`, `Normalize`, `Pretrained`) in one graph, evaluated with NumPy, Keras or CasADi.
- `ModularANN(architecture, solver='adam' | 'ols')` on `physXAI.expr` graphs: the whole graph is part of the keras
  model, the MPC only needs raw inputs. `solver='ols'` solves linear regressions on expressions in closed form.
- Constrained regression (replaces `ConstrainedRegression` of branch 26, an NLP over all samples with IPOPT):
  `ModularANN(px.Linear(terms), solver='qp', shape=ShapeConstraints(...))` fits a linear
  regression on expressions whose prediction is monotonic or convex in raw data columns, guaranteed on a domain (the
  box of the data or `bounds`) by an exchange method for the semi-infinite program, with a certificate per
  constraint (`exact`, `verified`, `data`, `violated`), the cost of the constraints and a report table. Solved as QP
  (HiGHS) in QR coordinates. `px.Linear(monotonicities=...)` bounds the sign of weights
  with `solver='qp'`.
- Warm start of `ModularANN` (`warm_start=True`): with a normalized `Linear` block as architecture, the gradient
  based training starts at the least squares solution of its data only terms, trainable terms (e.g. `px.MLP`) start
  with weight 0, so the training does not depend on a bad initialization.
- Inputs of the preprocessing can be derived from the model: `prep.pipeline(path, model=model)`, ordered as
  AgentLib-MPC expects them (recursive feature and its lags last, missing intermediate lags added).
- `columns={'name': expression}` in the preprocessing for derived data columns (e.g. PINN targets).
- `loss_columns` in the preprocessing: columns used only in the loss, added from `model.required_loss_columns` by
  `prep.pipeline(path, model=model)`.
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
- Example `examples/06_multi_step.py`.
- Recurrent models (`RNNModel`) warm themselves up: a sequence of `warmup_width + label_width` steps starts
  from zero states, the warmup is not part of the loss. This is how AgentLib-MPC applies the model.
- `RNNModel` with `Change(T)` outputs: the trajectory of T (measured start value plus accumulated changes) is
  predicted and trained on (`loss_on='trajectory'`, or `'outputs'`), with optional step weights
  (`first_step_weight`, `step_weights`). If T is also an input, the prediction is fed back (closed loop) in
  training and evaluation, as in an MPC.
- The saved RNN (`.keras`) is the step model `[sequence, *states] -> [prediction, *states]` used by AgentLib-MPC,
  verified by contract tests that run as soon as the installed AgentLib-MPC supports RNNs (branch
  `143-feature-multistep-model-support` of AgentLib-MPC; the released AgentLib-MPC 1.1.1 has no RNN support yet).
- `PreprocessingMultiStep(eval_horizon=..., eval_stride=...)`: evaluation over a horizon longer than the training
  horizon; `label_width` / `warmup_width` also as time delta strings ('12h'). RMSE, MAE and bias per step in
  `td.horizon_metrics` (per output in `by_feature`).
- Metrics per output: `td.metrics.by_output[output][split]` holds MSE, RMSE, MAE, R2 and bias; `HorizonMetrics`
  additionally the standard deviation of the error per step. Everything is part of the training data json.
- Plots: one report for all models (`physXAI.plotting.plotting.model_report`): per output the true vs. predicted values
  with small transparent points, the binned median per split as bold line and its interquartile band, the sorted
  predictions, the error over the horizon (RMSE, bias, bias ± σ), the training history (including a multi step
  training) and a metrics table per output. Large point clouds are thinned out for display.
- Feature selection `physXAI.feature_selection.selection.FeatureSelection` (experimental): backward elimination
  on the step weighted error of the recursive prediction (or the single step error), repeated on several block
  splits (`n_repeats`, mean and spread), absolute bias at the MPC horizon, plausibility check of the sign of the
  sensitivity to the controls (`controls={'u': 1}`), `tolerance` to prefer fewer inputs. Results as table, json
  and plot.
- `Run.save(name, preprocessing, model, training_data)`: saves the configs, the metrics, the trained model and its
  report in the layout AgentLib-MPC reads; `save_for_agentlib` uses it. `model.trained_model` holds the model of
  the last pipeline.
- Comparison of models (`physXAI.evaluation.comparison`): `repeat_training` trains a model with several seeds
  (split and / or initialization) and saves every run; `ModelComparison` loads saved runs (also of physXAI 0.x)
  or takes training data in memory and compares the distributions of the single step metrics and of the errors
  over a common prediction horizon: tables per run, summary per model (mean, std, min, median, max), box plots
  and horizon plots with median and band. Example `examples/11_compare_models.py`.
  Existing runs are loaded from model directories with arbitrarily named run directories (`results/cmnn/<run>/`),
  also with several models per run directory (AgentLib-MPC) and from the command line
  (`python -m physXAI.evaluation.comparison results`).
- Documentation with Sphinx (MyST Markdown, API reference from the docstrings): user guide (getting started,
  expressions, data pipeline and split, models, multi step evaluation and training, recurrent models,
  physics-informed models, evaluation and comparison, AgentLib-MPC export, examples) and migration guide from 0.x. Published at
  https://rwth-ebc.github.io/physXAI/main/docs/index.html.
- `PCNNModel`: physically consistent neural network after Di Natale et al. (2022): energy accumulator with
  positive heating, cooling and loss rates (initial rules of thumb per hour) in parallel to a recurrent network of
  the unforced dynamics; the predicted temperature provably rises with heating and outside temperature over the
  horizon. Initial rates given as `None` are estimated from the training data by nonnegative least squares of the
  single step energy balance (`td.physics['initial_rates']`). Closed loop, warmup and step model export as for
  `RNNModel`, verified against AgentLib-MPC (CasADi). Example `examples/10_pcnn.py`.
- Physics residuals with trainable physical parameters for all single step keras models:
  `physics=PhysicsResidual(px.Prediction() - model, weight)`, `physics_fit='calibrate'` (grey-box identification
  first, then the network), `'joint'` or `'fixed'`; `px.Prediction`, `px.Target`, `px.positive`, `px.freeze`;
  trained parameters in `td.physics`. The calibration is a deterministic least squares fit
  (`scipy.optimize.least_squares` with the CasADi Jacobian), gradient descent only for residuals without CasADi
  form. Only the network is exported. Example `examples/09_physics_residual.py`.
- `docs/branch_review.md` documents the review of the old branches.
- Saved configs record the environment (`environment`: versions of Python, NumPy, pandas, scikit-learn, Keras,
  TensorFlow and CasADi) next to `physXAI_version`.
- `check_agentlib_compatibility` warns about inputs computed by the preprocessing (`columns`, `Change(X)`,
  `transform`), which the MPC has to provide as variables, and about recurrent models if the installed AgentLib-MPC
  cannot load them.
- `constraints.txt`: the versions of all (also transitive) dependencies physXAI 1.0.0 was tested with
  (`pip install . -c constraints.txt`), checked on Linux and Windows in the CI.
- Optional extra `physXAI[gpu]` (TensorFlow with CUDA on Linux / WSL2) and `set_seed(seed, deterministic=True)`
  for exactly reproducible runs on a GPU.
- `PHYSXAI_REQUIRE_AGENTLIB=1` makes the AgentLib-MPC contract tests fail instead of being skipped if AgentLib-MPC
  is missing (set in the CI); the model saved with physXAI 0.x is part of the tests.
- `ModelComparison.paired(baseline)`: paired comparison of the runs of every model with a baseline on the same
  splits (mean difference with confidence interval and p-value of the corrected resampled t-test of Nadeau & Bengio
  2003, which accounts for the overlap of repeated splits of the same data, and the share of better splits), also in
  `save(run, baseline=...)` and `--baseline` of the command line. The training data config stores a fingerprint
  of the splits (`summary.fingerprint`), which checks that paired runs used the same rows.
- Sub-sampling to a longer `time_step` (point sampling) logs a warning: quantities over an interval (power, energy)
  have to be resampled before. Dropped duplicate and off-grid time stamps are logged.

### Changed
- `PINNModel(physics_targets={'pinn': w})`: physics targets are loss columns, no model outputs, so a PINN has one
  output and is exportable to AgentLib-MPC. The loss `mse(ŷ, y)/s + Σ w_i·mse(ŷ, p_i)/s` is normalized by the
  variance of the training targets and no longer averaged over the terms; early stopping monitors the validation
  data loss (`early_stopping_on`); history, metrics (`td.metrics.physics`) and report hold every term.
- `FeatureSelectionResult.save(run)` replaces `save_result`.
- Unknown arguments of the preprocessing and of the models raise a `TypeError` instead of being ignored.
- `shift` is 1 (next time step, default, required by AgentLib-MPC) or 0 (algebraic outputs of point sampled data,
  e.g. `P_el(t) = f(u(t))`; the output feature must not be an input). Values above 1 are rejected: a delay is
  modelled with lagged inputs, a prediction over several steps recursively (`eval_horizon`).
- `random_seed` of the keras models defaults to the global seed (`physXAI.set_seed`), else a new seed is drawn and
  stored in the config (before: 42).
- CI uses the shared template of the institute (RWTH-EBC/ci_templates, like AgentLib and AgentLib-MPC): tests,
  coverage, package build, pylint/ruff reports, documentation and GitHub Pages; additionally the unit tests on
  Windows, a smoke run of the examples (one epoch, `tests/smoke_examples.py`) and strict ruff and Sphinx
  checks. Badges and reports are published on GitHub Pages instead of being committed.
- Static versioning in `pyproject.toml` instead of `setuptools-scm`.
- `casadi` is now a declared dependency (it was imported, but missing).
- Upper bounds on the next major versions of all dependencies (e.g. `keras<4`, `tensorflow<3`): the CasADi converter
  of AgentLib-MPC recognizes keras layers by their names and attributes.
- Default run folder names no longer contain `:` (invalid on Windows).
- Single step data is split in blocks instead of random rows by default (random rows overestimate
  the test performance because of autocorrelation).
- Data is loaded onto a regular time grid: missing time stamps are missing values instead of being
  joined with their neighbours.
- `TrainingData` holds the whole time series as NumPy array with split labels; `TrainingDataMultiStep`
  holds sequences as views on it: `X_*` (warmup and prediction), `y_*` (true trajectories), `targets(split)`
  and `initial(split)`.
- The batch size of recurrent models is set at the model (`RNNModel(batch_size=...)`), not the preprocessing.
- Feature selection on the multi step error uses `PreprocessingSingleStep(eval_horizon=...)`; the controls (and
  with `criterion='horizon'` the recursive feature) are never removed. It no longer asks for input, does not
  change the preprocessing and saves its result as json and html into the run directory.
- Plots are not opened anymore by default: `pipeline(..., save_path=...)` saves them as `<model>.html` next to
  the model and `Run.save` as `<name>.html`; `show_plot=True` opens them. `Model.plot(td)` returns the figure.
- `pipeline` and `online_pipeline` save nothing without a `save_path` (there is no default directory anymore).
- Logging uses the standard `logging` module (logger `physXAI`): `set_log_level('warning')` replaces
  `Logger.print_level`; keras progress bars follow the log level. If the application configures logging,
  messages are passed on to it.
- `executables/` is replaced by `examples/`: a tutorial of 13 numbered, self-contained scripts from a first model
  to the export for AgentLib-MPC (`examples/README.md`). They use `Run`, save their models with `Run.save` and
  find the data relative to the repository. The experimental `regression_prototype.py` is replaced by the
  constrained regression (`examples/05_constrained_regression.py`).
- The test suite moved from `unittests/` to `tests/`.
- `random_state=None` draws a seed (or uses the global one) and stores it in the config.
- `PreprocessingData.from_config` restores the class from `__class_name__` and all settings.

### Fixed
- `RNNModel`, `PCNNModel`: a constant input was normalized with the variance floor 1e-12, i.e. scaled by 1e6; a
  rounded value in the MPC then shifted the hidden states strongly. Constant inputs are normalized with variance 1, as
  in the single step models.
- PINN: the arguments of the loss were swapped (`loss(y_pred, y_true)`), the RMSE metric accumulated over the data
  and physics terms, adding a physics term halved the weight of the data, the output rescaling used the physics
  columns, and a PINN could not be exported (two outputs).
- Single step data with `test_size=0`: the evaluation predicted on an empty test set.
- `px.Linear(bias=False)` with normalized inputs warns that the normalization acts as a bias; `nominal_range` has
  to be ordered.
- Multi step training and test windows overlapped (data leakage).
- With `ignore_nan=True`, targets were taken across gaps in the data.
- `PreprocessingMultiStep` ignored `ignore_nan`; configs missed csv options, so `from_config` was not
  reproducible.
- `y_val_pred_single` failed without a validation set.
- Modular models (now `physXAI.expr`): loading a config twice failed (config mutated in place); references to not
  yet built expressions became None; global caches of keras tensors broke a second model in the same process;
  auto names depended on the process history.
- The training history plot failed for models trained without keras history.
- The multi step evaluation of single step models wrote its predictions into the training data.
- Early stopping on `val_loss` without validation data; random seeds were set when a model object was
  created instead of when the model is generated.
- RNN: the summary was always printed, the verbosity ignored, the `Rescaling` layer held tensors instead of
  floats, and the config validation of `init_layer` compared strings with `is not`.
- `PINNModel.pipeline` and `online_pipeline` returned None instead of the trained model.
- Metrics of several outputs were computed on the flattened outputs (mixing their units).
- Feature selection: `input()` prompts, the preprocessing was changed in place, lags were parsed from strings.
- Keras single step networks trained on several outputs broadcast them against their single output neuron;
  they now require one output.
- `LinANNModel` wrote the evaluation of its linear regression into the training data; `RBFModel` and
  `LinANNModel` used a fixed k-means seed instead of the seed of the model.
- `RNNModel` accepted lags, changes and derived columns of a predicted feature as
  inputs, which held its measured values over the horizon (data leakage); they are rejected now.
- `CMNNModel` and `PINNModel` without monotonicities had convex neurons only, which with the non negative weights of
  the following layers made the prediction convex in all inputs. `activation_split` defaults to `[1, 1, 1]` now.
- `CMNNModel` and `PINNModel` with `n_layers=0` only had the non negative weights of the output layer, so the
  prediction was increasing in every input (also with monotonicity -1); they need `n_layers >= 1` now.
- A constant training target gave an output rescaling with the scale 0 (a constant model); the scale is 1 then.
- Single step networks shuffled with a buffer of 10000 samples: with more samples, ordered in
  time, a batch held neighbouring time steps only. The buffer holds all samples now.
- `CMNNModel` / `PINNModel` silently ignored monotonicities of names which are no inputs (now a `ValueError`) and
  built layers without neurons for small `n_neurons`.
- Names of trainable parameters of modular models (`ModularTrainable`, now `px.Param`) were lost when a saved model
  was loaded (`ConstantLayer` config).
- `LinearRegressionModel` was fitted on float32 data; all closed form solvers use float64
  (`TrainingData.X(split, dtype)`).

### Removed
- Compatibility code of 0.x: `recursive_feature_elimination_pipeline` and `save_result`, the flat metrics
  (`train_kpis`, `val_kpis`, `test_kpis`, `get_metrics`), `td.columns` / `td.output`, `pinn_weights` and
  `multi_y_loss`, the arguments `init_layer`, `init_features`, `warmup_columns_*` and the `batch_size` of
  `PreprocessingMultiStep`.
- `physXAI.preprocessing.constructed` (`Feature`, `FeatureConstruction`): lags and `Change(X)` are derived
  natively, other quantities are expressions (`physXAI.expr`), preferably inside the model.
- `physXAI.models.modular.modular_expression` and the `Modular*` classes of branch 26: replaced by `physXAI.expr`
  (`ModularModel` -> `px.MLP` / `px.MonotoneMLP`, `ModularLinear` / `ModularMonotoneLinear` -> `px.Linear`,
  `ModularPolynomial` -> `px.Polynomial`, `ModularTrainable` -> `px.Param`, `ModularAverage` -> `px.Mean`,
  `ModularNormalization` -> `px.Normalize`, `ModularExistingModel` -> `px.Pretrained`). The separate
  `modular_expression_config.json` is gone, the graph is part of the model config.
- `physXAI.models.modular.constrained_regression.ConstrainedRegression` of branch 26: replaced by
  `ModularANN(px.Linear(terms), solver='qp', shape=ShapeConstraints(...))`.
- RNN initialization models (`init_layer`) and the warmup features of the preprocessing (`init_features`,
  `warmup_columns_*`).
- Multi step evaluation of single step models with `PreprocessingMultiStep` (`SingleStepModel.evaluate_multi`):
  use `PreprocessingSingleStep(eval_horizon=...)`. `physXAI.evaluation.closed_loop` is replaced by the
  closed loop of `RNNModel` and by `physXAI.evaluation.rollout`.
- `physXAI.utils.logging` (`Logger`, `get_full_path`, `create_full_path`): replaced by `physXAI.io.Run` and the
  `logging` module, see the migration guide. Training data is no longer pickled (`TrainingData.save`).
- `GitPython` dependency; paths are no longer resolved relative to the git root.
- `requirements.txt`: replaced by the bounds in `pyproject.toml` (`pip install -e .[dev]`) and the tested versions in
  `constraints.txt`.

## [0.3.0] – 2026-11-07

### Changed
- Python 3.11 - 3.13 (3.9 and 3.10 are end of life, 3.9 is no longer available on the GitHub runners).
- `keras>=3.15.0` (older versions are vulnerable when loading models) and `GitPython>=3.2.1`.
- `requirements.txt`: the first releases fixing the open security advisories (Dependabot).
