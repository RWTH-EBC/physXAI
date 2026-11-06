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
  holds NumPy windows and builds `tf.data` pipelines on demand (reshuffled every epoch).
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

### Removed
- `physXAI.preprocessing.constructed` (`Feature`, `FeatureConstruction`): lags and `Change(X)` are derived
  natively, other quantities are expressions (`physXAI.expr`), preferably inside the model.
- `physXAI.models.modular.modular_expression` and the `Modular*` classes: replaced by `physXAI.expr`
  (`ModularModel` -> `px.MLP` / `px.MonotoneMLP`, `ModularLinear` / `ModularMonotoneLinear` -> `px.Linear`,
  `ModularTrainable` -> `px.Param`, `ModularAverage` -> `px.Mean`, `ModularNormalization` -> `px.Normalize`,
  `ModularExistingModel` -> `px.Pretrained`). The separate `modular_expression_config.json` is gone, the graph
  is part of the model config.
- `GitPython` dependency; paths are no longer resolved relative to the git root.
- `requirements.txt` with outdated pins (use `pip install -e .[dev]`).
