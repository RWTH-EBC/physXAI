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

### Removed
- `GitPython` dependency; paths are no longer resolved relative to the git root.
- `requirements.txt` with outdated pins (use `pip install -e .[dev]`).
