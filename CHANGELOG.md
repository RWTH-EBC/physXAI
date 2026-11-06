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

### Changed
- Python >= 3.11 is required (3.9 and 3.10 are end of life).
- CI runs on Linux and Windows.
- Static versioning in `pyproject.toml` instead of `setuptools-scm`.
- `casadi` is now a declared dependency (it was imported, but missing).
- Default run folder names no longer contain `:` (invalid on Windows).

### Removed
- `GitPython` dependency; paths are no longer resolved relative to the git root.
- `requirements.txt` with outdated pins (use `pip install -e .[dev]`).
