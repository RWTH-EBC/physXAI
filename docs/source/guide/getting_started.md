# Getting started

physXAI trains data driven models of building energy systems for model predictive control (MPC),
in particular for [AgentLib-MPC](https://github.com/RWTH-EBC/AgentLib-MPC). A typical workflow:

1. **Preprocessing** loads a csv file, derives lags and changes, splits the data without leakage and
   returns training data.
2. **A model** is trained and evaluated: single step metrics and, for models applied recursively in an
   MPC, the error over the prediction horizon.
3. **A run directory** stores configs, metrics, the model and a report, so runs can be reproduced and
   compared, and AgentLib-MPC can load the model.

## Installation

physXAI needs Python >= 3.11 and runs on Linux and Windows.

```bash
pip install .                    # user
pip install -e .[dev,agentlib]   # developer: tests, linting, AgentLib-MPC contract tests
pip install -e .[docs]           # documentation: sphinx-apidoc -M -o docs/source/code physXAI && sphinx-build -b html docs/source docs_output
```

## A first model

The air temperature of a room is predicted as change per time step, `Change(T) = T(t+1) - T(t)`.
A model predicting a change is applied recursively in the MPC: the predicted temperature becomes the
input of the next step.

```python
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

run = Run("stored_data", "TAir", overwrite=True)

prep = PreprocessingSingleStep(
    inputs=["weaSta_reaWeaTDryBul_y", "oveHeaPumY_u", "oveHeaPumY_u_lag1", "reaTZon_y"],
    output=px.Change("reaTZon_y"),   # or the string 'Change(reaTZon_y)'
    time_index_col="time",
    eval_horizon="12h",              # also evaluate the recursive prediction over 12 h
)
td = prep.pipeline("data/bestest_hydronic_heat_pump/pid_data.csv")

model = LinearRegressionModel()
model.pipeline(td)                   # train and evaluate

run.save("model", prep, model, td)   # configs, metrics, model and report
```

`run.save` writes into `stored_data/TAir/`:

| File | Content |
|---|---|
| `model_preprocessing.json` | config of the preprocessing (`PreprocessingData.from_config`) |
| `model_model.json` | config of the model (`AbstractModel.model_from_config`) |
| `model_training_data.json` | metrics per output and split, errors over the horizon, training time and history |
| `model.joblib` / `model.keras` | the trained model |
| `model.html` | report: true vs. predicted, errors over the horizon, training history, metrics |

Lagged inputs such as `oveHeaPumY_u_lag1` and the output `Change(reaTZon_y)` are derived from the data
automatically. More examples are in `executables/`.

## Run directories and logging

`Run(base_path, name, overwrite=False)` is an explicit directory. Paths are absolute or relative to the
current working directory, there are no interactive prompts and no files are deleted. Without
`overwrite=True`, existing files are not overwritten (`FileExistsError`).

physXAI logs with the standard `logging` module (logger `"physXAI"`). By default metrics and the
training progress are shown:

```python
from physXAI import set_log_level

set_log_level("warning")   # only warnings, no keras progress bars
```

If the application configures logging itself (e.g. `logging.basicConfig`), physXAI messages are passed
on to it instead of being printed.

## Reproducibility

Every random choice (split, weights, batches) is seeded. `random_state=None` in the preprocessing draws a
seed and stores it in the config, so every run can be reproduced from its saved config.
`physXAI.set_seed(42)` sets one global seed for Python, NumPy and Keras.
