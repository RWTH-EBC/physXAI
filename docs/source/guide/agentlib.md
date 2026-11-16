# Export for AgentLib-MPC

AgentLib-MPC (`agentlib_mpc.machine_learning_plugins.physXAI`) calls a training script of physXAI and
loads the trained model into the MPC. The script provides

```python
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(base_path: str, folder_name: str, training_data_path: str, time_step: int,
                output_name: str = "T") -> str:
    run = Run(base_path, folder_name, overwrite=True, log_level="warning")
    v = px.Vars()
    model = ModularANN(px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T]), solver="ols")
    prep = PreprocessingSingleStep(output=px.Change("T"), csv_delimiter=",", time_step=time_step)
    td = prep.pipeline(training_data_path, model=model)
    model.pipeline(td)
    return run.save_for_agentlib(output_name, prep, model, td)
```

`save_for_agentlib` checks the configuration (`check_agentlib_compatibility`) and writes
`{name}_preprocessing.json`, `{name}_model.json`, `{name}_training_data.json` and `{name}.keras` or
`{name}.joblib` into `base_path/folder_name`. Examples: `physXAI/agentlib_mpc_plugin/example.py`.

## Rules

AgentLib-MPC reads the inputs and the output from their names:

- `Change(X)` is an output predicting the difference of `X`, the MPC integrates it,
- `X_lagN` is the value of `X` N steps in the past,
- lags of a feature are consecutive and ascending, the recursive feature and its lags are the last inputs
  (inputs taken from the model are ordered like that automatically),
- the lags of a feature start at the feature itself: `['u', 'u_lag1', 'T']`, not `['u_lag1', 'T']`. AgentLib-MPC
  passes the values of a feature from lag 0 on and would otherwise feed the wrong time steps. physXAI itself allows
  isolated lags (e.g. a dead time), only the export needs the complete group,
- single step models have exactly one output and `shift=1` (AgentLib-MPC evaluates an output over the interval
  `[t, t+1)` of the inputs),
- inputs are variables of the MPC: inputs computed by the preprocessing (`columns`, `transform`) have to be
  computed by the MPC as well, `check_agentlib_compatibility` warns about them. Expressions inside the model need
  only the raw inputs,
- recurrent models have no lagged inputs (their memory is the state).

`check_agentlib_compatibility(prep)` raises an `AgentLibCompatibilityError` listing all violations before
the model is trained.

## Supported models

- Linear regression (`.joblib`), all keras models and `ModularANN` (translated to CasADi layer by layer;
  expressions inside the model are part of it, the MPC only needs raw inputs).
- Recurrent models (`RNNModel`, `PCNNModel`): the step model, with AgentLib-MPC versions supporting recurrent
  models (branch `143-feature-multistep-model-support`; AgentLib-MPC releases up to 1.1.1 do not support them,
  `check_agentlib_compatibility` warns if the installed AgentLib-MPC cannot load them).
