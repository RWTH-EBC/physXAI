# Migration from physXAI 0.x to 1.0

physXAI 1.0 is not backwards compatible. The integration into AgentLib-MPC keeps working: the training
script signature and the saved files are the same.

## Overview

| 0.x | 1.0 |
|---|---|
| `Logger.setup_logger(folder_name='x', override=True)` | `run = Run('stored_data', 'x', overwrite=True)` |
| `Logger.log_setup(prep, m)`, `Logger.save_training_data(td)` | `run.save('model', prep, m, td)` |
| default save path of `m.pipeline(td)` (Logger directory) | `run.save(...)`, or `m.pipeline(td, save_path=...)` |
| `Logger.print_level = 'warning'`, `print_level=` | `set_log_level('warning')`, `Run(..., log_level='warning')` |
| `Logger.save_name_model = name` + `log_setup(save_name_...)` | `run.save_for_agentlib(name, prep, m, td)` |
| `training_data.pkl` | `td.save(path)` / `TrainingData.load(path)` (npz + json) |
| `Feature('x')`, `FeatureConstruction` | `px.Vars()`: `v = px.Vars(); v.x` |
| constructed feature as input (`vT.rename('vT')`) | inside the model: `px.Linear([..., v.a * (v.b - v.T)])` or `columns={'vT': expr}` |
| `Feature('T').lag(1)`, `x3 = t - t.lag(1)` for `Change(T)` | derived automatically from the names `T_lag1`, `Change(T)` (or `px.Change('T')`) |
| `ModularModel(ClassicalANNModel)` | `px.MLP` |
| `ModularModel(CMNNModel)` | `px.MonotoneMLP` |
| `ModularLinear`, `ModularMonotoneLinear` | `px.Linear(monotonicities=...)` |
| `ModularTrainable`, `ModularAverage`, `ModularNormalization`, `ModularExistingModel` | `px.Param`, `px.Mean`, `px.Normalize`, `px.Pretrained` |
| `modular_expression_config.json` | part of the model config |
| `RNNModel(init_layer=...)`, `PreprocessingMultiStep(init_features=...)` | warmup from zero states: `warmup_width` |
| `PreprocessingMultiStep(batch_size=...)` | `RNNModel(batch_size=...)` |
| single step model on `PreprocessingMultiStep` data (`evaluate_multi`) | `PreprocessingSingleStep(eval_horizon='24h')` |
| `recursive_feature_elimination_pipeline` with `input()` prompts | `FeatureSelection(...).run(path)` |
| plots open in the browser | saved as html; `show_plot=True` opens them |
| random row split | block split without leakage (`split='random_rows'` for the old behaviour) |
| `np.random.seed(); random_state=np.random.randint(...)` | `random_state=None` (a seed is drawn and stored) |

## Example: training script for AgentLib-MPC

physXAI 0.x:

```python
def train_model(base_path, folder_name, training_data_path, time_step, output_name='TAir'):
    Logger.setup_logger(base_path=base_path, folder_name=folder_name, override=True, print_level='WARNING')

    v_flow_ahu = Feature('V_flow_AHU')
    ahu_temp_sup = Feature('T_AHU_sup')
    t_room = Feature('TAir')
    x3 = t_room - t_room.lag(1)
    x3.rename("Change(TAir)")
    vT = v_flow_ahu * (ahu_temp_sup - t_room)
    vT.rename("vT")

    np.random.seed()
    random_state = np.random.randint(low=0, high=100000000)
    prep = PreprocessingSingleStep(inputs=['V_flow_AHU', 'y_valve_h', 'T_sup_w_h', 'TDryBul', 'HDirNor', 'vT', 'TAir'],
                                   output='Change(TAir)', csv_delimiter=',', time_step=time_step,
                                   random_state=random_state, ignore_nan=True)
    td = prep.pipeline(training_data_path)

    m = LinearRegressionModel()
    Logger.save_name_model = output_name
    m.pipeline(td, plot=False)
    Logger.log_setup(prep, m, save_name_preprocessing=f'{output_name}_preprocessing.json',
                     save_name_model=f'{output_name}_model.json',
                     save_name_constructed=f'{output_name}_constructed.json')
    Logger.save_training_data(td, path=f'{output_name}_training_data')
    return output_name
```

physXAI 1.0, with `vT` inside the model. It is recomputed in every step of the MPC from the predicted
`TAir`, and the MPC only needs the raw inputs:

```python
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep


def train_model(base_path, folder_name, training_data_path, time_step, output_name='TAir'):
    run = Run(base_path, folder_name, overwrite=True, log_level='warning')

    v = px.Vars()
    model = ModularANN(px.Linear([
        v.V_flow_AHU, v.y_valve_h, v.T_sup_w_h, v.TDryBul, v.HDirNor,
        v.V_flow_AHU * (v.T_AHU_sup - v.TAir),   # vT
        v.TAir,
    ]), solver='ols')                             # linear regression in closed form

    prep = PreprocessingSingleStep(output=px.Change('TAir'), csv_delimiter=',', time_step=time_step,
                                   ignore_nan=True, eval_horizon='24h')
    td = prep.pipeline(training_data_path, model=model)   # inputs from the model
    model.pipeline(td)                                     # incl. the error over 24 h
    return run.save_for_agentlib(output_name, prep, model, td)
```

If the model has to stay a scikit-learn linear regression (`.joblib`), `vT` is a data column, which the
MPC then has to compute:

```python
v = px.Vars()
prep = PreprocessingSingleStep(inputs=['V_flow_AHU', 'y_valve_h', 'T_sup_w_h', 'TDryBul', 'HDirNor', 'vT', 'TAir'],
                               output='Change(TAir)', columns={'vT': v.V_flow_AHU * (v.T_AHU_sup - v.TAir)},
                               csv_delimiter=',', time_step=time_step, ignore_nan=True)
td = prep.pipeline(training_data_path)
model = LinearRegressionModel()
model.pipeline(td)
return run.save_for_agentlib(output_name, prep, model, td)
```

## Behaviour that changed

- **Split:** blocks of time instead of random rows. Test errors are higher than before and closer to the
  errors in operation.
- **Gaps:** data is put onto a regular time grid; targets are never taken across gaps.
- **Saving:** `pipeline` saves nothing without a `save_path`. Use `run.save`.
- **Plots:** not opened by default.
- **Metrics:** per output (`td.metrics.by_output`); the flat keys stay and hold the mean over the outputs.
- **RNNs:** the saved `.keras` file is the step model used by AgentLib-MPC; RNNs with initialization
  models have to be retrained.
- **Old runs** can still be compared with new ones: `ModelComparison.load` reads training data files of 0.x.
