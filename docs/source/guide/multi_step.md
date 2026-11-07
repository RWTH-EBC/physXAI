# Multi step evaluation and training

The single step error says little about the performance in an MPC. There a single step model is applied
recursively over the prediction horizon, and small errors add up: a slight bias in `Change(T)` becomes a
large deviation of `T` after 24 hours.

## Evaluation over the horizon

```python
prep = PreprocessingSingleStep(inputs=[..., "T", "T_lag1"], output="Change(T)", eval_horizon="24h")
td = prep.pipeline(path)
model.pipeline(td)
td.horizon_metrics.kpis["test"]["rmse"]   # RMSE per step of the horizon
```

With `eval_horizon`, every model predicting one of its inputs is applied recursively on windows of the
data, as in an MPC:

- `Change(T)` or `T` updates `T`, and `T_lag1`, `T_lag2`, ... are shifted,
- inputs `Change(T)` (= `T - T_lag1`, also lagged) and derived data columns depending on `T` are recomputed in
  every step,
- all other inputs are taken from the data.

`td.horizon_metrics` holds RMSE, MAE, bias and the standard deviation of the error of the absolute value
(`T`, not `Change(T)`) for every step and split. `eval_stride` sets the distance between the windows.
The report plots RMSE and bias ± σ over the horizon.

## Training on the error over the horizon

```python
from physXAI.training import RecursiveTraining

model = ClassicalANNModel(
    epochs=200,
    recursive_training=RecursiveTraining(horizon="6h", first_step_weight=0.25, mode="single_then_multi"),
)
```

- `mode='single_then_multi'` (recommended) trains on the single step error first and fine-tunes on the
  error of the recursive prediction; `'multi_step'` trains on the recursive error only.
- The loss is the step weighted squared error of the absolute trajectory: the first step has the weight
  `first_step_weight`, the remaining weight is distributed evenly (or `step_weights`).
- The saved model stays a single step model, AgentLib-MPC uses it unchanged.

All keras models support it, `ModularANN(solver='ols')` uses the closed form solution as initialization.
See `examples/06_multi_step.py`.
