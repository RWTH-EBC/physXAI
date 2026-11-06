# Evaluation, comparison and feature selection

## Metrics

After `model.pipeline(td)`:

- `td.metrics.by_output[output][split]`: MSE, RMSE, MAE, R², bias (mean of prediction − truth) and the
  number of samples, for every output and split separately. The flat dicts `td.metrics.test_kpis` etc.
  (`'RMSE Test'`) hold the mean over the outputs.
- `td.horizon_metrics`: RMSE, MAE, bias and standard deviation per step of the horizon (see
  [Multi step evaluation](multi_step.md)), per predicted feature in `by_feature`.

Both are written to `<name>_training_data.json` by `Run.save`.

## Report

`model.plot(td)` returns one figure per model: true vs. predicted values (small points, binned median per
split with its interquartile band), the sorted predictions, RMSE and bias over the horizon, the training
history and a metrics table. Plots are never opened unless requested (`show_plot=True`).

## Comparing models

One training run says little about a model: the split of the data and the initial weights change the
errors, often more than the difference between two models. `repeat_training` trains a model with several
seeds, `ModelComparison` compares the distributions:

```python
from physXAI.evaluation.comparison import ModelComparison, repeat_training

run = Run("stored_data", "TAir_compare", overwrite=True)
repeat_training(path, prep, LinearRegressionModel(), seeds=5, run=run, name="linreg")
repeat_training(path, prep, ClassicalANNModel(epochs=100), seeds=5, run=run, name="ann")
repeat_training(path, rnn_prep, RNNModel(rnn_layer="LSTM"), seeds=5, run=run, name="rnn")

comparison = ModelComparison.load(run.path)     # one model per sub directory
comparison.summary()                            # per model: runs, mean, std, min, median, max
comparison.save(run)                            # comparison.html and csv files
```

- Every run is saved with `Run.save` into `<run>/<name>/seed_<seed>/`. With the same seeds, all models are
  trained and tested on the same splits (`vary=('split', 'init')` by default; `vary=('init',)` keeps the
  split fixed and only varies the weights).
- `ModelComparison.load` also takes a list of directories or `{label: directory}`, so runs saved at any
  time (also by physXAI < 1.0) can be compared. Every `*training_data.json` is one run.
- Without saving: `comparison = repeat_training(...) + repeat_training(...)`, or
  `ModelComparison().add("label", td)` for evaluated training data.

Tables: `runs()` (seeds, model class, training time), `metrics()` (single step metrics),
`horizon()` (errors per step), `scores(split)` (one row per run) and `summary(split)`.

Models with different outputs (e.g. `Change(T)` of a single step model and the trajectory `T` of an RNN)
are compared on the error of `T` over the horizon. `scores` and `plot` use the horizon all runs reach,
or `horizon=` in hours.

The plot shows box plots of the single step RMSE (every point is a run), RMSE and bias over the horizon
(median over the runs and the range, or `band='iqr'`), box plots of the mean RMSE over the horizon and
the RMSE at its end, and the summary table.
See `executables/bestest_hydronic_heat_pump/TAir_compare.py`.

## Feature selection (experimental)

`FeatureSelection` removes inputs one by one (backward elimination) and keeps the best set:

```python
from physXAI.feature_selection.recursive_feature_elimination import save_result
from physXAI.feature_selection.selection import FeatureSelection

prep = PreprocessingSingleStep(inputs=candidates, output="Change(reaTZon_y)", eval_horizon=48)
selection = FeatureSelection(prep, LinearRegressionModel(), first_step_weight=0.25, n_repeats=3,
                             controls={"oveHeaPumY_u": 1}, tolerance=0.02)
result = selection.run(path)
result.selected.inputs
result.table()
save_result(result, run)      # feature_selection.json and .html
```

- The criterion is the step weighted RMSE of the recursive prediction over `eval_horizon` (optionally plus
  `bias_weight` · |bias| at `mpc_horizon`), or `criterion='single_step'`.
- `n_repeats` evaluates every candidate on several splits. Differences smaller than the spread are noise.
- `controls={'u': 1}`: the prediction has to increase with `u` on (almost) all validation samples,
  otherwise a candidate is implausible and not selected. Controls and the recursive feature are never removed.
- `tolerance` selects the smallest set within (1 + tolerance) × the best criterion.
