# Evaluation, comparison and feature selection

## Metrics

After `model.pipeline(td)`:

- `td.metrics.by_output[output][split]`: MSE, RMSE, MAE, R², bias (mean of prediction − truth) and the
  number of samples, for every output and split separately; `td.metrics.kpi('test', 'rmse')` returns one
  value. Splits without data (`test_size=0`, `val_size=0`) are left out.
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

run = Run("results", "TAir_compare", overwrite=True)
repeat_training(path, prep, LinearRegressionModel(), seeds=5, run=run, name="linreg")
repeat_training(path, prep, ClassicalANNModel(epochs=100), seeds=5, run=run, name="ann")
repeat_training(path, rnn_prep, RNNModel(rnn_layer="LSTM"), seeds=5, run=run, name="rnn")

comparison = ModelComparison.load(run.path)     # one model per sub directory
comparison.summary()                            # per model: runs, mean, std, min, median, max
comparison.paired("linreg")                     # run by run on the same splits, see below
comparison.save(run, baseline="linreg")         # comparison.html and csv files
```

- Every run is saved with `Run.save` into `<run>/<name>/seed_<seed>/`. With the same seeds and the same split
  settings, all models are trained and tested on the same splits (`vary=('split', 'init')` by default;
  `vary=('init',)` keeps the split fixed and only varies the weights). The block length depends on the window
  length (`eval_horizon`, `label_width`, `warmup_width`) and on the size of the data, so single step models and
  RNNs get other blocks with the default settings: pass the same `split=BlockSplit(..., block_length=...)` to all
  preprocessings (see `examples/11_compare_models.py`).
- Without saving: `comparison = repeat_training(...) + repeat_training(...)`, or
  `ModelComparison().add("label", td)` for evaluated training data.

### Paired comparison

The split changes the errors often more than the model. In `summary` the distributions of two models then
overlap, even if one of them is better on every single split. `paired(baseline)` compares every model with
the baseline run by run on the same split, so the spread caused by the split cancels out:

| Column | Meaning |
|---|---|
| `pairs` | number of runs on the same split (paired by the split seed, else by the run name, `by=`) |
| `baseline`, `model_median` | medians of the score |
| `mean_difference`, `ci_low`, `ci_high` | mean of model - baseline and its confidence interval (corrected resampled t-test, `confidence=0.95`) |
| `median_difference` | median of model - baseline |
| `better` | share of the splits on which the model is better (lower RMSE and absolute bias, higher R2) |
| `p_value` | two-sided p-value of the corrected resampled t-test |
| `resampling` | ratio n_test / n_train of the correction (0: all runs on the same split) |
| `same_data` | share of the pairs evaluated on the same rows (fingerprint of the split, physXAI >= 1.0) |

The runs are no independent samples: splits of the same data share most of their training data, and a plain
paired t-test (or Wilcoxon test) is too optimistic. Confidence interval and p-value follow the corrected resampled
t-test of Nadeau & Bengio (2003): the variance of the mean difference is `var(d) · (1/n + n_test/n_train)` instead
of `var(d)/n`. With `vary=('init',)` all runs share one split and nothing is corrected; `corrected=False` gives the
plain paired t-test. `paired` tests several scores and models at once: correct the p-values for multiple testing
(e.g. Holm) before drawing conclusions from several of them.

A `same_data` below 1 (also logged as a warning) means that the splits differ despite the same seed, e.g.
because of another block length. Single step models and RNNs on the same split can still be evaluated on
different windows (the warmup of the RNN has to lie inside the split).

### Existing runs

Runs saved earlier, also by physXAI < 1.0 or by AgentLib-MPC training scripts, are compared the same way.
A model is a directory, every directory below it holding a `*training_data.json` is one run, named
arbitrarily:

```
results/
  cmnn/
    a1b2c3d4/   training_data.json, model_config.json, preprocessing_config.json   (physXAI < 1.0)
    x9y8z7w6/   ...
    ...
  ann/
    seed_0/     ann_training_data.json, ann_model.json, ann_preprocessing.json, ann.keras   (Run.save)
```

```python
ModelComparison.load("results")                          # every sub directory is a model: cmnn, ann
ModelComparison.load(["results/cmnn", "old/ann"])        # every given directory is a model
ModelComparison.load({"CMNN": "results/cmnn"})           # with labels
ModelComparison.load(["results/cmnn"], name="T")         # only T_training_data.json (AgentLib-MPC runs)
```

- Outputs are named after the `output` of the saved preprocessing, so old and new runs of `Change(T)` share
  their scores. The RMSE per step of multi step models of physXAI < 1.0 is used as their error over the
  horizon (the bias was not stored then).
- A run directory with several models (`T_…` and `P_…` of AgentLib-MPC) gives separate models `cmnn/T` and
  `cmnn/P`. Unreadable files are skipped with a warning.

From the command line (prints the summary, saves plot and csv files into `--out`, by default
`<first path>/comparison`, e.g. `results/comparison` for the first command):

```bash
python -m physXAI.evaluation.comparison results
python -m physXAI.evaluation.comparison results/cmnn results/ann --split test --horizon 12
python -m physXAI.evaluation.comparison results/cmnn --models --band range
python -m physXAI.evaluation.comparison results --baseline cmnn     # adds the paired comparison
```

### Tables and plot

Tables: `runs()` (seeds, model class, training time), `metrics()` (single step metrics),
`horizon()` (errors per step), `scores(split)` (one row per run), `summary(split)` and `paired(baseline, split)`.

Models with different outputs (e.g. `Change(T)` of a single step model and the trajectory `T` of an RNN)
are compared on the error of `T` over the horizon. `scores` and `plot` use the horizon all runs reach,
or `horizon=` in hours.

The plot shows box plots of the single step RMSE (every point is a run), RMSE and bias over the horizon
(median over the runs and the interquartile range, or `band='range'` for min - max), box plots of the mean RMSE over the horizon and
the RMSE at its end, and the summary table.
See `examples/11_compare_models.py`.

## Feature selection (experimental)

`FeatureSelection` removes inputs one by one (backward elimination) and keeps the best set:

```python
from physXAI.feature_selection.selection import FeatureSelection

prep = PreprocessingSingleStep(inputs=candidates, output="Change(reaTZon_y)", eval_horizon=48)
selection = FeatureSelection(prep, LinearRegressionModel(), first_step_weight=0.25, n_repeats=3,
                             controls={"oveHeaPumY_u": 1}, tolerance=0.02)
result = selection.run(path)
result.selected.inputs
result.table()
result.save(run)              # feature_selection.json and .html
```

- The criterion is the step weighted RMSE of the recursive prediction over `eval_horizon` (optionally plus
  `bias_weight` · |bias| at `mpc_horizon`), or `criterion='single_step'`.
- `n_repeats` evaluates every candidate on several splits. Differences smaller than the spread are noise.
- `controls={'u': 1}`: the prediction has to increase with `u` on (almost) all validation samples,
  otherwise a candidate is implausible and not selected. Controls (and with `criterion='horizon'` the recursive feature) are never removed.
- `tolerance` selects the smallest set within (1 + tolerance) × the best criterion.
- The criterion is computed on the validation data, which also stops the training early, so the score of the
  selected set is optimistic. Judge the final model on the test data.
