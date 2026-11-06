# Data pipeline and split

## Loading

`PreprocessingSingleStep` and `PreprocessingMultiStep` share the options of `PreprocessingData`:

| Option | Meaning |
|---|---|
| `inputs`, `output` | column names; `X_lagN` and `Change(X)` are derived, `lags={'X': 2}` adds `X_lag1`, `X_lag2` |
| `columns` | derived data columns as expressions, see [Expressions](expressions.md) |
| `time_index_col`, `time_step` | time column (seconds or date time); the data is put onto a regular grid of `time_step` |
| `csv_delimiter`, `csv_encoding`, `csv_header`, `csv_skiprows` | csv options |
| `ignore_nan` | drop rows with missing values (targets are never taken across gaps) |
| `transform` | function `DataFrame -> DataFrame` applied after loading (not serialized) |
| `test_size`, `val_size`, `random_state` | size of the splits and their seed |
| `split`, `block_length`, `gap_steps` | split strategy, see below |

Time spans can be given in steps or as strings: `eval_horizon='24h'`, `block_length='7D'`.

## Split without leakage

Neighbouring time steps are strongly correlated. A random split of rows puts almost identical samples into
training and test data and overestimates the test performance. physXAI splits blocks of time instead
(`split='blocks'`, default):

- the time axis is cut into blocks (default: the longer of four windows and one day),
- validation and test blocks are drawn in every stratum of neighbouring blocks, so all seasons are part of
  every split,
- an embargo of `gap_steps` (default: the largest lag plus the warmup) separates blocks of different splits,
- windows of multi step data never cross a split boundary.

`split='holdout_end'` uses the end of the data for validation and test (non-stationary data),
`split='random_rows'` the old random split (for comparisons only).

The split depends on `random_state`. Training a model with several seeds shows how much the errors depend
on the split, see [Comparing models](evaluation.md#comparing-models).

## Training data

`prep.pipeline(path)` returns a `TrainingData` (single step) or `TrainingDataMultiStep` object:

- `td.X_train`, `td.y_train`, ... the samples of a split, `td.inputs`, `td.outputs`,
- after training: predictions, `td.metrics` and `td.horizon_metrics`,
- `td.save(path)` / `TrainingData.load(path)` (npz and json, no pickle),
- `td.get_config()` holds metrics and history, as written by `Run.save`.
