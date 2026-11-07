# Data pipeline and split

## Loading

`PreprocessingSingleStep` and `PreprocessingMultiStep` share the options of `PreprocessingData`:

| Option | Meaning |
|---|---|
| `inputs`, `output` | column names; `X_lagN` and `Change(X)` are derived, `lags={'X': 2}` adds `X_lag1`, `X_lag2` |
| `shift` | 1 (default): the output at the next time step, for states (`Change(T)`) and as AgentLib-MPC uses it; 0: the output at the time of the inputs, for algebraic outputs of point sampled data (the output feature must not be an input, no recursive evaluation, not exportable) |
| `columns` | derived data columns as expressions, see [Expressions](expressions.md) |
| `loss_columns` | columns used only in the training loss (e.g. physics targets), `'y:name'` at the predicted time; added by models via `prep.pipeline(path, model=model)` |
| `time_index_col`, `time_step` | time column (seconds or date time); the data is put onto a regular grid of `time_step`. A longer `time_step` takes the values at the time steps (point sampling, with a warning): resample quantities over an interval (power, energy) before, e.g. with `pandas.DataFrame.resample` |
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
- an embargo of `gap_steps` separates blocks of different splits, so no value of the data is part of a training
  and a test sample. The default is the reach of a sample into the past (lags, `Change(X)` = X(t) - X(t-1) and the
  variables of derived `columns`) plus `shift`. The warmup of multi step sequences needs no embargo, it lies inside
  the split of its sequence. The rows are taken from the
  training side, so validation and test blocks stay complete,
- windows of multi step data never cross a split boundary.

The embargo removes shared samples, not the autocorrelation of the building: test blocks lie between training
blocks of the same weeks and season, so the test error measures how well the model interpolates within the period
of the data. For a new period (another season or year) it is optimistic. To estimate that, use
`split='holdout_end'` (the end of the data for validation and test), a longer `gap_steps` or a separate data set.
`split='random_rows'` is the old random split (for comparisons only).

The split depends on `random_state`. Training a model with several seeds shows how much the errors depend
on the split, see [Comparing models](evaluation.md#comparing-models).

## Training data

`prep.pipeline(path)` returns a `TrainingData` (single step) or `TrainingDataMultiStep` object:

- `td.X_train`, `td.y_train`, ... the samples of a split, `td.inputs`, `td.outputs`,
- after training: predictions, `td.metrics` and `td.horizon_metrics`,
- `td.save(path)` / `TrainingData.load(path)` for single step data (npz and json, no pickle; multi step data is
  created from the same csv file and preprocessing config),
- `td.get_config()` holds metrics and history, as written by `Run.save`.
