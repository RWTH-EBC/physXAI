"""
02 - Data pipeline and split without leakage
============================================

The preprocessing turns a csv file into training data. This example shows its options:

- lags declared once per feature (`lags`) instead of listing every lagged column,
- derived data columns as expressions (`columns`),
- the split strategies: neighbouring time steps are almost identical, a random split of rows puts them into
  training and test data and overestimates the test performance. physXAI splits blocks of time with an embargo
  between them (default), or uses the end of the data (non-stationary data),
- saving and loading training data (npz and json, no pickle).
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.preprocessing.training_data import TrainingData

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

run = Run(ROOT / "results", "02_data_and_split", overwrite=True, log_level="warning")
physXAI.set_seed(0)

# --- Lags and derived columns -----------------------------------------------------------------------------------
# lags={'X': 2} adds X_lag1 and X_lag2 to the inputs. A derived data column is an expression of other columns
# (see example 04 for expressions inside the model, which is preferred: the MPC then only needs raw inputs).
v = px.Vars()
common = dict(
    inputs=["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "oveHeaPumY_u", "dT_amb", "reaTZon_y"],
    lags={"oveHeaPumY_u": 2, "reaTZon_y": 1},
    columns={"dT_amb": v.weaSta_reaWeaTDryBul_y - v.reaTZon_y},  # temperature difference to the ambient
    output="Change(reaTZon_y)",
    time_index_col="time",
    test_size=0.2,
    val_size=0.1,
)

# --- Split strategies -------------------------------------------------------------------------------------------
# Default block length: the longer of four windows and one day. The data covers only 14 days, so the blocks are
# shortened to get enough of them (warning).
splits = {
    "blocks": PreprocessingSingleStep(**common),  # default: blocks of time
    "blocks_12h": PreprocessingSingleStep(**common, block_length="12h"),  # blocks of 12 h
    "holdout_end": PreprocessingSingleStep(**common, split="holdout_end"),  # validation and test at the end
    "random_rows": PreprocessingSingleStep(**common, split="random_rows"),  # random rows, for comparison only
}
for name, prep in splits.items():
    td = prep.pipeline(DATA)
    model = LinearRegressionModel()
    model.pipeline(td)
    run.save(name, prep, model, td)
    print(f"{name:12s} test RMSE {td.metrics.kpi('test', 'rmse'):.4f} K, test samples {len(td.X_test)}")
print("The inputs (lags expanded):", td.inputs)
print("The random split mixes neighbouring time steps into training and test data, so its test error is not")
print("an estimate of the error on new data. The block split depends on the seed, see example 11 for repetitions.")

# --- Saving and loading training data ---------------------------------------------------------------------------
td = splits["blocks"].pipeline(DATA)
path = td.save(run.path / "training_data")
td_loaded = TrainingData.load(path)
print("Loaded training data:", td_loaded.X_train.shape, td_loaded.outputs)
