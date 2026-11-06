"""
01 - A first model
==================

The air temperature of a room of the BOPTEST case 'Bestest Hydronic Heat Pump' is predicted with a linear
regression. This example shows the workflow every other example builds on:

1. a run directory and a global seed,
2. the preprocessing: csv file -> derived columns (lags, changes) -> leakage free split -> training data,
3. training and evaluation, single step and recursively over a horizon (as in an MPC),
4. saving configs, metrics, model and report,
5. loading the saved run again.

Columns of the data used in the examples:

    reaTZon_y               air temperature of the zone in K
    weaSta_reaWeaTDryBul_y  ambient temperature in K
    weaSta_reaWeaHDirNor_y  direct normal solar radiation in W/m2
    oveHeaPumY_u            modulation signal of the heat pump (0 ... 1)
    reaPHeaPum_y            electrical power of the heat pump in W
    Func(logistic)          on/off state of the heat pump (logistic of the signal)
    time                    time in s, one sample every 15 min
"""

import json
from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import AbstractModel, LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingData, PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]  # repository root, the example runs from any directory
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

# --- 1. Run directory and seed ----------------------------------------------------------------------------------
# All files of this example are written into results/01_first_model/. Without overwrite=True, existing
# files are never overwritten. One global seed makes split, initial weights and batches reproducible.
run = Run(ROOT / 'results', '01_first_model', overwrite=True)
physXAI.set_seed(42)

# --- 2. Preprocessing -------------------------------------------------------------------------------------------
# The model predicts the change of the temperature per time step, Change(T) = T(t+1) - T(t). A model predicting a
# change of one of its inputs is applied recursively in the MPC: the predicted temperature is the next input.
# Lagged inputs ('X_lag1' = X(t-1)) and the change are derived from their names, they are no columns of the csv.
prep = PreprocessingSingleStep(
    inputs=['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'oveHeaPumY_u_lag1',
            'reaTZon_y', 'reaTZon_y_lag1'],
    output=px.Change('reaTZon_y'),  # the same as the string 'Change(reaTZon_y)'
    time_index_col='time',          # time spans can be given as strings ('12h') instead of steps
    eval_horizon='12h',             # also evaluate the recursive prediction over 12 h
)
td = prep.pipeline(DATA)            # training data, split into blocks of time (no leakage), see example 02
print('Inputs:', td.inputs, '| samples (train, val, test):', len(td.X_train), len(td.X_val), len(td.X_test))

# --- 3. Training and evaluation ---------------------------------------------------------------------------------
model = LinearRegressionModel()
model.pipeline(td)  # build, train and evaluate, the metrics are stored in td

print(f"Single step test RMSE of Change(T): {td.metrics.kpi('test', 'rmse'):.4f} K")
rmse = td.horizon_metrics.kpis['test']['rmse']  # RMSE of T (not Change(T)) per step of the horizon
print(f'Recursive prediction, test RMSE of T: {rmse[0]:.3f} K after 15 min, {rmse[-1]:.3f} K after 12 h')

# --- 4. Saving --------------------------------------------------------------------------------------------------
# model_preprocessing.json, model_model.json, model_training_data.json (metrics), model.joblib, model.html (report)
files = run.save('model', prep, model, td)
print('Saved:', *[f.name for f in files.values()])

# --- 5. Loading -------------------------------------------------------------------------------------------------
# Preprocessing and model are recreated from their configs, the seed of the split is part of the config.
prep_loaded = PreprocessingData.from_config(json.loads(files['preprocessing'].read_text(encoding='utf-8')))
model_loaded = AbstractModel.model_from_config(json.loads(files['model_config'].read_text(encoding='utf-8')))
trained = model_loaded.load_model(str(files['model']))
print('Loaded preprocessing with seed', prep_loaded.random_state, '| prediction of the first test sample:',
      trained.predict(td.X_test[:1]).ravel())
