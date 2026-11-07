"""
12 - Feature selection (experimental)
=====================================

`FeatureSelection` removes inputs one by one (backward elimination) and keeps the best set.

- Criterion: for a model predicting one of its inputs (state model), the step weighted RMSE of the recursive
  prediction over `eval_horizon`, with the first step counting `first_step_weight`. For an output model, or with
  `criterion='single_step'`, the single step error.
- `n_repeats` evaluates every candidate on several splits. Differences smaller than the spread are noise.
- `controls={'u': 1}`: the prediction has to increase with `u` on (almost) all validation samples, otherwise a
  candidate is implausible and not selected. Controls (and with `criterion='horizon'` the recursive feature) are never removed.
- `tolerance` selects the smallest set within (1 + tolerance) x the best criterion.

The single step error hardly correlates with the performance in an MPC, so the result is a suggestion, not a
decision. Compare the candidates afterwards over several seeds (example 11).
"""

from pathlib import Path

import physXAI
from physXAI.feature_selection.selection import FeatureSelection
from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'
physXAI.set_seed(0)

# --- State model: air temperature, criterion over 12 h -----------------------------------------------------------
run = Run(ROOT / 'results', '12_feature_selection/TAir', overwrite=True, log_level='warning')
candidates = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'reaTZon_y']
lags = {'weaSta_reaWeaTDryBul_y': 2, 'weaSta_reaWeaHDirNor_y': 2, 'oveHeaPumY_u': 2, 'reaTZon_y': 2}
prep = PreprocessingSingleStep(inputs=candidates, lags=lags, output='Change(reaTZon_y)', eval_horizon='12h',
                               time_index_col='time')
selection = FeatureSelection(prep, LinearRegressionModel(), first_step_weight=0.25, n_repeats=3,
                             controls={'oveHeaPumY_u': 1}, tolerance=0.02)
result = selection.run(DATA)
print('Selected inputs (air temperature):', result.selected.inputs)
print(result.table()[['n_inputs', 'score', 'score_std', 'single_step_rmse', 'plausible']].to_string())
result.save(run)  # feature_selection.json and feature_selection.html

# --- Output model: heat pump power, single step criterion ---------------------------------------------------------
run = Run(ROOT / 'results', '12_feature_selection/P_hp', overwrite=True, log_level='warning')
prep = PreprocessingSingleStep(inputs=['oveHeaPumY_u', 'Func(logistic)', 'weaSta_reaWeaTDryBul_y', 'reaTZon_y'],
                               lags={'oveHeaPumY_u': 2, 'weaSta_reaWeaTDryBul_y': 1, 'reaTZon_y': 1},
                               output='reaPHeaPum_y', time_index_col='time')
selection = FeatureSelection(prep, LinearRegressionModel(), controls={'oveHeaPumY_u': 1}, n_repeats=3)
result = selection.run(DATA)
print('Selected inputs (heat pump power):', result.selected.inputs)
result.save(run)
