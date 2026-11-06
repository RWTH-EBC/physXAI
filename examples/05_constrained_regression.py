"""
05 - Constrained regression
===========================

A linear regression on expressions with monotonicity and convexity constraints, solved with CasADi / IPOPT.
The model stays linear in its parameters (fast, interpretable), the constraints hold on all training samples.

The electrical power of the heat pump is modelled as linear combination of terms of the signal and the
temperatures. Physically, the power rises with the signal and falls with the ambient temperature (higher
efficiency). The constraints refer to raw data columns, the derivative of the whole model with respect to the
column is constrained (here e.g. d/dT_amb of `u_hp * T_amb`).

The constrained model is compared with the unconstrained least squares fit of the same terms.
"""

from pathlib import Path

import numpy as np

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ConstrainedRegression, ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '05_constrained_regression', overwrite=True, log_level='warning')
physXAI.set_seed(0)

v = px.Vars()
u_hp, logistic = v.oveHeaPumY_u, v['Func(logistic)']
T, T_amb = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y
terms = [u_hp, u_hp * T_amb, u_hp * T, logistic, T_amb]

models = {
    'constrained': ConstrainedRegression(terms, monotonies={'oveHeaPumY_u': 1, 'weaSta_reaWeaTDryBul_y': -1}),
    'unconstrained': ModularANN(px.Linear(terms), solver='ols'),
}
prep = PreprocessingSingleStep(output='reaPHeaPum_y', time_index_col='time')
for name, model in models.items():
    td = prep.pipeline(DATA, model=model)
    trained = model.pipeline(td)
    run.save(name, prep, model, td)

    # Effect of +1 K ambient temperature on the predicted power, on the test samples
    X = np.asarray(td.X_test, dtype=np.float32)
    X_warmer = X.copy()
    X_warmer[:, td.inputs.index('weaSta_reaWeaTDryBul_y')] += 1.0
    effect = (trained.predict(X_warmer, verbose=0) - trained.predict(X, verbose=0)).ravel()
    print(f"{name:13s} test RMSE {td.metrics.kpi('test', 'rmse'):6.1f} W | power rises with T_amb on "
          f"{np.mean(effect > 1e-6):.0%} of the test samples")
