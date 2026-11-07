"""
05 - Constrained regression with certified shape constraints
============================================================

A linear regression on physical terms whose prediction is monotonic in raw data columns, guaranteed on a domain.
The fit is a convex quadratic program: one global optimum, no seeds, fast.

The electrical power of the heat pump is a linear combination of terms of its signal and the temperatures.
Physically, the power rises with the signal and falls with the ambient temperature (higher efficiency). These are
constraints on the derivative of the whole model with respect to the raw columns, e.g. d/dT_amb of the term
`u_hp * T_amb`, so they have to hold for every value of u_hp, not only on the training samples:

- `ShapeConstraints(monotonicities=..., convexities=...)` with `domain='box'` enforces them on the box of the training
  data (or `bounds`, e.g. the bounds of the MPC) with the exchange method: violated points of the box are added as
  constraints until there are none.
- The certificate says what is guaranteed: 'exact' (the derivative is multilinear, e.g. bilinear terms, so the box
  corners suffice: holds on the whole box), 'verified' (no violation found by sampling and local search), 'data'
  (holds on the training samples) or 'violated'.
- The diagnosis compares the train RMSE with and without the constraints: the cost of the physics. Constraints that
  bind on many samples show where the data contradicts the physics.
- `px.Linear(monotonicities=...)` additionally bounds the sign of single weights.

The constrained model is compared with the unconstrained least squares fit (solver='ols') of the same terms, and the
saved keras model (the one exported to AgentLib-MPC) is checked independently with automatic differentiation.

Here the data contradicts the physics: it was recorded under PID control, the heat pump signal correlates with the
temperatures, and the unconstrained fit lets the power rise with T_amb everywhere. The constraint on T_amb binds on
all samples, the constrained model does not depend on T_amb at all. Data with more excitation resolves this.
"""

from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN, ShapeConstraints
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '05_constrained_regression', overwrite=True, log_level='warning')
physXAI.set_seed(0)

v = px.Vars()
u_hp, on = v.oveHeaPumY_u, v['Func(logistic)']
T, T_amb = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y
terms = [u_hp, u_hp * T_amb, u_hp * T, on, T_amb]
shape = ShapeConstraints(monotonicities={'oveHeaPumY_u': 1, 'weaSta_reaWeaTDryBul_y': -1})

models = {
    'constrained': ModularANN(px.Linear(terms), solver='qp', shape=shape),
    'unconstrained': ModularANN(px.Linear(terms), solver='ols'),
}
prep = PreprocessingSingleStep(output='reaPHeaPum_y', time_index_col='time')
for name, model in models.items():
    td = prep.pipeline(DATA, model=model)
    model.pipeline(td)
    run.save(name, prep, model, td)  # the report <name>.html contains the table of the shape constraints
    print(f"{name:13s} test RMSE {td.metrics.kpi('test', 'rmse'):6.1f} W")
    if name == 'constrained':
        result = td.physics['shape_constraints']
        for c in result['constraints']:
            print(f"  {c['type']} {c['column']} ({c['sign']:+d}): {c['certificate']}, binding on "
                  f"{c['active_share_data']:.0%} of the training samples")
        print(f"  cost of the physics: train RMSE {result['rmse_train']:.1f} W instead of "
              f"{result['rmse_train_unconstrained']:.1f} W")
        box = result['domain']['box']

# --- Independent check of the saved models: dP/dT_amb on random points of the box ------------------------------
rng = np.random.default_rng(1)
X = np.column_stack([rng.uniform(*box[column], 10000) for column in td.inputs]).astype(np.float32)
j = td.inputs.index('weaSta_reaWeaTDryBul_y')
for name in models:
    saved = keras.saving.load_model(run.path / f'{name}.keras')
    x = tf.constant(X)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = saved(x)
    gradient = tape.gradient(prediction, x).numpy()[:, j]
    print(f"{name:13s} power rises with T_amb on {np.mean(gradient > 1e-6):.1%} of 10000 random points of the box")
