"""
09 - Physics residuals with trainable physical parameters
=========================================================

A neural network for the change of the air temperature, regularized by a 1R1C model whose parameters are trained.

- 1R1C model, discretized exactly over the time step (zero order hold, no explicit Euler step):
      T(t+1) - T(t) = (1 - exp(-dt / tau)) * (T_amb + K_u * u_hp + K_rad * radiation + K_0 - T)
  Only the time constant tau and the steady state gains K are identifiable from temperatures, not R and C.
- `px.positive(value, name)` is a positive parameter (value * exp(p), p trainable), `px.Param` a signed one.
- The residual `prediction - dT_1R1C` (`px.Prediction()` is the output of the network) enters the loss with
  weight 0.5 relative to the data term.
- physics_fit='calibrate' (default): the parameters are fitted to the data first (grey-box identification, a
  deterministic least squares fit, independent of the seed), then fixed while the network is trained, so they keep
  their physical meaning instead of following the network. 'joint' trains both together, 'fixed' keeps the initial
  values.
- Only the network is saved (and exported to AgentLib-MPC), the parameters are part of the metrics and the report.
- The data was recorded under PID control: the heat pump signal correlates with T_amb - T, so the estimates of
  tau and K_u are correlated. Data with more excitation gives better estimates.
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ClassicalANNModel
from physXAI.models.ann.pinn.physics import PhysicsResidual
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '09_physics_residual', overwrite=True, log_level='warning')
physXAI.set_seed(0)
dt = 900.0  # time step of the data in s

v = px.Vars()
T, T_amb, u_hp, radiation = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u, v.weaSta_reaWeaHDirNor_y
tau = px.positive(20 * 3600.0, 'tau')  # time constant in s
K_u = px.positive(20.0, 'K_u')         # steady state temperature rise at full heat pump power in K
K_rad = px.positive(0.01, 'K_rad')     # K per W/m2
K_0 = px.Param(0.0, name='K_0')        # internal gains in K
dT_1R1C = (1 - px.exp(-dt / tau)) * (T_amb + K_u * u_hp + K_rad * radiation + K_0 - T)

model = ClassicalANNModel(
    n_neurons=16, epochs=200, early_stopping_epochs=20,
    physics=PhysicsResidual(px.Prediction() - dT_1R1C, weight=0.5, name='1R1C'),
    physics_fit='calibrate',
)
prep = PreprocessingSingleStep(inputs=['oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'reaTZon_y'],
                               output=px.Change('reaTZon_y'), time_index_col='time', eval_horizon='12h')
td = prep.pipeline(DATA, model=model)  # the columns of the residual are added as loss columns
model.pipeline(td)
run.save('model', prep, model, td)

parameters = {k: round(float(p), 4) for k, p in model.physics_parameters.items() if not k.startswith('log_')}
print('Identified 1R1C parameters:', parameters, f"(tau = {parameters['tau'] / 3600:.1f} h)")
