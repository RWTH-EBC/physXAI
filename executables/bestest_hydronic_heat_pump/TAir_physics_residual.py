from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import ClassicalANNModel
from physXAI.models.ann.pinn.physics import PhysicsResidual
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

"""
Neural network for the change of the air temperature, regularized by a 1R1C model with trainable parameters.

- 1R1C model, discretized exactly over the time step (zero order hold, no explicit Euler):
      T(t+1) - T(t) = (1 - exp(-dt / tau)) * (T_amb + K_u * u_hp + K_rad * radiation + K_0 - T)
  Only the time constant tau and the steady state gains K are identifiable from data, not R and C separately.
- The residual `prediction - dT_1R1C` enters the loss with weight 0.5 (relative to the data term).
- physics_fit='calibrate': the parameters are fitted to the data first (grey-box identification), then
  fixed while the network is trained, so they keep their physical meaning instead of following the network.
- Only the network is saved; the trained parameters are part of the training data json and the report.
- The Boptest data was recorded under PID control: the heat pump signal correlates with T_amb - T, the
  estimates of tau and K_u are correlated. Data with more excitation give better estimates.
"""
run = Run('stored_data', 'TAir_physics_residual', overwrite=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"
dt = 900.0  # time step of the data in seconds

v = px.Vars()
T, T_amb, u_hp, radiation = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u, v.weaSta_reaWeaHDirNor_y
tau = px.positive(20 * 3600.0, 'tau')        # time constant in s
K_u = px.positive(20.0, 'K_u')               # steady state temperature rise at full heat pump power in K
K_rad = px.positive(0.01, 'K_rad')           # K per W/m2
K_0 = px.Param(0.0, name='K_0')              # internal gains in K
dT_1R1C = (1 - px.exp(-dt / tau)) * (T_amb + K_u * u_hp + K_rad * radiation + K_0 - T)

m = ClassicalANNModel(
    n_neurons=16, epochs=200, early_stopping_epochs=20,
    physics=PhysicsResidual(px.Prediction() - dT_1R1C, weight=0.5, name='1R1C'),
    physics_fit='calibrate',
)

prep = PreprocessingSingleStep(inputs=['oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'reaTZon_y'],
                               output=px.Change('reaTZon_y'), time_index_col='time', eval_horizon='12h')
td = prep.pipeline(file_path, model=m)  # the columns of the residual are added as loss columns

m.pipeline(td)
print('Trained 1R1C parameters:', {k: v for k, v in m.physics_parameters.items() if not k.startswith('log_')})

run.save('model', prep, m, td)
