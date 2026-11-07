# Physics-informed models

physXAI offers five ways to bring physical knowledge into a model. They differ in where the physics acts and
what it guarantees:

| Approach | Physics | Guarantee | Exported model |
|---|---|---|---|
| Structure ([expressions](expressions.md)) | inside the model, e.g. `px.Linear(monotonicities=...)` | by construction | the model including the physics |
| [Constrained regression](models.md#constrained-regression) | monotonicity / convexity in raw columns of a linear regression on physical terms | on the domain (box or data), with certificate | the model |
| [PINN](#pinn-physics-targets) | data column of a physical model with fixed parameters, in the loss | none (soft) | the network |
| [Physics residual](#physics-residuals-with-trainable-parameters) | residual with trainable physical parameters, in the loss | none (soft) | the network |
| [PCNN](#physically-consistent-neural-network-pcnn) | linear energy balance in parallel to a recurrent network | sign of the effects of controls and temperatures over the whole horizon | step model (multi step) |

## PINN: physics targets

```python
v = px.Vars()
physics = v.u_hp * 10000 * (v.T_room + 15 - v.T_amb) / ((v.T_room + 15) * 0.55)
prep = PreprocessingSingleStep(inputs=[...], output='P_el', columns={'pinn': physics})
m = PINNModel(physics_targets={'pinn': 0.5})
td = prep.pipeline(path, model=m)
m.pipeline(td)
```

The loss is `mse(ŷ, y)/s + Σ w_i · mse(ŷ, p_i)/s` with `s = var(y_train)`, see [Models](models.md). Physics
targets are taken at the predicted time; for a step model of `Change(T)`, shift the column by one step
(`columns={'dT': step.lag(1)}`).

## Physics residuals with trainable parameters

A residual `r(x, y; θ)` that should be zero, written with `px.Prediction()` (the output of the model) and
trainable parameters, can be added to every single step keras model:

```python
from physXAI.models.ann.pinn.physics import PhysicsResidual

dt = 900.0
v = px.Vars()
tau, K_u = px.positive(20 * 3600.0, 'tau'), px.positive(20.0, 'K_u')
K_0 = px.Param(0.0, name='K_0')
dT = (1 - px.exp(-dt / tau)) * (v.T_amb + K_u * v.u_hp + K_0 - v.T)       # 1R1C, exact zero order hold

m = ClassicalANNModel(physics=PhysicsResidual(px.Prediction() - dT, weight=0.5, name='rc'),
                      physics_fit='calibrate')
td = prep.pipeline(path, model=m)   # the columns of the residual become loss columns
m.pipeline(td)
m.physics_parameters                # {'tau': ..., 'K_u': ..., 'K_0': ..., 'log_tau': ..., ...}
```

- `px.positive(value, name)` is `value · exp(p)` with a trainable `p = 0`: positive, scaled by the physical
  initial value, changed relatively by the training. `px.Param` is a plain (signed) parameter.
- The residual is evaluated with the prediction (`r(x, ŷ)`, the physics term of the network) and with the
  measured target (`r(x, y)`, the fit of the physical model to the data), with shared parameters.
- `physics_fit='calibrate'` (default): the parameters are first fitted to the data (grey-box identification),
  then fixed while the network is trained. When parameters are trained together with the network only through
  the physics term, they follow the network and lose their physical meaning. `'joint'` trains both together
  (including the fit term), `'fixed'` keeps the initial values.
- The calibration is a nonlinear least squares problem in the few parameters, `min Σ r(x, y; θ)²` on the training
  data. It is solved deterministically with a trust region method (`scipy.optimize.least_squares`) and the exact
  Jacobian of the residual (CasADi), independent of the seed, the batch order and a learning rate, and converged to
  the minimum (gradient descent with early stopping stops somewhere in the flat valley of correlated parameters, so
  its estimates depend on the seed). `td.physics['calibration_record']` holds the result of the solver. Residuals
  without CasADi form (e.g. a `px.Param` without initial value) are calibrated by gradient descent
  (`physics_learning_rate`).
- History: `data_loss`, `<name>_loss`, `<name>_fit_loss`; results in `td.physics` (parameters, fit per split,
  history of the calibration). Only parameters are trainable in residuals, no blocks.
- The variables of the residual are data columns at the time of the inputs; `px.Target(name)` is a column at the
  predicted time. They do not have to be inputs of the model (the MPC never sees them): only the network is
  saved and exported.
- Discretize physical models exactly over the time step (zero order hold), an explicit Euler step is inaccurate
  and unstable for time constants of the order of the time step. Only the time constant and steady state gains
  are identifiable from temperatures, not R and C separately. Data recorded under feedback control correlate the
  control with the temperatures, which correlates the estimates.
- With `RecursiveTraining`, the residual acts in the single step phase; the multi step phase uses the error of
  the recursive prediction.

See `examples/09_physics_residual.py`.

## Physically consistent neural network (PCNN)

Following Di Natale et al. (2022), *Physically Consistent Neural Networks for building thermal modeling: Theory
and analysis*, Applied Energy 325, 119806, the temperature is the sum of an energy accumulator `E` and the unforced
dynamics `D`:

```
E(k+1) = E(k) + Σ a·u_heating(k) − Σ d·u_cooling(k) + Σ b·(T_x(k) − T(k))
D(k+1) = D(k) + f(D(k), x(k))          T = D + E
```

`f` is a recurrent network on `D` and the remaining inputs `x` (e.g. solar radiation), which excludes the controls
and the temperatures `T_x`. Because `D` does not depend on them,

```
∂T(k+i)/∂u(k+i−j) = (1 − Σ b)^(j−1) · a > 0
```

for any network: the predicted temperature rises with heating and with `T_x` and falls with cooling over the
whole horizon, also on unseen inputs. The parameters are positive by construction (`s = s₀ · s̃`, `s̃ ≥ 0`,
starting at 1), `1 − Σ b > 0` is checked after the training (`consistency_margin`).

```python
from physXAI import expr as px
from physXAI.models import PCNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

prep = PreprocessingMultiStep(inputs=['oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'reaTZon_y'],
                              output=px.Change('reaTZon_y'), warmup_width='3h', label_width='12h', eval_horizon='24h',
                              time_index_col='time')   # the time column of the BOPTEST data
m = PCNNModel(heating={'oveHeaPumY_u': 0.5},              # +1 K in 2 h at full power: 0.5 K/h
              losses={'weaSta_reaWeaTDryBul_y': 0.01},    # -1.5 K in 6 h at 25 K colder outside: 0.01 1/h
              rnn_layer='LSTM', rnn_units=16)
td = prep.pipeline(path)
m.pipeline(td)
td.physics['parameters']        # trained rates per hour and the consistency margin
```

- The initial rates are given per hour (heating, cooling: K/h per unit of the input; losses: 1/h) and scaled to the
  time step: rules of thumb, or `None` to estimate them from the training data by nonnegative least squares of the
  single step energy balance `T(k+1) − T(k) = Σ a·u_heating − Σ d·u_cooling + Σ b·(T_x − T) + c` (the offset c
  belongs to D). `td.physics['initial_rates']` reports them. In a seed study on the BOPTEST data, the estimated start
  values reduced the RMSE over 12 h in 8 of 10 paired runs (−0.05 K on average) and the bias at the end of the
  horizon. A rate of 0 would switch a term off for good, so a term without positive effect in the data starts at a
  weak rate (warning).
- T is an input and fed back (closed loop); the model warms up on the measured past, as in AgentLib-MPC, and is
  trained on the trajectory over `label_width`.
- The saved step model `[sequence, E, *states of f] → [ΔT, E, *states of f]` consists of layers AgentLib-MPC
  translates to CasADi; it is used like an RNN.
- Consistency only holds for the inputs of the physics module (paper, section 6.3), e.g. not for solar gains
  processed by `f`. Time features (time of day, month) have to be data columns.

See `examples/10_pcnn.py`.
