"""
10 - Physically consistent neural network (PCNN)
================================================

Following Di Natale et al. (2022), Physically Consistent Neural Networks for building thermal modeling: Theory
and analysis, Applied Energy 325, 119806. The temperature is the sum of an energy accumulator E and the unforced
dynamics D:

    E(k+1) = E(k) + a * u_hp(k) + b * (T_amb(k) - T(k))      physics-inspired, linear, a, b > 0
    D(k+1) = D(k) + f(D(k), radiation(k))                     LSTM, black box
    T = D + E

D does not see the heat pump or the ambient temperature, so the predicted temperature rises with the heat pump
signal and with the ambient temperature over the whole horizon, whatever the LSTM learns. Unlike the PINN
(example 08), this is a guarantee, also on inputs outside the training data.

- a and b start at rates estimated from the training data (None: nonnegative least squares of the single step
  energy balance) or at rules of thumb per hour, and are trained with the network on the error of the trajectory.
  In a seed study on this data the estimated start values gave a lower error over 12 h than the rules of thumb.
- The predicted temperature is fed back (closed loop), the model warms up on the measured past (example 07).
- The step model is saved like an RNN and can be used in AgentLib-MPC.
"""

from pathlib import Path

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import PCNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '10_pcnn', overwrite=True, log_level='warning')
physXAI.set_seed(0)

prep = PreprocessingMultiStep(inputs=['oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'reaTZon_y'],
                              output=px.Change('reaTZon_y'), warmup_width='3h', label_width='12h',
                              eval_horizon='24h', time_index_col='time')
td = prep.pipeline(DATA)

model = PCNNModel(
    heating={'oveHeaPumY_u': None},            # from the data; rule of thumb: +1 K in 2 h at full power -> 0.5 K/h
    losses={'weaSta_reaWeaTDryBul_y': None},   # from the data; rule of thumb: -1.5 K in 6 h at 25 K colder -> 0.01 1/h
    rnn_layer='LSTM', rnn_units=16, epochs=100, early_stopping_epochs=20,
)
model.pipeline(td)
run.save('model', prep, model, td)  # configs, metrics (incl. the physical parameters), step model and report

print('Initial rates estimated from the training data (per hour):', td.physics['initial_rates'])
print('Trained physical parameters (per hour):', td.physics['parameters'])
print(f"PCNN test RMSE of T after 12 h: {td.horizon_metrics.kpis['test']['rmse'][47]:.3f} K")
