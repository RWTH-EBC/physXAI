from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import PCNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

"""
Physically consistent neural network (PCNN, Di Natale et al. 2022, Applied Energy 325, 119806) for the air
temperature of the Boptest case.

- The temperature is the sum of an energy accumulator E and the unforced dynamics D:
    E(k+1) = E(k) + a * u_hp(k) + b * (T_amb(k) - T(k))      (physics-inspired, linear, a, b > 0)
    D(k+1) = D(k) + f(D(k), radiation(k))                     (LSTM, black box)
  D does not see the heat pump or the ambient temperature, so the predicted temperature rises with the heat
  pump signal and with the ambient temperature over the whole horizon, whatever the LSTM learns.
- a and b start at rules of thumb (per hour) and are trained with the network on the error of the
  trajectory over 12 h. The predicted temperature is fed back (closed loop, as in an MPC), the model warms up
  on the 3 h before each prediction.
- The step model is saved like an RNN and can be used in AgentLib-MPC.
"""
run = Run('stored_data', 'TAir_pcnn', overwrite=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

inputs = ['oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'reaTZon_y']
prep = PreprocessingMultiStep(inputs=inputs, output=px.Change('reaTZon_y'), warmup_width='3h', label_width='12h',
                              eval_horizon='24h', time_index_col='time')
td = prep.pipeline(file_path)

m = PCNNModel(
    heating={'oveHeaPumY_u': 0.5},              # about +1 K in 2 h with the heat pump at full power
    losses={'weaSta_reaWeaTDryBul_y': 0.01},    # about -1.5 K in 6 h when it is 25 K colder outside
    rnn_layer='LSTM', rnn_units=16, epochs=100, early_stopping_epochs=20,
)
m.pipeline(td)
print('Trained physical parameters (per hour):', td.physics['parameters'])

# Configs, metrics (incl. the physical parameters), step model and report
run.save('model', prep, m, td)
