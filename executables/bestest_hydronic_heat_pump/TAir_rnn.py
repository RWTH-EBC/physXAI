from physXAI import expr as px
from physXAI.io import Run
from physXAI.models.ann.ann_design import RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

"""
Creates a recurrent neural network (RNN) to predict the air temperature using the Boptest data.

- The RNN predicts the change of the air temperature. Its trajectory is the measured start value
  plus the accumulated changes, and the model is trained on that trajectory.
- The air temperature is also an input: during the prediction the predicted temperature is fed back
  (closed loop), as in an MPC.
- There is no separate initialization model. The model starts from zero states and warms up on the
  measured past (warmup_width), which is not part of the loss. AgentLib-MPC warms it up the same way.
- label_width (training horizon) should be longer than the MPC horizon, eval_horizon shows the error
  beyond it.
"""
# Directory of the run (stored_data/TAir_rnn)
run = Run('stored_data', 'TAir_rnn', overwrite=True)

# File path to data
file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

inputs = ['weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y', 'oveHeaPumY_u', 'reaTZon_y']
output = px.Change('reaTZon_y')

# Create Training data. For RNNs MultiStep training data is required
prep = PreprocessingMultiStep(inputs=inputs, output=output, label_width='12h', warmup_width='6h',
                              eval_horizon='24h', time_index_col='time')
# Process Training data
td = prep.pipeline(file_path)

# Create RNN model
m = RNNModel(epochs=100, rnn_layer='LSTM', rnn_units=16, early_stopping_epochs=20)

# Training pipeline. The saved model is the step model used by AgentLib-MPC.
model = m.pipeline(td)

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, m, td)
