from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.training import RecursiveTraining

"""
Single step model of the air temperature change, evaluated and trained over a horizon.

- The architecture combines physically motivated terms with a small neural network. All of it is
  part of the keras model, so it is recomputed in every step of a recursive prediction and an MPC
  only needs the raw inputs.
- `eval_horizon` evaluates the model recursively (as in an MPC): RMSE and bias per step.
- `RecursiveTraining` fine-tunes the model on the error over the horizon, with the first step weighted
  more than the following ones.
"""

# Directory of the run (stored_data/TAir_recursive)
run = Run('stored_data', 'TAir_recursive', overwrite=True)

file_path = r"data/bestest_hydronic_heat_pump/pid_data.csv"

v = px.Vars()
T, T_amb = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y
u_hp, radiation = v.oveHeaPumY_u, v.weaSta_reaWeaHDirNor_y

architecture = px.Linear([
    T - T_amb,                          # heat losses through the envelope
    u_hp, u_hp.lag(1),                  # heat pump (with delay)
    radiation,                          # solar gains
    px.MLP([u_hp, T_amb, T], n_neurons=8),  # residual dynamics
], normalize=True)

model = ModularANN(
    architecture,
    epochs=200,
    recursive_training=RecursiveTraining(horizon='6h', first_step_weight=0.25),
)

# The inputs are taken from the model. 'Change(reaTZon_y)' is derived from 'reaTZon_y'.
prep = PreprocessingSingleStep(output=px.Change('reaTZon_y'), time_index_col='time', eval_horizon='12h')
td = prep.pipeline(file_path, model=model)

# Training pipeline: single step training, recursive fine-tuning, single step and horizon evaluation
model.pipeline(td)

# Save the configs of preprocessing and model, the metrics (training_data.json), the model and its report
run.save('model', prep, model, td)
