"""
03 - Neural networks and monotonicity constraints
=================================================

The electrical power of the heat pump is predicted from its signal and the temperatures. The power is no input of
the model, so the model is not applied recursively (an output model).

All keras models share the options epochs, learning_rate, batch_size, early_stopping_epochs and random_seed and
are trained with `model.pipeline(td)`:

- `ClassicalANNModel`: feed forward network,
- `CMNNModel`: constrained monotonic neural network, the prediction is monotonic in selected inputs by construction
  (here: the power rises with the heat pump signal and falls with the ambient temperature), which an MPC relies on,
- `RBFModel`: radial basis functions, the prediction extrapolates to 0 (useful for residuals),
- `LinANNModel`: linear regression plus an RBF network on its residuals.

At the end the saved CMNN is trained further on new data (online learning).
"""

from pathlib import Path

import physXAI
from physXAI.io import Run
from physXAI.models import ClassicalANNModel, CMNNModel, LinANNModel, RBFModel
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bestest_hydronic_heat_pump" / "pid_data.csv"

run = Run(ROOT / "results", "03_neural_networks", overwrite=True, log_level="warning")
physXAI.set_seed(0)

inputs = ["oveHeaPumY_u", "Func(logistic)", "weaSta_reaWeaTDryBul_y", "reaTZon_y"]
prep = PreprocessingSingleStep(inputs=inputs, output="reaPHeaPum_y", time_index_col="time")
td = prep.pipeline(DATA)

models = {
    "ann": ClassicalANNModel(
        n_layers=1, n_neurons=32, activation_function="softplus", epochs=200, early_stopping_epochs=20
    ),
    "cmnn": CMNNModel(
        monotonies={  # 1 increasing, -1 decreasing, 0 no constraint
            "oveHeaPumY_u": 1,
            "Func(logistic)": 1,
            "weaSta_reaWeaTDryBul_y": -1,
            "reaTZon_y": 1,
        },
        activation_split=[1, 1, 1],  # proportions of convex, concave and saturated activations
        epochs=200,
        early_stopping_epochs=20,
    ),
    "rbf": RBFModel(n_neurons=32, epochs=200, early_stopping_epochs=20),
    "linann": LinANNModel(epochs=200, early_stopping_epochs=20),
}
for name, model in models.items():
    td = prep.pipeline(DATA)  # the same split for every model (same seed), fresh predictions and metrics
    model.pipeline(td)
    run.save(name, prep, model, td)  # <name>_*.json, <name>.keras and the report <name>.html
    print(f"{name:7s} test RMSE {td.metrics.kpi('test', 'rmse'):7.1f} W, R2 {td.metrics.kpi('test', 'r2'):.3f}")

# --- Online learning --------------------------------------------------------------------------------------------
# A saved model is loaded and trained further, e.g. on data recorded during operation (here: the same data).
cmnn = models["cmnn"]
cmnn.epochs = 20
td = prep.pipeline(DATA)
cmnn.online_pipeline(td, load_path=str(run.path / "cmnn.keras"))
run.save("cmnn_online", prep, cmnn, td)
print(f"cmnn after online learning: test RMSE {td.metrics.kpi('test', 'rmse'):.1f} W")
