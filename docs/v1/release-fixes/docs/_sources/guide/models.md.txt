# Models

All models share `model.pipeline(td)`: build, train, evaluate. The trained model is returned and kept as
`model.trained_model`, `Run.save` stores it with its configs.

## Single step models

Predict the next time step (`shift=1`), or an output at the time of the inputs (`shift=0`). Applied recursively
in the MPC when they predict one of their inputs. The keras networks have one output; train one model per output.

| Model | Description |
|---|---|
| `LinearRegressionModel()` | linear regression (scikit-learn, saved as joblib) |
| `ClassicalANNModel(n_layers, n_neurons, activation_function)` | feed forward network |
| `CMNNModel(monotonies, activation_split)` | constrained monotonic neural network |
| `RBFModel(n_neurons)` | radial basis function network |
| `LinANNModel()` | linear regression plus an RBF network on its residuals |
| `PINNModel(physics_targets)` | physics-informed loss on physics targets, see below |
| `ModularANN(architecture, solver='adam', 'ols' or 'qp')` | any expression graph, see [Expressions](expressions.md) |
| `ModularANN(px.Linear(terms), solver='qp', shape=ShapeConstraints(...))` | constrained regression, see below |

Keras models share the training options `epochs`, `learning_rate`, `batch_size`,
`early_stopping_epochs` and `random_seed`. Early stopping needs validation data. `random_seed` seeds the weight
initialization and the batch order; it defaults to the global seed (`physXAI.set_seed`), else a new seed is drawn and
stored in the config. Unknown arguments raise a `TypeError`.

## Constrained regression

A linear regression on physical terms whose prediction is monotonic or convex in raw data columns, guaranteed on a
domain. The fit is a convex quadratic program: one global optimum, no seeds.

```python
from physXAI.models import ModularANN, ShapeConstraints

v = px.Vars()
u, T, T_amb = v.oveHeaPumY_u, v.reaTZon_y, v.weaSta_reaWeaTDryBul_y
model = ModularANN(
    px.Linear([u, u * T_amb, u * T, T_amb]),
    solver="qp",
    shape=ShapeConstraints(monotonicities={"oveHeaPumY_u": 1, "weaSta_reaWeaTDryBul_y": -1},
                           bounds={"weaSta_reaWeaTDryBul_y": (250.0, 310.0)}),
)
td = prep.pipeline(path, model=model)
model.pipeline(td)
td.physics["shape_constraints"]     # certificates, violations, binding share, cost of the physics
```

- **Constraints on raw columns:** `monotonicities={'x': 1 | -1}` constrains `dy/dx` of the whole model, e.g.
  `d/dT_amb (w1 · u · T_amb + w2 · T_amb) = w1 · u + w2`, which depends on `u`. `convexities={'x': 1 | -1}`
  constrains `d²y/dx²` (convex or concave along `x`, not the joint convexity in several columns). Every column is
  constrained separately, also lags (`u` and `u_lag1`).
- **Sign of single weights:** `px.Linear(terms, monotonicities={'term': 1})` bounds the weight of a term; this holds
  globally and is also kept by `recursive_training`.
- **Domain:** `domain='box'` (default) enforces the constraints on the box of the training data, or on `bounds`
  (e.g. the bounds of the MPC). The derivative is linear in the weights, so this is a semi-infinite program, solved
  with the exchange method: the least squares problem on a finite set of constraint points, the most violated points
  of the box (Sobol sample, local minimization) are added until no violation is left. `domain='data'` only enforces
  them on the training samples, e.g. when a physical relation only holds in the operating range of the data
  (more cold supply air only cools while `T_in < T`).
- **Certificates** per constraint: `exact` (the derivative is multilinear in its columns, e.g. linear, bilinear and
  quadratic terms: its minimum over the box lies in a corner, and all corners are constraint points, so the
  constraint holds on the whole box), `verified` (no violation found on the box), `data` (holds on the training
  samples), `violated` (no convergence after `max_iter`, with warning).
- **Diagnosis:** the train RMSE with and without the constraints (the cost of the physics) and the share of training
  samples where a constraint binds: constraints binding on many samples show where the data contradicts the physics.
  The report contains a table of it.
- **Numerics:** terms and target are standardized, the least squares problem is solved in the coordinates of its QR
  decomposition (no squared condition number), a tiny ridge (`ridge=1e-10`) makes the solution unique for collinear
  terms. Terms that are not defined everywhere in the box (e.g. a division by zero) are left out at those points
  with a warning; restrict the box with `bounds`.
- `shape` cannot be combined with `recursive_training` (gradient descent would not keep it) or a physics residual.
  Closed form solvers (`ols`, `qp`) have no `online_pipeline`: train a new model on the combined data.
- The model is exported like any modular model: the keras model of the terms with one dense layer.

See `examples/05_constrained_regression.py`.

## Physics-informed neural networks

A physics target is a data column computed from a physical model of the output. The network keeps one
output; the physics targets only enter the loss:

```python
v = px.Vars()
physics = v.u_hp * 10000 * (v.T_room + 15 - v.T_amb) / ((v.T_room + 15) * 0.55)
prep = PreprocessingSingleStep(inputs=[...], output='P_el', columns={'pinn': physics})
m = PINNModel(physics_targets={'pinn': 0.5})
td = prep.pipeline(path, model=m)     # the model adds its physics targets to the loss columns
m.pipeline(td)
```

- Loss: `mse(ŷ, y)/s + Σ w_i · mse(ŷ, p_i)/s` with the characteristic scale `s = var(y_train)`
  (`normalize_terms=True`). The data term is 1 − R² on the training data, the weight `w_i` the relative
  importance of the physics. The terms are not averaged, adding a physics term does not change the weight of
  the data.
- The weight encodes the trust in the physical model and is chosen by validation (e.g. the error over the
  horizon, see [Comparing models](evaluation.md#comparing-models)). Adaptive weighting schemes (e.g. GradNorm,
  learning rate annealing) target PDE residuals of very different magnitudes and are not used.
- Early stopping monitors the validation **data** loss (`early_stopping_on='data'`), so the selected model does
  not depend on the physics weights; `'total'` monitors the weighted loss.
- History and report hold every term (`data_loss`, `<target>_loss`); `td.metrics.physics` the disagreement with
  each physics target per split.
- Physics targets are taken at the predicted time, like the output. For `Change(T)` and a physical model of
  the step T(t) → T(t+1) computed from the data at t, shift the column: `columns={'dT': step.lag(1)}`.
- The saved model has one output and is exported to AgentLib-MPC like any other model.

## Multi step models

`RNNModel(rnn_layer='RNN' | 'GRU' | 'LSTM', rnn_units)` predicts a trajectory, see
[Recurrent models](rnn.md). `PCNNModel(heating, cooling, losses)` is a physically consistent recurrent model, see
[Physics-informed models](physics.md).

Every single step keras model takes a physics residual with trainable physical parameters
(`physics=PhysicsResidual(...)`), see [Physics-informed models](physics.md).

## Saving and loading

```python
run.save("model", prep, model, td)                       # recommended
model.pipeline(td, save_path="results/x/model")      # only the model file and its report
trained = model.load_model("results/x/model.keras")
model = AbstractModel.model_from_config(json.load(open("results/x/model_model.json")))
```

`online_pipeline(td, load_path)` loads a saved model and trains it further on new data.

## Reports

`pipeline` does not open plots. `model.plot(td)` returns the report as plotly figure,
`pipeline(td, show_plot=True)` opens it, and `Run.save` writes it as `<name>.html`.
