# Models

All models share `model.pipeline(td)`: build, train, evaluate. The trained model is returned and kept as
`model.trained_model`, `Run.save` stores it with its configs.

## Single step models

Predict the next time step. Applied recursively in the MPC when they predict one of their inputs.

| Model | Description |
|---|---|
| `LinearRegressionModel()` | linear regression (scikit-learn, saved as joblib) |
| `ClassicalANNModel(n_layers, n_neurons, activation_function)` | feed forward network |
| `CMNNModel(monotonies, activation_split)` | constrained monotonic neural network |
| `RBFModel(n_neurons)` | radial basis function network |
| `LinANNModel()` | linear regression plus an RBF network on its residuals |
| `PINNModel(physics_targets)` | physics-informed loss on physics targets, see below |
| `ModularANN(architecture, solver='adam' or 'ols')` | any expression graph, see [Expressions](expressions.md) |
| `ConstrainedRegression(inputs, monotonies, convexities)` | constrained linear regression on expressions |

Keras models share the training options `epochs`, `learning_rate`, `batch_size`,
`early_stopping_epochs` and `random_seed`. Early stopping needs validation data.

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
[Recurrent models](rnn.md).

## Saving and loading

```python
run.save("model", prep, model, td)                       # recommended
model.pipeline(td, save_path="stored_data/x/model")      # only the model file and its report
trained = model.load_model("stored_data/x/model.keras")
model = AbstractModel.model_from_config(json.load(open("stored_data/x/model_model.json")))
```

`online_pipeline(td, load_path)` loads a saved model and trains it further on new data.

## Reports

`pipeline` does not open plots. `model.plot(td)` returns the report as plotly figure,
`pipeline(td, show_plot=True)` opens it, and `Run.save` writes it as `<name>.html`.
