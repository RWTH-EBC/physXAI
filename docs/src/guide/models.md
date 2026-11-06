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
| `PINNModel(pinn_weights)` | physics informed loss on additional targets (`columns`) |
| `ModularANN(architecture, solver='adam' or 'ols')` | any expression graph, see [Expressions](expressions.md) |
| `ConstrainedRegression(inputs, monotonies, convexities)` | constrained linear regression on expressions |

Keras models share the training options `epochs`, `learning_rate`, `batch_size`,
`early_stopping_epochs` and `random_seed`. Early stopping needs validation data.

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
