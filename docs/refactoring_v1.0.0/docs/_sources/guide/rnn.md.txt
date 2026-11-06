# Recurrent models

A recurrent model carries its memory in hidden states instead of lagged inputs.

```python
from physXAI import expr as px
from physXAI.models.ann.ann_design import RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep

prep = PreprocessingMultiStep(
    inputs=["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "oveHeaPumY_u", "reaTZon_y"],
    output=px.Change("reaTZon_y"),
    label_width="12h",      # predicted (and trained) horizon
    warmup_width="6h",      # measured past the model warms up on
    eval_horizon="24h",     # evaluation beyond the training horizon
    time_index_col="time",
)
td = prep.pipeline(path)
model = RNNModel(rnn_layer="LSTM", rnn_units=16, epochs=100, early_stopping_epochs=20)
model.pipeline(td)
```

- **Warmup:** every sequence starts from zero states and runs over `warmup_width` measured steps, which are
  not part of the loss. AgentLib-MPC warms the model up the same way. There is no initialization model.
- **Change outputs:** with `Change(T)` the model predicts the trajectory of `T`: the measured start value plus
  the accumulated changes. It is trained on this trajectory (`loss_on='trajectory'`, or `'outputs'` for the
  changes), optionally with `first_step_weight` or `step_weights`.
- **Closed loop:** if `T` is also an input, the predicted value is fed back in training and evaluation, as in
  the MPC.
- **Horizon:** choose `label_width` longer than the MPC horizon. `eval_horizon` shows the stability beyond it.
- **Export:** the saved `.keras` file is the step model `[sequence, *states] -> [prediction, *states]` that
  AgentLib-MPC applies step by step.

For a model whose prediction provably rises with heating and outside temperature over the whole horizon,
see the PCNN in [Physics-informed models](physics.md).

`feature_architecture` adds an expression graph (e.g. physical features) in front of the recurrent layer.
