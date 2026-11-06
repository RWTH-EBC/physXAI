# Examples

A tutorial in 13 scripts. Each script is self-contained and documented in its docstring; together they show the
features of physXAI 1.0 in a sensible reading order. All examples use the data of the BOPTEST case
[Bestest Hydronic Heat Pump](https://ibpsa.github.io/project1-boptest/testcases/ibpsa/testcases_ibpsa_bestest_hydronic_heat_pump/)
(`data/bestest_hydronic_heat_pump/pid_data.csv`, 14 days, one sample every 15 min): the air temperature of the zone
and the electrical power of the heat pump.

```bash
python examples/01_first_model.py
```

The scripts find the data relative to the repository and can be run from any directory. Results (configs,
metrics, models and html reports) are written to `results/<script>/`.

| Script | Topic | Shows |
|---|---|---|
| `01_first_model.py` | A first model | `Run`, `set_seed`, `PreprocessingSingleStep`, `Change(T)` and lags, metrics, error over the horizon, `run.save`, loading a run |
| `02_data_and_split.py` | Data pipeline | `lags`, derived `columns`, split strategies (blocks, holdout, random rows) and leakage, saving training data |
| `03_neural_networks.py` | Keras models | `ClassicalANNModel`, `CMNNModel` (monotonicity), `RBFModel`, `LinANNModel`, online learning |
| `04_expressions_modular.py` | Physics inside the model | `px` expressions, `ModularANN` (`solver='ols'` and neural blocks), inputs taken from the model, `px.Pretrained` and `px.Polynomial` |
| `05_constrained_regression.py` | Constrained regression | `ModularANN(solver='qp')` with `ShapeConstraints`: monotonicity on raw columns guaranteed on the box, certificates, cost of the physics, check of the exported model |
| `06_multi_step.py` | Prediction horizon | `eval_horizon` (recursive evaluation), `RecursiveTraining` |
| `07_rnn.py` | Recurrent models | `PreprocessingMultiStep`, `RNNModel` with warmup, closed loop and trajectory loss |
| `08_pinn.py` | Physics-informed loss | `PINNModel` with a physics target |
| `09_physics_residual.py` | Trainable physics | `PhysicsResidual`, `px.positive`, `px.Param`, grey-box calibration of a 1R1C model |
| `10_pcnn.py` | Physically consistent | `PCNNModel` (Di Natale et al. 2022) |
| `11_compare_models.py` | Model comparison | `repeat_training` over seeds, `ModelComparison`, loading saved runs |
| `12_feature_selection.py` | Feature selection | `FeatureSelection` on the error over the horizon, plausibility of controls (experimental) |
| `13_agentlib_mpc.py` | AgentLib-MPC | a training script `train_model(...)`, `save_for_agentlib`, `check_agentlib_compatibility` |

Notes:

- The data covers only 14 days. The split shortens its blocks to get enough of them and warns about it.
- The examples train with the full number of epochs. The CI runs every example with one epoch per training
  (`python tests/smoke_examples.py`).
- New examples are added here with the next number and listed in this table.
