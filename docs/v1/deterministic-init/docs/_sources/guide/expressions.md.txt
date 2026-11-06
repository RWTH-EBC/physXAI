# Expressions: physics inside the model

`physXAI.expr` is one language for physically motivated terms and trainable blocks. An expression can be
evaluated on data (NumPy / pandas), built into a Keras model and translated to CasADi for the MPC.

```python
from physXAI import expr as px

v = px.Vars()                       # v.T, v.T_in, v['Func(logistic)']: no declarations needed
vT = v.mDot * (v.T_in - v.T)        # heat flow of the supply air
```

## Inside the model (recommended)

Derived quantities are best placed inside the model. They are recomputed in every step of a recursive
prediction, and the MPC only needs the raw inputs:

```python
from physXAI.models import ModularANN

model = ModularANN(px.Linear([v.mDot, v.load, vT, v.T]), solver="ols")   # linear regression in closed form
prep = PreprocessingSingleStep(output=px.Change("T"))
td = prep.pipeline(path, model=model)    # the inputs are taken from the model
model.pipeline(td)
```

`prep.pipeline(path, model=model)` collects the free variables of the model (including lags like
`v.T.lag(1)`) and orders them as AgentLib-MPC expects: the recursive feature and its lags last.

## As data column (fallback)

If a quantity has to be a column of the data, e.g. the target of a PINN, it is added with `columns`:

```python
prep = PreprocessingSingleStep(inputs=["mDot", "vT", "T"], output="Change(T)", columns={"vT": vT})
```

The MPC then has to provide `vT` as input. In a recursive evaluation, columns depending on the predicted
feature are recomputed in every step.

## Building blocks

| Block | Purpose |
|---|---|
| `px.Linear(inputs, monotonicities=None, normalize=None)` | weighted sum, optionally with monotonic weights |
| `px.MLP(inputs, n_layers=1, n_neurons=32, activation='softplus')` | neural network |
| `px.MonotoneMLP(inputs, monotonicities, activation_split=None)` | constrained monotonic network (CMNN) |
| `px.Polynomial(inputs, degree=2, interaction_degree=1)` | polynomial features |
| `px.RBF(inputs, units=32)` | radial basis functions (extrapolate to 0, for residual models) |
| `px.Mean(inputs)` | mean of expressions |
| `px.Normalize(x)` | normalization with the training data statistics |
| `px.Param(init=None, trainable=True)` | (trainable) scalar parameter |
| `px.Pretrained(model, inputs, trainable=False)` | an already trained keras model |

Arithmetic (`+ - * / **`), `px.exp`, `px.relu`, `px.sigmoid`, `px.softplus`, `px.tanh` and constants
(`px.Const`) combine them. Blocks normalize their inputs; arithmetic works on physical values.

```python
v = px.Vars()
T, T_amb, u_hp = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y, v.oveHeaPumY_u
architecture = px.Linear([
    T - T_amb,                                # losses through the envelope
    u_hp, u_hp.lag(1),                        # heat pump with delay
    v.weaSta_reaWeaHDirNor_y,                 # solar gains
    px.MLP([u_hp, T_amb, T], n_neurons=8),    # residual dynamics
], normalize=True)
model = ModularANN(architecture, epochs=200)
```

Warm start (`warm_start=True`, default): when the architecture is a normalized `Linear` block, the training starts
at the least squares solution of its data only terms (with the sign bounds of `monotonicities`), the trainable terms
(here the `px.MLP`) start with weight 0. The model starts as the linear model and the network learns its residual.
A bad initialization can no longer spoil the training: without warm start, single seeds of this architecture
diverge.

Physical parameters: `px.Param(init)` is a trainable scalar, `px.positive(value, name)` a positive one
(`value · exp(p)`), `px.freeze(expr, values)` replaces the parameters by numbers. `px.Prediction()` (the model
output) and `px.Target(name)` (a column at the predicted time) are used in physics residuals, see
[Physics-informed models](physics.md).

The whole graph is stored in the model config (`model.get_config()`), shared nodes stay shared.
`ModularANN(px.Linear(terms), solver='qp', shape=ShapeConstraints(...))` fits a linear regression on expressions
whose prediction is monotonic or convex in raw columns on a whole domain, see
[constrained regression](models.md#constrained-regression).
