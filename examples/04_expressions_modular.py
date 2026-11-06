"""
04 - Expressions: physics inside the model
==========================================

`physXAI.expr` (imported as `px`) is one language for physically motivated terms and trainable blocks. The same
expression is evaluated on data (NumPy), built into the keras model and translated to CasADi for the MPC.

- Variables need no declaration: `v = px.Vars(); v.reaTZon_y`, `v['Func(logistic)']` for other names.
- Derived quantities placed inside the model are recomputed in every step of a recursive prediction, and the MPC
  only needs the raw inputs. The inputs of the preprocessing are taken from the model (no second list).
- Blocks: px.Linear, px.MLP, px.MonotoneMLP (CMNN), px.Polynomial, px.RBF, px.Param, px.Pretrained, ...

Part A fits a linear model of physical terms in closed form (solver='ols'). Part B adds a small neural network
for the remaining dynamics. Part C is a modular model of the heat pump power. Part D reuses the trained model of
part B as a frozen block (px.Pretrained, e.g. to transfer it to a similar building) and trains a polynomial
correction (px.Polynomial) on top of it.
"""

from pathlib import Path

import pandas as pd

import physXAI
from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import ModularANN
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'bestest_hydronic_heat_pump' / 'pid_data.csv'

run = Run(ROOT / 'results', '04_expressions_modular', overwrite=True, log_level='warning')
physXAI.set_seed(0)

v = px.Vars()
T, T_amb = v.reaTZon_y, v.weaSta_reaWeaTDryBul_y
u_hp, radiation = v.oveHeaPumY_u, v.weaSta_reaWeaHDirNor_y

# An expression is evaluated on data like a formula
heat_loss = (T - T_amb).named('heat_loss')
df = pd.read_csv(DATA, sep=';', encoding='latin1')
print('T - T_amb of the first rows:', heat_loss.evaluate(df)[:3], '| free variables:', heat_loss.free_vars())

# --- A: linear model of physical terms, closed form -------------------------------------------------------------
linear = ModularANN(px.Linear([heat_loss, u_hp, u_hp.lag(1), radiation]), solver='ols')
prep = PreprocessingSingleStep(output=px.Change('reaTZon_y'), time_index_col='time', eval_horizon='12h')
td = prep.pipeline(DATA, model=linear)  # inputs from the model, incl. lags; the recursive feature T comes last
print('Inputs taken from the model:', td.inputs)
linear.pipeline(td)
run.save('linear_ols', prep, linear, td)
print(f"A linear (ols):  RMSE of T after 12 h {td.horizon_metrics.kpis['test']['rmse'][-1]:.3f} K")

# --- B: physical terms plus a neural network ---------------------------------------------------------------------
architecture = px.Linear([
    heat_loss,                              # heat losses through the envelope
    u_hp, u_hp.lag(1),                      # heat pump with delay
    radiation,                              # solar gains
    px.MLP([u_hp, T_amb, T], n_neurons=8),  # remaining dynamics
], normalize=True)
hybrid = ModularANN(architecture, epochs=200, early_stopping_epochs=20)
td = prep.pipeline(DATA, model=hybrid)
hybrid.pipeline(td)
run.save('hybrid', prep, hybrid, td)
hybrid_inputs = list(td.inputs)
print(f"B linear + MLP:  RMSE of T after 12 h {td.horizon_metrics.kpis['test']['rmse'][-1]:.3f} K")
# The whole graph is part of the model config (hybrid_model.json), shared nodes stay shared
print('Blocks in the config:', sorted({n['type'] for n in hybrid.get_config()['architecture']['nodes']}))

# --- C: heat pump power ------------------------------------------------------------------------------------------
logistic = v['Func(logistic)']
power = ModularANN(px.MLP([u_hp / logistic, logistic ** 2, T_amb, T]), rescale_output=True, epochs=200,
                   early_stopping_epochs=20)
prep_power = PreprocessingSingleStep(output='reaPHeaPum_y', time_index_col='time')
td = prep_power.pipeline(DATA, model=power)
power.pipeline(td)
run.save('power', prep_power, power, td)
print(f"C heat pump power: test RMSE {td.metrics.kpi('test', 'rmse'):.1f} W")

# --- D: a trained model as block, plus a correction -----------------------------------------------------------------
# The saved model of part B gets its inputs in the order it was trained with; it stays frozen (trainable=False).
# On the same data the correction has little to add; with the data of a similar building it adapts the model.
pretrained = px.Pretrained(run.path / 'hybrid.keras', [v[name] for name in hybrid_inputs])
correction = px.Polynomial([u_hp, T_amb], degree=2, interaction_degree=2, normalize=True)
# the correction is scaled by a parameter starting at 0, so the training starts at the pretrained model
transfer = ModularANN(pretrained + px.Param(0.0, name='k_correction') * correction, epochs=50, early_stopping_epochs=10)
td = prep.pipeline(DATA, model=transfer)
transfer.pipeline(td)
run.save('transfer', prep, transfer, td)
print(f"D pretrained + polynomial correction: RMSE of T after 12 h {td.horizon_metrics.kpis['test']['rmse'][-1]:.3f} K")
