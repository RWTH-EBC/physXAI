"""
physXAI expression language: physical arithmetic and trainable blocks in one graph.

    >>> from physXAI import expr as px
    >>> v = px.Vars()
    >>> architecture = px.Linear([v.V_flow, v.V_flow * (v.T_sup - v.T), v.T])

See `physXAI.expr.core` for details.
"""

from physXAI.expr.blocks import (
    MLP,
    RBF,
    Block,
    Linear,
    Mean,
    MonotoneMLP,
    Normalize,
    Polynomial,
    Pretrained,
    product,
)
from physXAI.expr.core import (
    Activation,
    Add,
    BuildContext,
    Change,
    Const,
    Div,
    Expr,
    Mul,
    Param,
    Pow,
    Prediction,
    Sub,
    Target,
    Var,
    Vars,
    as_expr,
    exp,
    freeze,
    positive,
    relu,
    sigmoid,
    softplus,
    tanh,
)
from physXAI.expr.serialization import from_config, to_config

__all__ = [
    "Expr",
    "Var",
    "Vars",
    "Const",
    "Param",
    "Prediction",
    "Target",
    "positive",
    "freeze",
    "Change",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Activation",
    "softplus",
    "exp",
    "sigmoid",
    "tanh",
    "relu",
    "as_expr",
    "BuildContext",
    "Block",
    "Linear",
    "MLP",
    "MonotoneMLP",
    "Polynomial",
    "RBF",
    "Mean",
    "Normalize",
    "Pretrained",
    "product",
    "to_config",
    "from_config",
]
