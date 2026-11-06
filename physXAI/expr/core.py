"""
Core of the physXAI expression language.

An expression is an immutable node of a directed acyclic graph. It can be evaluated by three
backends:

- `evaluate(values)`: NumPy / pandas, e.g. to create a data column in the preprocessing.
- `to_keras(ctx)`:    Keras layers, i.e. the expression becomes part of the model. Only layers
                       the CasADi export of AgentLib-MPC supports are used.
- `to_casadi(syms)`:  CasADi, e.g. for constrained regressions.

Example:
    >>> from physXAI import expr as px
    >>> v = px.Vars()
    >>> heat_flow = v.mDot * (v.T_in - v.T)            # raw physical arithmetic
    >>> model_output = px.Linear([v.load, heat_flow])  # trainable block
    >>> model_output.free_vars()
    ['load', 'mDot', 'T_in', 'T']

Variables are data columns. `v.T.lag(1)` refers to the column 'T_lag1', which the preprocessing
derives automatically. Arithmetic works on raw (not normalized) values; trainable blocks
normalize their inputs themselves.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

EXPRESSION_REGISTRY: dict[str, type[Expr]] = dict()


def register_expression(cls):
    """Registers an expression class for the deserialization."""
    EXPRESSION_REGISTRY[cls.__name__] = cls
    return cls


def as_expr(value: Expr | float | int | str) -> Expr:
    """Converts numbers to constants and strings to variables."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return Var(value)
    if isinstance(value, numbers.Real):
        return Const(float(value))
    raise TypeError(f"Cannot use {type(value).__name__} in an expression.")


class Expr:
    """
    Base class of all expressions.

    Subclasses define `children` (the input expressions), `params()` (the json serializable
    settings) and the three backends.
    """

    #: Name used in automatically generated names, e.g. '+' for an addition
    symbol: ClassVar[str] = ""

    def __init__(self, *children: Expr, name: str | None = None):
        self.children: tuple[Expr, ...] = tuple(children)
        self._name: str | None = name

    # --- naming ---------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Explicit name or a name derived from the structure of the expression."""
        return self._name if self._name is not None else self.default_name()

    def default_name(self) -> str:
        return f"{self.__class__.__name__}({', '.join(c.name for c in self.children)})"

    def named(self, name: str) -> Expr:
        """Sets the name of the expression (e.g. used for monotonicity constraints) and returns it."""
        self._name = name
        return self

    def __repr__(self) -> str:
        return self.name

    # --- graph ----------------------------------------------------------------------------------
    def nodes(self) -> list[Expr]:
        """All nodes of the graph in topological order (children before parents), without duplicates."""
        order, seen = list(), set()

        def visit(node: Expr):
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node.children:
                visit(child)
            order.append(node)

        visit(self)
        return order

    def free_vars(self) -> list[str]:
        """Names of all data columns the expression depends on, in order of appearance."""
        return list(dict.fromkeys(node.column for node in self.nodes() if isinstance(node, Var)))

    @property
    def trainable(self) -> bool:
        """Whether the expression holds trainable parts (parameters or blocks)."""
        return any(node.has_weights for node in self.nodes())

    #: Whether the node itself holds (trainable) weights
    has_weights: ClassVar[bool] = False

    def lag(self, steps: int) -> Expr:
        """The expression with all variables `steps` time steps in the past."""
        return self.substitute({v: Var(v).lag(steps) for v in self.free_vars()})

    def substitute(self, mapping: Mapping[str, Expr]) -> Expr:
        """Replaces variables by expressions (new graph, shared sub-graphs stay shared)."""
        memo: dict[int, Expr] = dict()

        def rebuild(node: Expr) -> Expr:
            if id(node) in memo:
                return memo[id(node)]
            if isinstance(node, Var) and node.column in mapping:
                new = mapping[node.column]
            elif not node.children:
                new = node
            else:
                new = node.with_children([rebuild(c) for c in node.children])
            memo[id(node)] = new
            return new

        return rebuild(self)

    def with_children(self, children: list[Expr]) -> Expr:
        """Copy of the node with other children."""
        return self.__class__.from_params(self.params(), children, name=self._name)

    # --- serialization --------------------------------------------------------------------------
    def params(self) -> dict[str, Any]:
        """Json serializable settings of the node (without children and name)."""
        return dict()

    @classmethod
    def from_params(cls, params: dict[str, Any], children: list[Expr], name: str | None = None) -> Expr:
        return cls(*children, name=name, **params)

    # --- backends -------------------------------------------------------------------------------
    def evaluate(self, values: Mapping[str, Any]) -> np.ndarray:
        """
        Evaluates the expression with NumPy.

        Args:
            values: Values of the variables, e.g. a pandas DataFrame or a dict of arrays.
        """
        raise NotImplementedError(f"{self.__class__.__name__} cannot be evaluated outside of a model.")

    def to_keras(self, ctx: BuildContext):
        """Builds the keras layers of this node. Use `ctx.tensor(expr)`, which caches shared nodes."""
        raise NotImplementedError(f"{self.__class__.__name__} cannot be part of a keras model.")

    def to_casadi(self, syms: Mapping[str, Any]):
        """Builds the CasADi expression of this node from the symbols of the variables."""
        raise NotImplementedError(f"{self.__class__.__name__} has no CasADi implementation.")

    # --- operators ------------------------------------------------------------------------------
    def __add__(self, other):
        return Add(self, as_expr(other))

    def __radd__(self, other):
        return Add(as_expr(other), self)

    def __sub__(self, other):
        return Sub(self, as_expr(other))

    def __rsub__(self, other):
        return Sub(as_expr(other), self)

    def __mul__(self, other):
        return Mul(self, as_expr(other))

    def __rmul__(self, other):
        return Mul(as_expr(other), self)

    def __truediv__(self, other):
        return Div(self, as_expr(other))

    def __rtruediv__(self, other):
        return Div(as_expr(other), self)

    def __pow__(self, other):
        return Pow(self, as_expr(other))

    def __rpow__(self, other):
        return Pow(as_expr(other), self)

    def __neg__(self):
        return Mul(Const(-1.0), self)


class BuildContext:
    """
    State of building one keras model out of expressions.

    Args:
        input_tensor: Keras input holding all input columns on its last axis.
        input_names: Names of these columns.
        train_values: Training values of the columns, used to adapt normalizations. Optional.
    """

    def __init__(self, input_tensor, input_names: list[str], train_values: Mapping[str, np.ndarray] | None = None):
        self.input = input_tensor
        self.input_names = list(input_names)
        self.train_values = train_values
        self._tensors: dict[int, Any] = dict()
        #: Keras layers holding the weights of blocks, by id of the block
        self.layers: dict[int, Any] = dict()

    def index(self, column: str) -> int:
        try:
            return self.input_names.index(column)
        except ValueError:
            raise KeyError(
                f"The model needs the input '{column}', which is not part of the inputs {self.input_names}."
            ) from None

    def tensor(self, expr: Expr):
        """Keras tensor of an expression, built only once per node."""
        key = id(expr)
        if key not in self._tensors:
            self._tensors[key] = expr.to_keras(self)
        return self._tensors[key]

    def values(self, expr: Expr) -> np.ndarray | None:
        """Training values of a data only expression, None if not available."""
        if self.train_values is None or expr.trainable:
            return None
        return np.asarray(expr.evaluate(self.train_values), dtype=np.float64).reshape(-1)


def _keras():
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras

    return keras


@register_expression
class Var(Expr):
    """A data column, e.g. `Var('T')`. `Var('T').lag(1)` is the column 'T_lag1'."""

    _lag_pattern = re.compile(r"^(.*)_lag(\d+)$")

    def __init__(self, column: str, name: str | None = None):
        super().__init__(name=name)
        self.column = column

    def default_name(self) -> str:
        return self.column

    def lag(self, steps: int) -> Var:
        if steps == 0:
            return self
        match = self._lag_pattern.match(self.column)
        base, lag = (match.group(1), int(match.group(2))) if match else (self.column, 0)
        return Var(f"{base}_lag{lag + steps}")

    def params(self):
        return {"column": self.column}

    def evaluate(self, values):
        return np.asarray(values[self.column], dtype=np.float64)

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import InputSliceLayer

        return InputSliceLayer([ctx.index(self.column)])(ctx.input)

    def to_casadi(self, syms):
        return syms[self.column]


class Vars:
    """
    Namespace creating variables on access: `v = Vars(); v.T`, or `v['Func(logistic)']` for
    column names that are no python identifiers.
    """

    def __getattr__(self, column: str) -> Var:
        if column.startswith("__"):
            raise AttributeError(column)
        return Var(column)

    def __getitem__(self, column: str) -> Var:
        return Var(column)


@register_expression
class Const(Expr):
    """A constant number."""

    def __init__(self, value: float, name: str | None = None):
        super().__init__(name=name)
        self.value = float(value)

    def default_name(self) -> str:
        return f"{self.value:g}"

    def params(self):
        return {"value": self.value}

    def evaluate(self, values):
        return np.float64(self.value)

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import ConstantLayer

        return ConstantLayer(value=self.value)(ctx.input)

    def to_casadi(self, syms):
        return self.value


@register_expression
class Param(Expr):
    """
    A trainable scalar parameter, e.g. a physical coefficient.

    Args:
        init: Initial value. Random (glorot uniform) if None.
        trainable: Whether the parameter is trained.
        name: Name of the parameter (also the name of the keras weight).
    """

    has_weights = True

    def __init__(self, init: float | None = None, trainable: bool = True, name: str | None = None):
        super().__init__(name=name)
        self.init = init
        self.is_trainable = trainable

    def default_name(self) -> str:
        return "param" if self.init is None else f"param({self.init:g})"

    def params(self):
        return {"init": self.init, "trainable": self.is_trainable}

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import ConstantLayer

        layer = ConstantLayer(value=self.init, trainable=self.is_trainable, weight_name=self._name)
        ctx.layers[id(self)] = layer
        return layer(ctx.input)


class Binary(Expr):
    """An operation of two expressions."""

    def __init__(self, left: Expr, right: Expr, name: str | None = None):
        super().__init__(as_expr(left), as_expr(right), name=name)

    def default_name(self) -> str:
        return f"({self.children[0].name}{self.symbol}{self.children[1].name})"

    def evaluate(self, values):
        return self.numpy(self.children[0].evaluate(values), self.children[1].evaluate(values))

    def to_casadi(self, syms):
        return self.numpy(self.children[0].to_casadi(syms), self.children[1].to_casadi(syms))

    @staticmethod
    def numpy(a, b):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


@register_expression
class Add(Binary):
    symbol = "+"

    @staticmethod
    def numpy(a, b):
        return a + b

    def to_keras(self, ctx):
        return _keras().layers.Add()([ctx.tensor(self.children[0]), ctx.tensor(self.children[1])])


@register_expression
class Sub(Binary):
    symbol = "-"

    @staticmethod
    def numpy(a, b):
        return a - b

    def to_keras(self, ctx):
        return _keras().layers.Subtract()([ctx.tensor(self.children[0]), ctx.tensor(self.children[1])])


@register_expression
class Mul(Binary):
    symbol = "*"

    @staticmethod
    def numpy(a, b):
        return a * b

    def to_keras(self, ctx):
        return _keras().layers.Multiply()([ctx.tensor(self.children[0]), ctx.tensor(self.children[1])])


@register_expression
class Div(Binary):
    symbol = "/"

    @staticmethod
    def numpy(a, b):
        return a / b

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import DivideLayer

        return DivideLayer()([ctx.tensor(self.children[0]), ctx.tensor(self.children[1])])


@register_expression
class Pow(Binary):
    symbol = "**"

    @staticmethod
    def numpy(a, b):
        return a**b

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import PowerLayer

        return PowerLayer()([ctx.tensor(self.children[0]), ctx.tensor(self.children[1])])


# NumPy and CasADi implementations of the activations. The keys are keras activations, which
# the CasADi export of AgentLib-MPC supports as well, so training and MPC use identical functions.
def _casadi():
    import casadi as ca

    return ca


ACTIVATIONS = {
    "softplus": (lambda x: np.logaddexp(0, x), lambda x: _casadi().log(1 + _casadi().exp(x))),
    "exponential": (np.exp, lambda x: _casadi().exp(x)),
    "sigmoid": (lambda x: 1 / (1 + np.exp(-x)), lambda x: 1 / (1 + _casadi().exp(-x))),
    "tanh": (np.tanh, lambda x: _casadi().tanh(x)),
    "relu": (lambda x: np.maximum(0, x), lambda x: _casadi().fmax(0, x)),
    "linear": (lambda x: x, lambda x: x),
}


@register_expression
class Activation(Expr):
    """
    An elementwise activation function (softplus, exponential, sigmoid, tanh, relu, linear).
    """

    def __init__(self, x: Expr, activation: str = "softplus", name: str | None = None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activation '{activation}' is not supported, use one of {list(ACTIVATIONS)}.")
        super().__init__(as_expr(x), name=name)
        self.activation = activation

    def default_name(self) -> str:
        return f"{self.activation}({self.children[0].name})"

    def params(self):
        return {"activation": self.activation}

    def evaluate(self, values):
        return ACTIVATIONS[self.activation][0](self.children[0].evaluate(values))

    def to_casadi(self, syms):
        return ACTIVATIONS[self.activation][1](self.children[0].to_casadi(syms))

    def to_keras(self, ctx):
        # A frozen Dense layer with identity kernel applies the activation: activation(x @ I).
        # keras.layers.Activation would be simpler, but Dense is supported by the CasADi export.
        keras = _keras()
        x = ctx.tensor(self.children[0])
        return keras.layers.Dense(
            units=x.shape[-1],
            activation=self.activation,
            use_bias=False,
            kernel_initializer=keras.initializers.Identity(),
            trainable=False,
        )(x)


def softplus(x, name: str | None = None) -> Activation:
    """log(1 + exp(x))"""
    return Activation(x, "softplus", name)


def exp(x, name: str | None = None) -> Activation:
    """exp(x)"""
    return Activation(x, "exponential", name)


def sigmoid(x, name: str | None = None) -> Activation:
    """1 / (1 + exp(-x))"""
    return Activation(x, "sigmoid", name)


def tanh(x, name: str | None = None) -> Activation:
    """tanh(x)"""
    return Activation(x, "tanh", name)


def relu(x, name: str | None = None) -> Activation:
    """max(0, x)"""
    return Activation(x, "relu", name)


def Change(feature: str) -> str:  # noqa: N802 - reads like the naming convention it creates
    """
    Name of the output predicting the change of `feature` to the next step: 'Change(feature)'.
    The preprocessing derives the column automatically.
    """
    return f"Change({feature})"
