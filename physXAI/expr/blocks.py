"""
Trainable building blocks of modular models.

A block maps a list of input expressions to one output. Blocks normalize their inputs
themselves (with statistics of the training data, if the inputs are data only), so arithmetic
outside of blocks can work on raw physical values.
"""

from __future__ import annotations

import functools
import logging
import operator
from itertools import combinations
from pathlib import Path

import numpy as np

from physXAI.expr.core import BuildContext, Expr, _keras, as_expr, register_expression
from physXAI.utils.seed import resolve_seed

logger = logging.getLogger("physXAI")


def _nominal(nominal_range):
    """Mean and sigma of a nominal output range (4 sigma cover the range)."""
    if nominal_range is None:
        return None
    if len(nominal_range) != 2:
        raise ValueError(f"nominal_range must be (min, max), but is {nominal_range}.")
    low, high = nominal_range
    if not high > low:
        raise ValueError(f"nominal_range must be (min, max) with max > min, but is {nominal_range}.")
    return (high + low) / 2.0, (high - low) / 4.0


class Block(Expr):
    """
    Base class of trainable blocks.

    Args:
        inputs: Input expressions (or column names).
        name: Name of the block.
        normalize: Whether the inputs are normalized with the statistics of the training data.
        nominal_range: Optional (min, max) of the output. The output of the block is scaled to it.
    """

    has_weights = True
    default_normalize = True

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        normalize: bool | None = None,
        nominal_range: tuple[float, float] | None = None,
    ):
        if isinstance(inputs, (Expr, str)):
            inputs = [inputs]
        if len(inputs) == 0:
            raise ValueError(f"{self.__class__.__name__} needs at least one input.")
        super().__init__(*[as_expr(i) for i in inputs], name=name)
        self.normalize = self.default_normalize if normalize is None else normalize
        self.nominal_range = tuple(nominal_range) if nominal_range is not None else None
        _nominal(self.nominal_range)

    @property
    def inputs(self) -> tuple[Expr, ...]:
        return self.children

    def params(self):
        nominal = list(self.nominal_range) if self.nominal_range is not None else None
        return {"normalize": self.normalize, "nominal_range": nominal}

    @classmethod
    def from_params(cls, params, children, name=None):
        return cls(list(children), name=name, **params)

    def _per_input(self, values: dict | None, default: int = 0) -> list[int]:
        """Maps a dict {input name: value} to one value per input."""
        values = values or dict()
        names = [c.name for c in self.inputs]
        unknown = set(values) - set(names)
        if unknown:
            raise ValueError(f"{self.__class__.__name__}: unknown inputs {sorted(unknown)}, inputs are {names}.")
        return [values.get(n, default) for n in names]

    def input_tensor(self, ctx: BuildContext):
        """Concatenated (and normalized) inputs."""
        keras = _keras()
        tensors = [ctx.tensor(c) for c in self.inputs]
        x = tensors[0] if len(tensors) == 1 else keras.layers.Concatenate()(tensors)
        if not self.normalize:
            return x
        means, variances = list(), list()
        for child in self.inputs:
            values = ctx.values(child)
            if values is None or len(values) == 0:
                means.append(0.0)
                variances.append(1.0)
            else:
                variance = float(np.nanvar(values))
                means.append(float(np.nanmean(values)))
                variances.append(variance if variance > 1e-12 else 1.0)
        return keras.layers.Normalization(mean=np.array(means), variance=np.array(variances))(x)

    def rescale(self, x):
        nominal = _nominal(self.nominal_range)
        if nominal is None:
            return x
        return _keras().layers.Rescaling(scale=nominal[1], offset=nominal[0])(x)

    def evaluate(self, values):
        raise NotImplementedError(f"{self.__class__.__name__} is trainable and can only be evaluated in a model.")


@register_expression
class Linear(Block):
    """
    Linear combination `w @ inputs + b`.

    Args:
        inputs: Input expressions.
        monotonicities: Optional {input name: 1 (increasing) | -1 (decreasing)}.
        bias: Whether to use a bias.
        normalize: Normalize the inputs (default False, so the weights keep their physical meaning
                   and the block can be solved by ordinary least squares).
    """

    default_normalize = False

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        monotonicities: dict[str, int] | None = None,
        bias: bool = True,
        normalize: bool | None = None,
        nominal_range=None,
    ):
        super().__init__(inputs, name, normalize, nominal_range)
        self.monotonicities = dict(monotonicities or {})
        self._per_input(self.monotonicities)
        self.bias = bias
        if not bias and self.normalize:
            logger.warning(
                "Linear '%s' has no bias, but normalizes its inputs: the normalization subtracts the mean of the "
                "training data, which acts as a bias. Use normalize=False for a model without offset.",
                self.name,
            )

    def params(self):
        return {**super().params(), "monotonicities": self.monotonicities, "bias": self.bias}

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import NonNegPartial

        keras = _keras()
        constraint = NonNegPartial(self._per_input(self.monotonicities)) if self.monotonicities else None
        dense = keras.layers.Dense(1, activation="linear", use_bias=self.bias, kernel_constraint=constraint)
        ctx.layers[id(self)] = dense
        return self.rescale(dense(self.input_tensor(ctx)))

    def to_casadi(self, syms):
        raise NotImplementedError("Use the trained keras model to evaluate a Linear block.")


@register_expression
class MLP(Block):
    """
    Feed forward network.

    Args:
        inputs: Input expressions.
        n_layers: Number of hidden layers.
        n_neurons: Neurons per hidden layer (int or one per layer).
        activation: Activation of the hidden layers (one or one per layer).
    """

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        n_layers: int = 1,
        n_neurons: int | list[int] = 32,
        activation: str | list[str] = "softplus",
        normalize: bool | None = None,
        nominal_range=None,
    ):
        super().__init__(inputs, name, normalize, nominal_range)
        self.n_layers = n_layers
        self.n_neurons = _per_layer(n_neurons, n_layers, "n_neurons")
        self.activation = _per_layer(activation, n_layers, "activation")

    def params(self):
        return {
            **super().params(),
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
        }

    def to_keras(self, ctx):
        keras = _keras()
        x = self.input_tensor(ctx)
        for units, activation in zip(self.n_neurons, self.activation, strict=True):
            x = keras.layers.Dense(units, activation=activation)(x)
        return self.rescale(keras.layers.Dense(1, activation="linear")(x))


@register_expression
class MonotoneMLP(Block):
    """
    Constrained monotonic neural network (CMNN): monotone in the given inputs.

    Args:
        inputs: Input expressions.
        monotonicities: {input name: 1 (increasing) | -1 (decreasing)}.
        n_layers: Number of hidden layers, see `MLP`.
        n_neurons: Neurons per hidden layer, see `MLP`.
        activation: Activation function(s), see `MLP`.
        activation_split: Shares of convex, concave and saturated neurons. Defaults to [1, 1, 1].
    """

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        monotonicities: dict[str, int] | None = None,
        n_layers: int = 1,
        n_neurons: int | list[int] = 32,
        activation: str | list[str] = "softplus",
        activation_split: list[float] | None = None,
        normalize: bool | None = None,
        nominal_range=None,
    ):
        super().__init__(inputs, name, normalize, nominal_range)
        self.monotonicities = dict(monotonicities or {})
        self._per_input(self.monotonicities)
        if n_layers < 1:
            # without a hidden layer only the non negative weights of the output layer remain: the prediction would
            # be increasing in every input
            raise ValueError("MonotoneMLP needs at least one hidden layer (n_layers >= 1).")
        self.n_layers = n_layers
        self.n_neurons = _per_layer(n_neurons, n_layers, "n_neurons")
        self.activation = _per_layer(activation, n_layers, "activation")
        self.activation_split = list(activation_split) if activation_split is not None else [1.0, 1.0, 1.0]
        if len(self.activation_split) != 3 or min(self.activation_split) < 0 or sum(self.activation_split) == 0:
            raise ValueError("activation_split needs three non negative shares, not all zero.")

    def params(self):
        return {
            **super().params(),
            "monotonicities": self.monotonicities,
            "n_layers": self.n_layers,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "activation_split": self.activation_split,
        }

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import ConcaveActivation, NonNegPartial, SaturatedActivation

        keras = _keras()
        x = self.input_tensor(ctx)
        constraint = NonNegPartial(self._per_input(self.monotonicities))
        total = sum(self.activation_split)
        for units, activation in zip(self.n_neurons, self.activation, strict=True):
            activations = [activation, ConcaveActivation(activation), SaturatedActivation(activation)]
            parts = list()
            for share, act in zip(self.activation_split, activations, strict=True):
                n = int(round(units * share / total))
                if share > 0 and n == 0:
                    n = 1  # every active part keeps at least one neuron
                if n > 0:
                    parts.append(keras.layers.Dense(n, activation=act, kernel_constraint=constraint)(x))
            x = parts[0] if len(parts) == 1 else keras.layers.Concatenate()(parts)
            # after the first layer, non negative weights preserve the monotonicity
            constraint = keras.constraints.NonNeg()
        out = keras.layers.Dense(1, activation="linear", kernel_constraint=keras.constraints.NonNeg())(x)
        return self.rescale(out)


@register_expression
class Polynomial(Block):
    """
    Polynomial regression: powers of every input up to `degree` and products of up to
    `interaction_degree` inputs, combined linearly.
    """

    default_normalize = False

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        degree: int = 2,
        interaction_degree: int = 1,
        normalize: bool | None = None,
        nominal_range=None,
    ):
        super().__init__(inputs, name, normalize, nominal_range)
        if degree < 1 or interaction_degree < 1:
            raise ValueError("degree and interaction_degree must be at least 1.")
        self.degree = degree
        self.interaction_degree = interaction_degree

    def params(self):
        return {**super().params(), "degree": self.degree, "interaction_degree": self.interaction_degree}

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import InputSliceLayer

        keras = _keras()
        x = self.input_tensor(ctx)
        columns = [InputSliceLayer([i])(x) for i in range(len(self.inputs))] if len(self.inputs) > 1 else [x]
        # powers as repeated products: unlike a power layer they are defined for negative bases
        terms = list(columns)
        for column in columns:
            power = column
            for _ in range(2, self.degree + 1):
                power = keras.layers.Multiply()([power, column])
                terms.append(power)
        for k in range(2, self.interaction_degree + 1):
            for combo in combinations(columns, k):
                terms.append(functools.reduce(lambda a, b: keras.layers.Multiply()([a, b]), combo))
        features = terms[0] if len(terms) == 1 else keras.layers.Concatenate()(terms)
        dense = keras.layers.Dense(1, activation="linear")
        ctx.layers[id(self)] = dense
        return self.rescale(dense(features))


@register_expression
class RBF(Block):
    """
    Radial basis function network with `units` gaussian neurons. Centers are initialized by
    k-means on the (normalized) training data, if available. `random_state` is the seed of the k-means,
    defaults to the global seed (`physXAI.set_seed`), else a new seed is drawn (stored in the config).
    """

    def __init__(
        self,
        inputs: list,
        name: str | None = None,
        units: int = 32,
        normalize: bool | None = None,
        nominal_range=None,
        random_state: int | None = None,
    ):
        super().__init__(inputs, name, normalize, nominal_range)
        self.units = units
        self.random_state = resolve_seed(random_state)

    def params(self):
        return {**super().params(), "units": self.units, "random_state": self.random_state}

    def to_keras(self, ctx):
        from physXAI.models.ann.keras_models.keras_models import RBFLayer

        keras = _keras()
        x = self.input_tensor(ctx)
        centers, gamma = None, 1.0
        values = [ctx.values(c) for c in self.inputs]
        if all(v is not None for v in values):
            from sklearn.cluster import KMeans

            from physXAI.models.ann.model_construction.rbf_models import gamma_init

            data = np.column_stack(values)
            if self.normalize:
                data = (data - data.mean(axis=0)) / np.where(data.std(axis=0) > 0, data.std(axis=0), 1.0)
            units = min(self.units, len(data))
            centers = (
                KMeans(n_clusters=units, random_state=self.random_state % 2**32, n_init="auto")
                .fit(data)
                .cluster_centers_
            )
            gamma = gamma_init(centers) if units > 1 else 1.0
        layer = RBFLayer(self.units if centers is None else len(centers), gamma=gamma, initial_centers=centers)
        return self.rescale(keras.layers.Dense(1, activation="linear")(layer(x)))


@register_expression
class Mean(Block):
    """Average of the inputs."""

    has_weights = False
    default_normalize = False

    def params(self):
        return dict()

    def evaluate(self, values):
        return np.mean([c.evaluate(values) for c in self.inputs], axis=0)

    def to_casadi(self, syms):
        return sum(c.to_casadi(syms) for c in self.inputs) / len(self.inputs)

    def to_keras(self, ctx):
        tensors = [ctx.tensor(c) for c in self.inputs]
        return tensors[0] if len(tensors) == 1 else _keras().layers.Average()(tensors)


@register_expression
class Normalize(Block):
    """
    Normalizes one expression: with the statistics of the training data if it is data only,
    else with batch normalization.
    """

    def __init__(self, x, name: str | None = None):
        super().__init__([x], name)

    def params(self):
        return dict()

    @classmethod
    def from_params(cls, params, children, name=None):
        return cls(children[0], name=name)

    def to_keras(self, ctx):
        keras = _keras()
        child = self.inputs[0]
        values = ctx.values(child)
        if values is None:
            return keras.layers.BatchNormalization()(ctx.tensor(child))
        return self.input_tensor(ctx)


@register_expression
class Pretrained(Block):
    """
    An existing keras model applied to the inputs.

    Args:
        model: Path to a saved keras model (or the model itself, which then cannot be serialized).
        inputs: Input expressions, in the order the model expects them.
        trainable: Whether the model is trained further.
    """

    def __init__(self, model, inputs: list, name: str | None = None, trainable: bool = False):
        self.path = str(model) if isinstance(model, (str, Path)) else None
        if self.path is not None:
            model = _keras().saving.load_model(self.path)
        model.trainable = trainable
        self.model = model
        self.is_trainable = trainable
        super().__init__(inputs, name=name or f"{model.name}_pretrained", normalize=False)

    def params(self):
        if self.path is None:
            raise ValueError("A Pretrained block built from a model object cannot be serialized, pass a path instead.")
        return {"model": self.path, "trainable": self.is_trainable}

    @classmethod
    def from_params(cls, params, children, name=None):
        return cls(params["model"], list(children), name=name, trainable=params["trainable"])

    def to_keras(self, ctx):
        return self.model(self.input_tensor(ctx))


def _per_layer(value, n_layers: int, label: str) -> list:
    values = list(value) if isinstance(value, (list, tuple)) else [value] * n_layers
    if len(values) != n_layers:
        raise ValueError(f"{label} needs one value per layer ({n_layers}), got {values}.")
    return values


def product(expressions: list[Expr]) -> Expr:
    """Product of several expressions."""
    return functools.reduce(operator.mul, expressions)
