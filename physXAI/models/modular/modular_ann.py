"""
Modular artificial neural networks: a trainable model built from a `physXAI.expr` graph.

    >>> from physXAI import expr as px
    >>> v = px.Vars()
    >>> model = ModularANN(px.Linear([v.mDot, v.load, v.mDot * (v.T_in - v.T), v.T]), solver="ols")

Physical arithmetic (like the heat flow above) is part of the keras model. It is therefore
recomputed in every step of a recursive prediction and exported to AgentLib-MPC together with
the model, so the MPC only needs the raw inputs.
"""

from __future__ import annotations

import os
import time

import numpy as np

from physXAI.expr import Expr, Linear, from_config, to_config
from physXAI.expr.core import BuildContext
from physXAI.models.ann.ann_design import ANNModel
from physXAI.models.models import register_model
from physXAI.preprocessing.training_data import TrainingDataGeneric

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402


@register_model
class ModularANN(ANNModel):
    """
    Trainable keras model defined by an expression graph.

    Args:
        architecture: The expression computing the output, e.g. `px.Linear([...])`.
        rescale_output: Rescale the output with mean and standard deviation of the training targets.
        solver: 'adam' (gradient descent) or 'ols' (closed form least squares; the architecture has to
                be a `Linear` block on data only inputs without monotonicities).
        batch_size: Batch size of the training.
        epochs: Maximum number of epochs.
        learning_rate: Learning rate of the Adam optimizer.
        early_stopping_epochs: Patience of the early stopping on the validation loss (None: no early stopping).
        random_seed: Seed of the weights and the batch order, set when the model is generated.
    """

    plot_title = "Modular ANN"

    def __init__(
        self,
        architecture: Expr | dict,
        rescale_output: bool = False,
        solver: str = "adam",
        batch_size: int = 32,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        early_stopping_epochs: int | None = 100,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        if isinstance(architecture, dict):
            architecture = from_config(architecture)
        if solver not in ("adam", "ols"):
            raise ValueError(f"Unknown solver '{solver}', use 'adam' or 'ols'.")
        self.architecture: Expr = architecture
        self.rescale_output = rescale_output
        self.solver = solver
        self._ctx: BuildContext | None = None
        self._rescaling: tuple[float, float] | None = None
        if solver == "ols":
            self._check_ols()

    @property
    def required_inputs(self) -> list[str]:
        """Data columns the model needs (free variables of the architecture)."""
        return self.architecture.free_vars()

    def _check_ols(self):
        arch = self.architecture
        if not isinstance(arch, Linear):
            raise ValueError("solver='ols' requires a Linear block as architecture.")
        if arch.monotonicities:
            raise ValueError("solver='ols' cannot enforce monotonicities, use solver='adam' or ConstrainedRegression.")
        if arch.normalize or arch.nominal_range is not None:
            raise ValueError("solver='ols' requires a Linear block without normalization and nominal range.")
        trainable = [c.name for c in arch.inputs if c.trainable]
        if trainable:
            raise ValueError(f"solver='ols' requires data only inputs, but {trainable} are trainable.")

    def generate_model(self, **kwargs):
        td: TrainingDataGeneric = kwargs["td"]
        inputs = list(td.columns)
        X = td.X_train_single
        missing = [v for v in self.required_inputs if v not in inputs]
        if missing:
            raise ValueError(f"The architecture needs the inputs {missing}, which are not part of the data {inputs}.")
        ctx = BuildContext(keras.Input(shape=(len(inputs),)), inputs, {name: X[:, i] for i, name in enumerate(inputs)})
        x = ctx.tensor(self.architecture)
        self._rescaling = None
        if self.rescale_output:
            y = td.y_train_single
            self._rescaling = (float(np.mean(y)), float(np.std(y, ddof=1)))
            x = keras.layers.Rescaling(scale=self._rescaling[1], offset=self._rescaling[0])(x)
        self._ctx = ctx
        return keras.Model(ctx.input, x)

    def fit_model(self, model, td: TrainingDataGeneric):
        if self.solver == "adam":
            return super().fit_model(model, td)

        start = time.perf_counter()
        arch: Linear = self.architecture
        features = np.column_stack([self._ctx.values(c) for c in arch.inputs])
        y = td.y_train_single[:, 0].astype(np.float64)
        if self._rescaling is not None:
            y = (y - self._rescaling[0]) / self._rescaling[1]
        if arch.bias:
            features = np.column_stack([features, np.ones(len(features))])
        coef, *_ = np.linalg.lstsq(features, y, rcond=None)
        dense = self._ctx.layers[id(arch)]
        weights = [coef[: len(arch.inputs)].reshape(-1, 1)] + ([coef[-1:]] if arch.bias else [])
        dense.set_weights([w.astype(np.float32) for w in weights])
        td.add_training_time(time.perf_counter() - start)
        if self.recursive_training is not None:
            # the least squares solution initializes the training on the multi step error
            self.fit_multi_step(model, td)
        else:
            dense.trainable = False

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "architecture": to_config(self.architecture),
                "rescale_output": self.rescale_output,
                "solver": self.solver,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> ModularANN:
        config = dict(config)
        config.pop("__class_name__", None)
        config.pop("physXAI_version", None)
        return cls(**config)
