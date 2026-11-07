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

import logging
import os
import time

import numpy as np

from physXAI.expr import Expr, Linear, from_config, to_config
from physXAI.expr.core import BuildContext
from physXAI.models.ann.ann_design import ANNModel
from physXAI.models.models import register_model
from physXAI.models.modular.shape_constraints import ShapeConstraints, ShapeFitResult, fit_shape_constrained
from physXAI.preprocessing.training_data import TrainingDataGeneric

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402

logger = logging.getLogger("physXAI")

SOLVERS = ("adam", "ols", "qp")
#: Solvers computing the weights of a Linear block in closed form (no gradient descent)
CLOSED_FORM = ("ols", "qp")


@register_model
class ModularANN(ANNModel):
    """
    Trainable keras model defined by an expression graph.

    Args:
        architecture: The expression computing the output, e.g. `px.Linear([...])`.
        rescale_output: Rescale the output with mean and standard deviation of the training targets.
        solver: 'adam' (gradient descent), 'ols' (closed form least squares) or 'qp' (constrained least squares:
                sign bounds of `Linear(monotonicities=...)` and `shape`). 'ols' and 'qp' need a `Linear` block on data
                only inputs without normalization as architecture.
        shape: Optional `ShapeConstraints`: monotonicity and convexity of the prediction with respect to raw data
               columns, guaranteed on a domain (requires solver='qp', see `physXAI.models.modular.shape_constraints`).
               The result is stored in `td.physics['shape_constraints']`.
        batch_size: Batch size of the training.
        epochs: Maximum number of epochs.
        learning_rate: Learning rate of the Adam optimizer.
        early_stopping_epochs: Patience of the early stopping on the validation loss (None: no early stopping).
        random_seed: Seed of the weights and the batch order, set when the model is generated. Defaults to the global
                     seed (`physXAI.set_seed`), else a new seed is drawn (stored in the config).
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
        random_seed: int | None = None,
        shape: ShapeConstraints | dict | None = None,
        **kwargs,
    ):
        super().__init__(batch_size, epochs, learning_rate, early_stopping_epochs, random_seed, **kwargs)
        if isinstance(architecture, dict):
            architecture = from_config(architecture)
        if isinstance(shape, dict):
            shape = ShapeConstraints.from_config(shape)
        if solver not in SOLVERS:
            raise ValueError(f"Unknown solver '{solver}', use one of {SOLVERS}.")
        self.architecture: Expr = architecture
        self.rescale_output = rescale_output
        self.solver = solver
        self.shape: ShapeConstraints | None = shape
        self._ctx: BuildContext | None = None
        self._rescaling: tuple[float, float] | None = None
        if shape is not None and solver != "qp":
            raise ValueError("Shape constraints require solver='qp'.")
        if solver in CLOSED_FORM:
            self._check_closed_form()
            if self.physics is not None:
                raise ValueError(f"solver='{solver}' cannot be combined with a physics residual, use solver='adam'.")
            if shape is not None and self.recursive_training is not None:
                raise ValueError("Shape constraints cannot be combined with recursive_training: the training on the "
                                 "multi step error (gradient descent) would not keep them.")

    @property
    def required_inputs(self) -> list[str]:
        """Data columns the model needs (free variables of the architecture)."""
        return self.architecture.free_vars()

    def _check_closed_form(self):
        arch, solver = self.architecture, self.solver
        if not isinstance(arch, Linear):
            raise ValueError(f"solver='{solver}' requires a Linear block as architecture.")
        if arch.monotonicities and solver == "ols":
            raise ValueError("solver='ols' cannot enforce monotonicities, use solver='qp'.")
        if arch.normalize or arch.nominal_range is not None:
            raise ValueError(f"solver='{solver}' requires a Linear block without normalization and nominal range.")
        trainable = [c.name for c in arch.inputs if c.trainable]
        if trainable:
            raise ValueError(f"solver='{solver}' requires data only inputs, but {trainable} are trainable.")

    def generate_model(self, **kwargs):
        td: TrainingDataGeneric = kwargs["td"]
        inputs = list(td.inputs)
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
        y = td.y_train_single[:, 0].astype(np.float64)
        if self._rescaling is not None:
            y = (y - self._rescaling[0]) / self._rescaling[1]
        if self.solver == "ols":
            features = np.column_stack([self._ctx.values(c) for c in arch.inputs])
            if arch.bias:
                features = np.column_stack([features, np.ones(len(features))])
            coef, *_ = np.linalg.lstsq(features, y, rcond=None)
            kernel, bias = coef[: len(arch.inputs)], coef[-1:]
        else:
            result = fit_shape_constrained(list(arch.inputs), self._ctx.train_values, y, bias=arch.bias,
                                           weight_signs=arch._per_input(arch.monotonicities), shape=self.shape)
            kernel, bias = result.weights, np.array([result.bias])
            self._store_shape_result(result, td)
        dense = self._ctx.layers[id(arch)]
        weights = [kernel.reshape(-1, 1)] + ([bias] if arch.bias else [])
        dense.set_weights([w.astype(np.float32) for w in weights])
        td.add_training_time(time.perf_counter() - start)
        if self.recursive_training is not None:
            # the least squares solution initializes the training on the multi step error
            self.fit_multi_step(model, td)
        else:
            dense.trainable = False

    def _store_shape_result(self, result: ShapeFitResult, td: TrainingDataGeneric):
        """Writes the diagnosis of the constrained fit into `td.physics['shape_constraints']` and the log."""
        td.physics = {**(td.physics or {}), "shape_constraints": result.get_config()}
        logger.info("Constrained least squares: train RMSE %.4g (unconstrained %.4g), %d iteration(s), %.2f s",
                    result.rmse_train, result.rmse_train_unconstrained, result.iterations, result.seconds)
        for c in result.constraints:
            logger.info("  %s %s (%+d): %s, max violation %.2g, %d constraint points", c["type"], c["column"],
                        c["sign"], c["certificate"], c["max_violation"], c["constraint_points"])

    def online_pipeline(self, td: TrainingDataGeneric, load_path: str, *args, **kwargs):
        if self.solver in CLOSED_FORM:
            raise NotImplementedError(
                f"solver='{self.solver}' solves the least squares problem in closed form, there is no further training. "
                "Train a new model on the combined data instead."
            )
        return super().online_pipeline(td, load_path, *args, **kwargs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "architecture": to_config(self.architecture),
                "rescale_output": self.rescale_output,
                "solver": self.solver,
                "shape": self.shape.get_config() if self.shape is not None else None,
            }
        )
        return config

