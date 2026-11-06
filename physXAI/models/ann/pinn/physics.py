"""
Loss and metrics of physics-informed training.

The loss of a model with one output y and physics targets p_1, ..., p_n (e.g. the prediction of a
physical model of the same quantity) is

    L = mse(y_hat, y) / s + sum_i w_i * mse(y_hat, p_i) / s

with the characteristic scale s = var(y_train) (`normalize_terms=True`, else s = 1). The scale makes the
terms dimensionless: the data term is 1 - R^2 on the training data and w_i is the relative importance of
physics target i. The terms are not averaged, so adding a physics term does not change the weight of the
data term. The weights encode the trust in the physics and are chosen by validation (e.g. the error over
the prediction horizon); they are not adapted during the training.

The targets passed to keras are `[y, p_1, ..., p_n]` (columns), the model predicts y only, so the trained
model can be used (and exported to AgentLib-MPC) like any other single output model.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402
from keras import ops  # noqa: E402


@keras.saving.register_keras_serializable(package="physXAI", name="PhysicsLoss")
class PhysicsLoss(keras.losses.Loss):
    """
    Data loss plus weighted physics terms, all divided by the characteristic scale.

    Args:
        weights: Weight of each physics target (columns 1, 2, ... of y_true).
        scale: Characteristic scale (variance of the training targets).
    """

    def __init__(self, weights: list[float], scale: float = 1.0, name: str = "physics_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = [float(w) for w in weights]
        self.scale = float(scale)

    def call(self, y_true, y_pred):
        prediction = y_pred[:, 0]
        loss = ops.square(prediction - y_true[:, 0])
        for i, weight in enumerate(self.weights):
            loss = loss + weight * ops.square(prediction - y_true[:, i + 1])
        return loss / self.scale

    def get_config(self) -> dict:
        return {**super().get_config(), "weights": self.weights, "scale": self.scale}


@keras.saving.register_keras_serializable(package="physXAI", name="TermLoss")
class TermLoss(keras.metrics.Mean):
    """
    One unweighted term of the loss: mean of (y_hat - y_true[:, column])^2 / scale.
    Column 0 is the data term ('data_loss'), columns 1, ... the physics targets.
    """

    def __init__(self, column: int, scale: float = 1.0, name: str = "data_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.column = int(column)
        self.scale = float(scale)

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.square(y_pred[:, 0] - y_true[:, self.column]) / self.scale
        return super().update_state(values, sample_weight=sample_weight)

    def get_config(self) -> dict:
        return {**super().get_config(), "column": self.column, "scale": self.scale}


@keras.saving.register_keras_serializable(package="physXAI", name="OutputRMSE")
class OutputRMSE(keras.metrics.RootMeanSquaredError):
    """RMSE of the prediction with respect to the data (column 0 of the targets), in output units."""

    def __init__(self, name: str = "rmse", **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :1], y_pred[:, :1], sample_weight=sample_weight)


def physics_metrics(terms: list[str], scale: float) -> list[keras.metrics.Metric]:
    """Metrics of all terms (`data_loss`, `<term>_loss`) and the RMSE of the data."""
    metrics = [TermLoss(0, scale, name="data_loss")]
    metrics += [TermLoss(i + 1, scale, name=f"{term}_loss") for i, term in enumerate(terms)]
    return [*metrics, OutputRMSE()]
