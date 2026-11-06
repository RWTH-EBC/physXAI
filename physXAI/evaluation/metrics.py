"""
Metrics of the predictions of a model, per output and split.

`Metrics.by_output` holds MSE, RMSE, MAE, R2 and bias (mean of prediction - truth) for every output and
split. `MetricsPINN` additionally holds the disagreement of the prediction with every physics target.
"""

import logging
import math

import numpy as np
from sklearn.metrics import r2_score

from physXAI.preprocessing.training_data import TrainingDataGeneric, TrainingDataMultiStep

logger = logging.getLogger("physXAI")

SPLITS = ("train", "val", "test")


def regression_kpis(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, MAE, R2 and bias (mean of prediction - truth) of one output."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "bias": float(np.mean(error)),
        "n": int(len(y_true)),
    }


def output_names(td: TrainingDataGeneric) -> list[str]:
    """Names of the predicted quantities: the outputs, for multi step data the predicted features."""
    if isinstance(td, TrainingDataMultiStep):
        return list(td.features if td.absolute else td.outputs)
    return list(td.outputs) if getattr(td, "outputs", None) else ["output"]


class Metrics:
    """
    Regression metrics of the training, validation and test predictions, per output.

    Args:
        td: Training data holding the true values and the predictions.
        names: Names of the outputs. Defaults to `output_names(td)`.

    Attributes:
        by_output: {output: {split: {'mse', 'rmse', 'mae', 'r2', 'bias', 'n'}}}. Splits without data are missing.
    """

    def __init__(self, td: TrainingDataGeneric, names: list[str] | None = None):
        data = {
            "train": (td.y_train_single, td.y_train_pred_single),
            "val": (td.y_val_single, td.y_val_pred_single),
            "test": (td.y_test_single, td.y_test_pred_single),
        }
        first = next(p for _, p in data.values() if p is not None)  # outputs of the model
        n_outputs = 1 if np.ndim(first) == 1 else first.shape[-1]
        names = list(names) if names is not None else output_names(td)
        if len(names) != n_outputs:
            names = names[:n_outputs] if len(names) > n_outputs else [f"output_{j}" for j in range(n_outputs)]
        self.names = names
        self.by_output: dict[str, dict[str, dict]] = {name: dict() for name in names}
        for split, (y, prediction) in data.items():
            if y is None or prediction is None:
                continue
            y = np.asarray(y).reshape(len(y), -1)
            prediction = np.asarray(prediction).reshape(len(prediction), -1)
            for j, name in enumerate(names):
                kpis = regression_kpis(y[:, j], prediction[:, j])
                self.by_output[name][split] = kpis
                logger.info("%s %s: RMSE %.4g, R2 %.4f", split.capitalize(), name, kpis["rmse"], kpis["r2"])

    def kpi(self, split: str, metric: str = "rmse", output: str | None = None) -> float | None:
        """
        One metric of one output (default: the first) on a split, None if the split has no data.

        Example:
            >>> td.metrics.kpi('test', 'rmse')
        """
        output = self.names[0] if output is None else output
        return self.by_output[output].get(split, {}).get(metric)

    def get_config(self) -> dict:
        return {"by_output": self.by_output}


class MetricsPINN(Metrics):
    """
    Metrics of a physics-informed model: the metrics of its output (`by_output`) and, per physics target,
    the disagreement of the prediction with it.

    Args:
        td: Training data holding the true values and the predictions.
        physics: {term: {split: physics target values}}, aligned with the predictions of the split.
        scale: Characteristic scale of the loss terms (variance of the training targets, 1 if not normalized).

    Attributes:
        physics: {term: {split: {'mse', 'rmse', 'normalized'}}}, `normalized` = mse / scale is the term of the loss.
    """

    def __init__(self, td: TrainingDataGeneric, physics: dict[str, dict[str, np.ndarray]], scale: float = 1.0):
        super().__init__(td)
        predictions = {"train": td.y_train_pred_single, "val": td.y_val_pred_single, "test": td.y_test_pred_single}
        self.scale = float(scale)
        self.physics: dict[str, dict[str, dict]] = dict()
        for term, splits in physics.items():
            self.physics[term] = dict()
            for split, target in splits.items():
                prediction = predictions.get(split)
                if prediction is None or target is None:
                    continue
                prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
                error = prediction - np.asarray(target, dtype=np.float64).reshape(-1)
                mse = float(np.mean(error**2))
                self.physics[term][split] = {"mse": mse, "rmse": math.sqrt(mse), "normalized": mse / self.scale}
                logger.info("%s physics '%s': RMSE %.4g", split.capitalize(), term, math.sqrt(mse))

    def get_config(self) -> dict:
        return {**super().get_config(), "physics": self.physics, "scale": self.scale}


class MetricsMultiStep(Metrics):
    """
    Metrics of multi step predictions over all predicted steps, per output (the predicted trajectories
    of the features if their start values are known). The error per step is in `td.horizon_metrics`.
    """
