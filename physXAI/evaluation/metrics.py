"""
Metrics of the predictions of a model, per output and split.

`Metrics.by_output` holds MSE, RMSE, MAE, R2 and bias (mean of prediction - truth) for every output and
split. The flat dicts `train_kpis`, `val_kpis` and `test_kpis` ('RMSE Train', ...) are kept for
compatibility; with several outputs they hold the mean over the outputs.
"""

import logging
import math

import numpy as np
from sklearn.metrics import r2_score

from physXAI.preprocessing.training_data import TrainingDataGeneric, TrainingDataMultiStep

logger = logging.getLogger("physXAI")

SPLIT_LABELS = {"train": "Train", "val": "Val", "test": "Test"}


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
    outputs = getattr(td, "outputs", None) or getattr(td, "output", None)
    return list(outputs) if outputs else ["output"]


class Metrics:
    """
    Regression metrics of the training, validation and test predictions, per output.

    Args:
        td: Training data holding the true values and the predictions.
        names: Names of the outputs. Defaults to `output_names(td)`.

    Attributes:
        by_output: {output: {split: {'mse', 'rmse', 'mae', 'r2', 'bias', 'n'}}}.
        train_kpis, val_kpis, test_kpis: {'MSE Train': ..., 'RMSE Train': ..., 'R2 Train': ...} (mean over
                                         the outputs), None for a split without data.
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
                self.by_output[name][split] = regression_kpis(y[:, j], prediction[:, j])
        self.train_kpis = self._flat("train")
        self.val_kpis = self._flat("val")
        self.test_kpis = self._flat("test")

    def _flat(self, split: str) -> dict[str, float] | None:
        values = [kpis[split] for kpis in self.by_output.values() if split in kpis]
        if not values:
            return None
        label = SPLIT_LABELS[split]
        mse = float(np.mean([v["mse"] for v in values]))
        r2 = float(np.mean([v["r2"] for v in values]))
        logger.info("%s RMSE: %.4g, R2: %.4f", label, math.sqrt(mse), r2)
        return {f"MSE {label}": mse, f"RMSE {label}": math.sqrt(mse), f"R2 {label}": r2}

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict[str, float]:
        """MSE, RMSE and R2 of all values (legacy interface)."""
        kpis = regression_kpis(y_true, y_pred)
        return {f"MSE {label}": kpis["mse"], f"RMSE {label}": kpis["rmse"], f"R2 {label}": kpis["r2"]}

    def get_metrics(self, nround: int = 2) -> tuple[list[str], list[float]]:
        """
        Labels and rounded values of the flat metrics.

        Args:
            nround (int, optional): The number of decimal places to round the metric values to.
        """
        kpis = dict()
        for split_kpis in (self.train_kpis, self.val_kpis, self.test_kpis):
            if split_kpis is not None:
                kpis |= split_kpis
        return list(kpis.keys()), [round(v, nround) for v in kpis.values()]

    def get_config(self) -> dict:
        return {
            "train_kpis": self.train_kpis,
            "val_kpis": self.val_kpis,
            "test_kpis": self.test_kpis,
            "by_output": self.by_output,
        }


class MetricsPINN(Metrics):
    """
    A class to calculate and store metrics for Physics-Informed Neural Networks (PINNs).
    It evaluates performance using a list of provided loss functions.
    """

    def __init__(self, td: TrainingDataGeneric, pinn_losses: list):
        """
        Initializes the MetricsPINN object by calculating metrics for train,
        validation (if available), and test sets using specified PINN loss functions.

        Args:
            td (TrainingData): An object containing the true and predicted values.
            pinn_losses (list): A list of loss functions to be used for evaluation.
        """

        self.by_output = None
        self.train_kpis = self.evaluate(
            td.y_train_single, td.y_train_pred_single, label="Train", pinn_losses=pinn_losses
        )
        if td.y_val_single is not None:
            self.val_kpis = self.evaluate(td.y_val_single, td.y_val_pred_single, label="Val", pinn_losses=pinn_losses)
        else:
            self.val_kpis = None
        self.test_kpis = self.evaluate(td.y_test_single, td.y_test_pred_single, label="Test", pinn_losses=pinn_losses)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "", **kwargs) -> dict[str, float]:
        """
        Calculates metrics based on the provided list of PINN loss functions.

        Args:
            y_true (np.ndarray): The true target values.
            y_pred (np.ndarray): The predicted target values.
            label (str, optional): A label for the dataset being evaluated (e.g., 'Train', 'Test').
                                   Defaults to ''.

        Returns:
            dict: A dictionary where keys are loss function names appended with the label,
                  and values are the calculated loss values.
        """

        kpis = dict()
        for loss in kwargs["pinn_losses"]:
            val = float(loss(y_true, y_pred))
            kpis[loss.__name__ + " " + label] = val
            logger.info("%s %s: %.4g", loss.__name__, label, val)

        return kpis


class MetricsMultiStep(Metrics):
    """
    Metrics of multi step predictions over all predicted steps, per output (the predicted trajectories
    of the features if their start values are known). The error per step is in `td.horizon_metrics`.
    """
