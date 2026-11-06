"""
Training of single step models on the multi step (recursive) error.

A single step model trained on the single step error can still drift when it is applied
recursively in an MPC. Training the recursive rollout (`physXAI.evaluation.rollout`) directly on
the error over a horizon counters that. The first step is weighted more, since it is the most
relevant one for the MPC (it is applied to the plant); the following steps keep the model stable
over the horizon.

    >>> m = ClassicalANNModel(recursive_training=RecursiveTraining(horizon='24h', first_step_weight=0.25))

Only the weights of the step model are trained, so the saved model stays a single step model,
which AgentLib-MPC uses as before.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras  # noqa: E402

logger = logging.getLogger("physXAI")

#: Modes of the training of a single step model ('single_step': the training without `RecursiveTraining`)
Mode = Literal["single_step", "multi_step", "single_then_multi"]


class RecursiveTraining:
    """
    Settings of the training on the multi step error.

    Args:
        horizon: Training horizon in steps or as time delta string ('24h'). Defaults to the
                 evaluation horizon of the training data.
        first_step_weight: Share of the first step in the loss (the other steps share the rest
                           equally). Ignored if `step_weights` is given.
        step_weights: Optional weight per step (normalized to a sum of one).
        mode: 'single_then_multi' (default: train on the single step error first, then fine-tune on
              the multi step error) or 'multi_step' (only the multi step error).
        epochs: Epochs of the multi step training. Defaults to the epochs of the model.
        learning_rate: Learning rate of the multi step training. Defaults to the one of the model, for
                       the fine-tuning of 'single_then_multi' to a tenth of it. After a least squares
                       solution (`ModularANN(solver='ols')`) of an unnormalized `Linear` block, the weights are
                       small and need a much smaller learning rate (e.g. 1e-6).
        batch_size: Windows per batch. Defaults to the batch size of the model.
        stride: Distance between two training windows in steps.
        early_stopping_epochs: Patience of the early stopping on the validation loss. Defaults to
                               the one of the model.
    """

    def __init__(
        self,
        horizon: int | str | None = None,
        first_step_weight: float = 0.25,
        step_weights: list[float] | None = None,
        mode: Mode = "single_then_multi",
        epochs: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        stride: int = 1,
        early_stopping_epochs: int | None = None,
    ):
        if mode not in ("multi_step", "single_then_multi"):
            raise ValueError(f"Unknown mode '{mode}', use 'multi_step' or 'single_then_multi'.")
        if not 0 < first_step_weight <= 1:
            raise ValueError("first_step_weight has to be in (0, 1].")
        self.horizon = horizon
        self.first_step_weight = first_step_weight
        self.step_weights = list(step_weights) if step_weights is not None else None
        self.mode: Mode = mode
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.stride = stride
        self.early_stopping_epochs = early_stopping_epochs

    def resolve_horizon(self, td) -> int:
        horizon = self.horizon
        if isinstance(horizon, str):
            if not td.time_step:
                raise ValueError("A horizon given as time delta requires the time step of the data.")
            horizon = int(round(pd.Timedelta(horizon).total_seconds() / td.time_step))
        horizon = horizon or td.eval_horizon
        if not horizon:
            raise ValueError(
                "No horizon for the multi step training: set RecursiveTraining(horizon=...) or "
                "eval_horizon of the preprocessing."
            )
        return int(horizon)

    def weights(self, horizon: int) -> np.ndarray:
        """Normalized weight per step."""
        return step_weights(horizon, self.first_step_weight, self.step_weights)

    def get_config(self) -> dict:
        return {
            "horizon": self.horizon,
            "first_step_weight": self.first_step_weight,
            "step_weights": self.step_weights,
            "mode": self.mode,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "stride": self.stride,
            "early_stopping_epochs": self.early_stopping_epochs,
        }

    @classmethod
    def from_config(cls, config: dict | None) -> RecursiveTraining | None:
        return None if config is None else cls(**config)


def step_weights(
    horizon: int, first_step_weight: float | None = None, weights: list[float] | None = None
) -> np.ndarray:
    """
    Normalized weight per prediction step.

    Args:
        horizon: Number of steps.
        first_step_weight: Share of the first step (the other steps share the rest equally).
                           None weights all steps equally.
        weights: Optional weight per step, overrides `first_step_weight`.
    """
    if weights is not None:
        if len(weights) != horizon:
            raise ValueError(f"step_weights need {horizon} values, got {len(weights)}.")
        result = np.asarray(weights, dtype=np.float64)
    elif horizon == 1 or first_step_weight is None:
        result = np.ones(horizon)
    else:
        result = np.full(horizon, (1 - first_step_weight) / (horizon - 1))
        result[0] = first_step_weight
    if np.any(result < 0) or result.sum() == 0:
        raise ValueError("Step weights have to be non negative and not all zero.")
    return result / result.sum()


def weighted_step_loss(weights: np.ndarray, scale: np.ndarray | None = None):
    """
    Squared error summed over the horizon with a weight per step.

    Args:
        weights: Weight per step (normalized).
        scale: Optional scale per output; errors are divided by it, so outputs of different units
               contribute alike.
    """
    w = keras.ops.convert_to_tensor(np.asarray(weights, dtype="float32").reshape(1, -1, 1))
    n_outputs = 1 if scale is None else len(scale)
    inverse = np.ones(n_outputs) if scale is None else 1.0 / np.maximum(np.asarray(scale, dtype=np.float64), 1e-12)
    s = keras.ops.convert_to_tensor(inverse.astype("float32").reshape(1, 1, -1))

    def step_weighted_mse(y_true, y_pred):
        error = (y_true - y_pred) * s
        return keras.ops.sum(w * keras.ops.mean(keras.ops.square(error), axis=-1, keepdims=True), axis=(1, 2))

    return step_weighted_mse


def fit_recursive(keras_model: keras.Model, td, config: RecursiveTraining, model_settings) -> keras.callbacks.History:
    """
    Trains the weights of `keras_model` (a single step model) on the multi step error.

    Args:
        keras_model: The step model, trained in place.
        td: `TrainingData` with the single step data.
        config: Settings of the multi step training.
        model_settings: The physXAI model, providing defaults (epochs, learning rate, ...).

    Returns:
        The keras training history of the rollout model.
    """
    import tensorflow as tf

    from physXAI.evaluation.rollout import build_rollout_model, recursion_spec, rollout_data
    from physXAI.io.run import keras_verbosity

    if getattr(td, "shift", 1) != 1:
        raise ValueError("Multi step training predicts the next step recursively and needs shift=1.")
    spec = recursion_spec(td)
    if spec is None:
        raise ValueError(
            "Multi step training requires a model that predicts one of its own inputs "
            "(e.g. output 'Change(T)' with input 'T')."
        )
    horizon = config.resolve_horizon(td)
    rollout = build_rollout_model(keras_model, td, spec, horizon)
    learning_rate = config.learning_rate
    if learning_rate is None:
        learning_rate = model_settings.learning_rate * (0.1 if config.mode == "single_then_multi" else 1.0)
    rollout.compile(
        keras.optimizers.Adam(learning_rate=learning_rate), loss=weighted_step_loss(config.weights(horizon))
    )

    train = rollout_data(td, spec, "train", horizon, config.stride)
    if train is None:
        raise ValueError(f"No training window of {horizon} steps fits into the training data.")
    batch_size = config.batch_size or model_settings.batch_size
    rows, history, truth = train
    train_ds = (
        tf.data.Dataset.from_tensor_slices(((rows, history), truth))
        .shuffle(len(truth), seed=model_settings.random_seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val = rollout_data(td, spec, "val", horizon, config.stride)
    val_ds = None
    if val is not None:
        val_ds = tf.data.Dataset.from_tensor_slices(((val[0], val[1]), val[2])).batch(batch_size)

    callbacks = list()
    patience = config.early_stopping_epochs or model_settings.early_stopping_epochs
    if patience is not None and val_ds is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    start = time.perf_counter()
    record = rollout.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs or model_settings.epochs,
        callbacks=callbacks,
        verbose=keras_verbosity(),
        shuffle=False,  # the dataset is reshuffled in every epoch
    )
    logger.info("Multi step training over %d steps took %.1f s.", horizon, time.perf_counter() - start)
    return record
