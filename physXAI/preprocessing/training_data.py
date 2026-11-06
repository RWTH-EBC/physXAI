"""
Containers for training data.

`TrainingData` holds the whole preprocessed time series as one NumPy array together with a split
label and a validity flag per row. Single step samples and multi step windows of any length are
views on that array, so the same data serves the single step training as well as the multi step
evaluation. Only NumPy objects are stored: the container can be saved and loaded without pickle
(`save` / `load`), `tf.data` pipelines are built by the models when they are fitted.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from physXAI.preprocessing.splitting import SPLITS, split_summary
from physXAI.preprocessing.windows import gather_windows, window_starts


class TrainingDataGeneric(ABC):
    """
    Common results of all training data containers: predictions, metrics, training history and
    training time.
    """

    def __init__(self):
        self.y_train_pred = None
        self.y_val_pred = None
        self.y_test_pred = None

        self.file_path = None
        self.training_record = None
        self.training_time = None
        self.metrics = None

    def add_training_record(self, data):
        """Stores the training history (e.g. the keras `History` object)."""
        self.training_record = data

    def add_predictions(self, y_train_pred: np.ndarray, y_val_pred: np.ndarray | None, y_test_pred: np.ndarray):
        """Stores the predictions of the model for the training, validation and test set."""
        self.y_train_pred = y_train_pred
        self.y_val_pred = y_val_pred
        self.y_test_pred = y_test_pred

    def add_metrics(self, metrics):
        """Stores the evaluation metrics."""
        self.metrics = metrics

    def add_training_time(self, time: float):
        """Stores the training time in seconds."""
        self.training_time = time

    def add_file_path(self, path: str):
        self.file_path = str(path)

    def _results_config(self) -> dict:
        return {
            "file_path": self.file_path,
            "metrics": self.metrics.get_config() if self.metrics is not None else None,
            "training_time": self.training_time,
            "training_record": self.training_record.history if self.training_record is not None else None,
        }

    @abstractmethod
    def get_config(self) -> dict:
        pass

    # Views on the single step samples, implemented by the subclasses
    @property
    @abstractmethod
    def X_train_single(self): ...

    @property
    @abstractmethod
    def y_train_single(self): ...

    @property
    @abstractmethod
    def X_val_single(self): ...

    @property
    @abstractmethod
    def y_val_single(self): ...

    @property
    @abstractmethod
    def X_test_single(self): ...

    @property
    @abstractmethod
    def y_test_single(self): ...

    @property
    def y_train_pred_single(self):
        return _flatten_steps(self.y_train_pred)

    @property
    def y_val_pred_single(self):
        return _flatten_steps(self.y_val_pred)

    @property
    def y_test_pred_single(self):
        return _flatten_steps(self.y_test_pred)


def _flatten_steps(y: np.ndarray | None) -> np.ndarray | None:
    """Flattens (samples, steps, outputs) to (samples * steps, outputs)."""
    if y is None:
        return None
    if y.ndim == 3:
        return y.reshape(-1, *y.shape[2:])
    return y


class TrainingData(TrainingDataGeneric):
    """
    The preprocessed time series of a run.

    Args:
        data: Array of shape (rows, columns) on a regular time grid.
        columns: Names of the columns of `data`.
        inputs: Names of the input columns of the model, in model order.
        outputs: Names of the outputs (e.g. 'Change(T)').
        targets: Names of the target columns in `data`, one per output. Row t of a target column holds
                 the value the model predicts from the inputs of row t (targets are already shifted).
                 Defaults to `outputs`.
        labels: Split label per row (0 train, 1 val, 2 test, -1 unused).
        valid: True for rows without missing values in the inputs and targets.
        time: Time stamp per row (seconds or datetime64).
        time_step: Sampling time in seconds.
        eval_horizon: Number of steps of the multi step evaluation (None: no multi step evaluation).
        eval_stride: Distance between two evaluation windows in steps.
    """

    def __init__(
        self,
        data: np.ndarray,
        columns: list[str],
        inputs: list[str],
        outputs: list[str],
        labels: np.ndarray,
        valid: np.ndarray,
        time: np.ndarray | None = None,
        time_step: float | None = None,
        eval_horizon: int | None = None,
        eval_stride: int = 1,
        targets: list[str] | None = None,
    ):
        super().__init__()
        targets = list(outputs) if targets is None else list(targets)
        if len(targets) != len(outputs):
            raise ValueError("One target column is needed per output.")
        missing = [c for c in [*inputs, *targets] if c not in columns]
        if missing:
            raise ValueError(f"Columns {missing} are not part of the data.")
        self.data = np.asarray(data, dtype=np.float64)
        self.all_columns: list[str] = list(columns)
        self.inputs: list[str] = list(inputs)
        self.outputs: list[str] = list(outputs)
        self.targets: list[str] = targets
        self.labels = np.asarray(labels, dtype=np.int8)
        self.valid = np.asarray(valid, dtype=bool)
        self.time = time if time is not None else np.arange(len(self.data), dtype=np.float64)
        self.time_step = time_step
        self.eval_horizon = eval_horizon
        self.eval_stride = eval_stride
        self._input_index = [self.all_columns.index(c) for c in self.inputs]
        self._output_index = [self.all_columns.index(c) for c in self.targets]
        self._cache: dict = dict()

    # --- generic views -------------------------------------------------------------------------
    def column(self, name: str) -> np.ndarray:
        """All values of one column."""
        return self.data[:, self.all_columns.index(name)]

    def rows(self, split: str) -> np.ndarray:
        """Indices of all valid rows of a split ('train', 'val' or 'test')."""
        key = ("rows", split)
        if key not in self._cache:
            self._cache[key] = np.flatnonzero((self.labels == SPLITS[split]) & self.valid)
        return self._cache[key]

    def X(self, split: str) -> np.ndarray:
        """Single step inputs of a split, shape (samples, inputs)."""
        return self.data[np.ix_(self.rows(split), self._input_index)].astype(np.float32)

    def y(self, split: str) -> np.ndarray:
        """Single step targets of a split, shape (samples, outputs)."""
        return self.data[np.ix_(self.rows(split), self._output_index)].astype(np.float32)

    def window_starts(self, split: str, length: int, stride: int = 1) -> np.ndarray:
        """Start rows of all windows of `length` rows within a split."""
        return window_starts(self.labels, self.valid, SPLITS[split], length, stride)

    def windows(self, split: str, length: int, stride: int = 1, columns: list[str] | None = None) -> np.ndarray:
        """Windows of a split, shape (windows, length, columns). Defaults to all columns."""
        starts = self.window_starts(split, length, stride)
        data = self.data if columns is None else self.data[:, [self.all_columns.index(c) for c in columns]]
        return gather_windows(data, starts, length)

    def has_split(self, split: str) -> bool:
        return len(self.rows(split)) > 0

    def summary(self) -> dict:
        return {"rows": len(self.data), "valid_rows": int(self.valid.sum()), **split_summary(self.labels)}

    # --- legacy views (physXAI < 1.0) ----------------------------------------------------------
    @property
    def columns(self) -> list[str]:
        """Input names (legacy name, use `inputs`)."""
        return self.inputs

    @columns.setter
    def columns(self, value: list[str]):
        self.inputs = list(value)

    def _split_or_none(self, split: str, getter):
        return getter(split) if self.has_split(split) else None

    @property
    def X_train(self):
        return self.X("train")

    @property
    def X_val(self):
        return self._split_or_none("val", self.X)

    @property
    def X_test(self):
        return self.X("test")

    @property
    def y_train(self):
        return self.y("train")

    @property
    def y_val(self):
        return self._split_or_none("val", self.y)

    @property
    def y_test(self):
        return self.y("test")

    @property
    def X_train_single(self):
        return self.X_train

    @property
    def y_train_single(self):
        return self.y_train

    @property
    def X_val_single(self):
        return self.X_val

    @property
    def y_val_single(self):
        return self.y_val

    @property
    def X_test_single(self):
        return self.X_test

    @property
    def y_test_single(self):
        return self.y_test

    # --- (de)serialization --------------------------------------------------------------------
    def get_config(self) -> dict:
        return {**self._results_config(), "summary": self.summary()}

    def _meta(self) -> dict:
        time_is_datetime = np.issubdtype(np.asarray(self.time).dtype, np.datetime64)
        return {
            "columns": self.all_columns,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "targets": self.targets,
            "time_step": self.time_step,
            "eval_horizon": self.eval_horizon,
            "eval_stride": self.eval_stride,
            "file_path": self.file_path,
            "time_is_datetime": bool(time_is_datetime),
        }

    def save(self, path: str | Path) -> Path:
        """
        Saves the data as `<path>.npz` (arrays) and `<path>.json` (names and settings).

        Returns:
            Path: The path of the npz file.
        """
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.asarray(self.time)
        if np.issubdtype(time.dtype, np.datetime64):
            time = time.astype("datetime64[ns]").astype(np.int64)
        np.savez_compressed(path.with_suffix(".npz"), data=self.data, labels=self.labels, valid=self.valid, time=time)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self._meta(), f, indent=4)
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> TrainingData:
        """Loads data saved with `save`."""
        path = Path(path).with_suffix("")
        with open(path.with_suffix(".json"), encoding="utf-8") as f:
            meta = json.load(f)
        arrays = np.load(path.with_suffix(".npz"))
        time = arrays["time"]
        if meta.pop("time_is_datetime"):
            time = time.astype("datetime64[ns]")
        file_path = meta.pop("file_path")
        td = cls(
            arrays["data"],
            meta["columns"],
            meta["inputs"],
            meta["outputs"],
            arrays["labels"],
            arrays["valid"],
            time=time,
            time_step=meta["time_step"],
            eval_horizon=meta["eval_horizon"],
            eval_stride=meta["eval_stride"],
            targets=meta["targets"],
        )
        td.file_path = file_path
        return td

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = dict()
        return state


class TrainingDataMultiStep(TrainingDataGeneric):
    """
    Multi step windows of a run, as used by recurrent models.

    The windows are stored as NumPy arrays. `train_ds`, `val_ds` and `test_ds` build batched
    `tf.data` pipelines on demand; the training data is reshuffled in every epoch.

    Args:
        X_train, X_val, X_test: Input windows (windows, steps, features), or a tuple of input
            windows and warmup windows.
        y_train, y_val, y_test: Target windows (windows, steps, outputs).
        columns: Input names.
        output: Output names.
        init_columns: Names of the warmup features.
        batch_size: Batch size of the tf.data pipelines.
        seed: Seed of the shuffling.
    """

    def __init__(
        self,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        columns: list[str],
        output: list[str],
        init_columns: list[str],
        batch_size: int = 32,
        seed: int | None = None,
    ):
        super().__init__()
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
        self.columns = columns
        self.output = output
        self.init_columns = init_columns
        self.batch_size = batch_size
        self.seed = seed

        self.single_step_metrics = None
        self.closed_loop_metrics = None
        self.y_train_pred_closed_loop = None
        self.y_val_pred_closed_loop = None
        self.y_test_pred_closed_loop = None

    def _dataset(self, X, y, shuffle: bool):
        if X is None or y is None:
            return None
        import tensorflow as tf

        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(y), seed=self.seed, reshuffle_each_iteration=True)
        return ds.batch(self.batch_size)

    @property
    def train_ds(self):
        return self._dataset(self.X_train, self.y_train, shuffle=True)

    @property
    def val_ds(self):
        return self._dataset(self.X_val, self.y_val, shuffle=False)

    @property
    def test_ds(self):
        return self._dataset(self.X_test, self.y_test, shuffle=False)

    def add_closed_loop_predictions(self, y_train_pred, y_val_pred, y_test_pred):
        """Stores the closed loop predictions (the model gets its own predictions as inputs)."""
        self.y_train_pred_closed_loop = y_train_pred
        self.y_val_pred_closed_loop = y_val_pred
        self.y_test_pred_closed_loop = y_test_pred

    def add_closed_loop_metrics(self, metrics):
        self.closed_loop_metrics = metrics

    def add_single_step_metrics(self, metrics):
        self.single_step_metrics = metrics

    def get_config(self) -> dict:
        config = self._results_config()
        config["closed_loop_metrics"] = (
            self.closed_loop_metrics.get_config() if self.closed_loop_metrics is not None else None
        )
        return config

    @staticmethod
    def _main(X):
        return X[0] if isinstance(X, tuple) else X

    @staticmethod
    def _init(X):
        return X[1] if isinstance(X, tuple) else None

    @staticmethod
    def _single(X):
        return X.reshape(-1, *X.shape[2:]) if X is not None else None

    @property
    def X_train_features(self):
        return self._main(self.X_train)

    @property
    def X_val_features(self):
        return self._main(self.X_val)

    @property
    def X_test_features(self):
        return self._main(self.X_test)

    @property
    def X_train_init(self):
        return self._init(self.X_train)

    @property
    def X_val_init(self):
        return self._init(self.X_val)

    @property
    def X_test_init(self):
        return self._init(self.X_test)

    @property
    def X_train_single(self):
        return self._single(self.X_train_features)

    @property
    def X_val_single(self):
        return self._single(self.X_val_features)

    @property
    def X_test_single(self):
        return self._single(self.X_test_features)

    @property
    def y_train_single(self):
        return self._single(self.y_train)

    @property
    def y_val_single(self):
        return self._single(self.y_val)

    @property
    def y_test_single(self):
        return self._single(self.y_test)
