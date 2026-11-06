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
from types import SimpleNamespace

import numpy as np

from physXAI.expr.serialization import from_config, to_config
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
        #: History of the training on the multi step error (`physXAI.training.RecursiveTraining`)
        self.recursive_training_record = None
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

    def __getstate__(self):
        # keras histories reference their (possibly not picklable) model: keep the recorded values only
        state = self.__dict__.copy()
        for key in ("training_record", "recursive_training_record"):
            record = state.get(key)
            if record is not None and hasattr(record, "history"):
                state[key] = SimpleNamespace(history=record.history)
        return state

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
        derived: Expressions of derived data columns {name: expression}. They are recomputed in every
                 step of a recursive (multi step) prediction.
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
        derived: dict | None = None,
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
        self.derived: dict = dict(derived or {})
        self._cache: dict = dict()

        #: Results of the recursive multi step evaluation (`physXAI.evaluation.rollout`)
        self.horizon_metrics = None
        self.horizon_predictions: dict | None = None

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
        return {
            **self._results_config(),
            "horizon_metrics": self.horizon_metrics.get_config() if self.horizon_metrics is not None else None,
            "summary": self.summary(),
        }

    def _meta(self) -> dict:
        time_is_datetime = np.issubdtype(np.asarray(self.time).dtype, np.datetime64)
        return {
            "columns": self.all_columns,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "targets": self.targets,
            "derived": {name: to_config(e) for name, e in self.derived.items()},
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
            derived={name: from_config(e) for name, e in meta.get("derived", {}).items()},
        )
        td.file_path = file_path
        return td

    def __getstate__(self):
        state = super().__getstate__()
        state["_cache"] = dict()
        return state


class TrainingDataMultiStep(TrainingDataGeneric):
    """
    Sequences of a run, as used by recurrent (multi step) models.

    A sequence starts `warmup_width` steps before the prediction: the model starts from a zero state
    and warms up on the measured past, as it does in an MPC. Only the following `label_width` steps
    are predicted (the warmup is not part of the loss). The sequences are views on the time series
    (`series`), so evaluation sequences can be longer than the training sequences (`eval_horizon`).

    For an output 'Change(T)' the model predicts the change of T. Its trajectory is the measured
    start value T(t0) plus the accumulated changes; the `y_*` attributes hold this trajectory, the
    targets of the outputs themselves are available via `targets(split)`.

    Args:
        series: The preprocessed time series (`TrainingData`).
        warmup_width: Number of warmup steps.
        label_width: Number of predicted steps (training horizon).
        stride: Distance between two training sequences.
        eval_horizon: Number of predicted steps of the evaluation. Defaults to `label_width`.
        eval_stride: Distance between two evaluation sequences. Defaults to `stride`.
    """

    def __init__(
        self,
        series: TrainingData,
        warmup_width: int,
        label_width: int,
        stride: int = 1,
        eval_horizon: int | None = None,
        eval_stride: int | None = None,
    ):
        super().__init__()
        from physXAI.preprocessing.preprocessing import CHANGE_PATTERN, output_feature

        self.series = series
        self.warmup_width = int(warmup_width)
        self.label_width = int(label_width)
        self.stride = int(stride)
        self.eval_horizon = int(eval_horizon or label_width)
        self.eval_stride = int(eval_stride or stride)
        self.inputs: list[str] = list(series.inputs)
        self.outputs: list[str] = list(series.outputs)
        #: Predicted feature of every output ('T' for 'Change(T)')
        self.features: list[str] = [output_feature(o) for o in self.outputs]
        #: Whether an output is the change of its feature
        self.difference: list[bool] = [CHANGE_PATTERN.match(o) is not None for o in self.outputs]
        #: Whether the start values of the features are known, then trajectories are absolute
        self.absolute: bool = all(f in series.all_columns for f in self.features)
        self.time_step = series.time_step
        self._cache: dict = dict()

        #: Results of the evaluation over `eval_horizon` steps (`physXAI.evaluation.rollout.HorizonMetrics`)
        self.horizon_metrics = None
        self.horizon_predictions: dict | None = None

    # --- sequences ------------------------------------------------------------------------------
    def sequences(self, split: str, horizon: int | None = None, stride: int | None = None):
        """
        Sequences of a split.

        Args:
            split: 'train', 'val' or 'test'.
            horizon: Number of predicted steps. Defaults to `label_width`.
            stride: Distance between two sequences. Defaults to `stride`.

        Returns:
            tuple: inputs (n, warmup + horizon, inputs), targets of the outputs (n, horizon, outputs)
                   and start values of the features (n, outputs) or None, if `absolute` is False.
                   None, if the split holds no sequence.
        """
        horizon = horizon or self.label_width
        stride = stride or self.stride
        key = (split, horizon, stride)
        if key not in self._cache:
            self._cache[key] = self._sequences(split, horizon, stride)
        return self._cache[key]

    def _sequences(self, split: str, horizon: int, stride: int):
        series = self.series
        length = self.warmup_width + horizon
        starts = series.window_starts(split, length, stride)
        initial = None
        if self.absolute:
            first = starts + self.warmup_width
            initial = np.stack([series.column(f)[first] for f in self.features], axis=1)
            keep = np.all(np.isfinite(initial), axis=1)
            starts, initial = starts[keep], initial[keep].astype(np.float32)
        if len(starts) == 0:
            return None
        X = gather_windows(series.data[:, series._input_index], starts, length).astype(np.float32)
        targets = gather_windows(series.data[:, series._output_index], starts + self.warmup_width, horizon)
        return X, targets.astype(np.float32), initial

    def trajectory(self, targets: np.ndarray, initial: np.ndarray | None) -> np.ndarray:
        """True trajectory of the features: accumulated changes plus start value (if known)."""
        if initial is None:
            return targets
        result = targets.astype(np.float64).copy()
        for j, difference in enumerate(self.difference):
            if difference:
                result[..., j] = initial[:, None, j] + np.cumsum(targets[..., j], axis=1)
        return result.astype(np.float32)

    def has_split(self, split: str) -> bool:
        return self.sequences(split) is not None

    def X(self, split: str):
        """Input sequences (warmup and prediction) of a split."""
        data = self.sequences(split)
        return None if data is None else data[0]

    def targets(self, split: str):
        """Targets of the outputs (e.g. the changes) over the prediction steps of a split."""
        data = self.sequences(split)
        return None if data is None else data[1]

    def initial(self, split: str):
        """Start values of the predicted features of a split (None if unknown)."""
        data = self.sequences(split)
        return None if data is None else data[2]

    def y(self, split: str):
        """True trajectories of the predicted features over the prediction steps of a split."""
        data = self.sequences(split)
        return None if data is None else self.trajectory(data[1], data[2])

    @property
    def X_train(self):
        return self.X("train")

    @property
    def X_val(self):
        return self.X("val")

    @property
    def X_test(self):
        return self.X("test")

    @property
    def y_train(self):
        return self.y("train")

    @property
    def y_val(self):
        return self.y("val")

    @property
    def y_test(self):
        return self.y("test")

    # legacy names
    @property
    def columns(self) -> list[str]:
        """Input names (legacy name, use `inputs`)."""
        return self.inputs

    @property
    def output(self) -> list[str]:
        """Output names (legacy name, use `outputs`)."""
        return self.outputs

    @staticmethod
    def _single(X):
        return X.reshape(-1, *X.shape[2:]) if X is not None else None

    @property
    def X_train_single(self):
        return self._single(self.X_train)

    @property
    def X_val_single(self):
        return self._single(self.X_val)

    @property
    def X_test_single(self):
        return self._single(self.X_test)

    @property
    def y_train_single(self):
        return self._single(self.y_train)

    @property
    def y_val_single(self):
        return self._single(self.y_val)

    @property
    def y_test_single(self):
        return self._single(self.y_test)

    def summary(self) -> dict:
        sequences = dict()
        for split in SPLITS:
            data = self.sequences(split)
            sequences[split] = 0 if data is None else len(data[0])
        return {**self.series.summary(), "sequences": sequences}

    def get_config(self) -> dict:
        return {
            **self._results_config(),
            "warmup_width": self.warmup_width,
            "label_width": self.label_width,
            "eval_horizon": self.eval_horizon,
            "horizon_metrics": self.horizon_metrics.get_config() if self.horizon_metrics is not None else None,
            "summary": self.summary(),
        }

    def __getstate__(self):
        state = super().__getstate__()
        state["_cache"] = dict()
        return state
