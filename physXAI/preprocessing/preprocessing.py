"""
Preprocessing of time series data for single step and multi step models.

Both preprocessings share one pipeline:

1. Load the csv file onto a regular time grid of `time_step` (missing time stamps become rows
   with missing values instead of silently joining the neighbouring rows).
2. Apply an optional `transform(df) -> df` hook (not serialized).
3. Create derived columns from the naming conventions used by AgentLib-MPC:
   `X_lagN` is `X` shifted by N steps, `Change(X)` is `X(t) - X(t-1)`. Lags can also be given as
   `lags={'X': 2}`, which adds `X_lag1` and `X_lag2` directly after `X` to the inputs.
4. Shift the outputs by `shift` steps, so row t holds the target predicted from the inputs at t.
5. Mark rows with missing values as invalid (with `ignore_nan=False`, missing values between
   valid rows raise an error).
6. Split the rows leakage free into training, validation and test data
   (`physXAI.preprocessing.splitting.BlockSplit`).

The class names `PreprocessingSingleStep` and `PreprocessingMultiStep` and the keys of
`get_config()` are read by AgentLib-MPC and must not change.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from physXAI.expr.serialization import to_config as expr_to_config
from physXAI.preprocessing.splitting import BlockSplit, split_summary
from physXAI.preprocessing.training_data import TrainingData, TrainingDataGeneric, TrainingDataMultiStep
from physXAI.utils.seed import resolve_seed

logger = logging.getLogger("physXAI")

LAG_PATTERN = re.compile(r"^(.*)_lag(\d+)$")
CHANGE_PATTERN = re.compile(r"^Change\((.*)\)$")
TARGET_PREFIX = "y:"

PREPROCESSING_CLASS_REGISTRY: dict[str, type[PreprocessingData]] = dict()


def register_preprocessing(cls):
    PREPROCESSING_CLASS_REGISTRY[cls.__name__] = cls
    return cls


def expand_lags(inputs: list[str], lags: dict[str, int] | None) -> list[str]:
    """
    Adds lagged inputs: `lags={'X': 2}` inserts 'X_lag1', 'X_lag2' directly after 'X'.
    Lags of features which are no input are appended at the end.
    """
    if not lags:
        return list(inputs)
    result = list()
    for name in inputs:
        result.append(name)
        for lag in range(1, lags.get(name, 0) + 1):
            if f"{name}_lag{lag}" not in inputs:
                result.append(f"{name}_lag{lag}")
    for name, n in lags.items():
        if name not in inputs:
            result.extend(f"{name}_lag{lag}" for lag in range(1, n + 1))
    return result


def max_lag(names: list[str]) -> int:
    """Largest lag of a list of feature names."""
    lags = [int(m.group(2)) for m in (LAG_PATTERN.match(n) for n in names) if m]
    return max(lags, default=0)


def ensure_column(df: pd.DataFrame, name: str) -> None:
    """
    Creates `name` from the naming conventions if it is not a column of `df`.

    Raises:
        KeyError: If the column neither exists nor can be derived.
    """
    if name in df.columns:
        return
    lag = LAG_PATTERN.match(name)
    if lag:
        base, n = lag.group(1), int(lag.group(2))
        ensure_column(df, base)
        df[name] = df[base].shift(n)
        return
    change = CHANGE_PATTERN.match(name)
    if change:
        base = change.group(1).strip()
        ensure_column(df, base)
        df[name] = df[base] - df[base].shift(1)
        return
    raise KeyError(f"Column '{name}' is not part of the data and cannot be derived from it.")


def _split_lag(name: str) -> tuple[str, int]:
    match = LAG_PATTERN.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, 0)


def output_feature(output: str) -> str:
    """Feature predicted by an output: 'T' for 'Change(T)' and for 'T'."""
    change = CHANGE_PATTERN.match(output)
    return change.group(1).strip() if change else output


def order_inputs(names: list[str], outputs: list[str]) -> list[str]:
    """
    Orders input names as AgentLib-MPC expects them: lags directly after their feature in
    ascending order (missing intermediate lags are added), the recursive feature (predicted by an
    output) and its lags last.

    Example:
        >>> order_inputs(["T_lag2", "u", "T"], ["Change(T)"])
        ['u', 'T', 'T_lag1', 'T_lag2']
    """
    groups: dict[str, list[int]] = dict()
    for name in names:
        base, lag = _split_lag(name)
        groups.setdefault(base, []).append(lag)
    recursive = {output_feature(o) for o in outputs}
    ordered = [b for b in groups if b not in recursive] + [b for b in groups if b in recursive]
    result = list()
    for base in ordered:
        lags = groups[base]
        for lag in range(min(lags), max(lags) + 1):
            result.append(base if lag == 0 else f"{base}_lag{lag}")
    return result


def _try_ensure_column(df: pd.DataFrame, name: str) -> bool:
    try:
        ensure_column(df, name)
        return True
    except KeyError:
        return False


class PreprocessingData(ABC):
    """
    Common preprocessing of single step and multi step models, see the module docstring.

    Args:
        inputs: Input features of the model. Lagged inputs ('X_lag1') and changes ('Change(X)')
                are derived automatically. If None, the inputs are taken from the model passed to
                `pipeline` (the free variables of its expression graph).
        output: Output feature(s), e.g. 'Change(T)' for a model predicting the change of T.
        columns: Optional derived data columns {name: expression}, e.g. {'Q': v.mDot * (v.T_in - v.T)}.
                 Prefer expressions inside the model, then the MPC does not need to compute them.
        lags: Optional number of lags per feature, e.g. {'T': 2}.
        shift: Number of steps between the inputs and the predicted output (1: next time step).
        time_step: Sampling time in seconds. The data is sub-sampled to it. Defaults to the
                   sampling time of the data.
        test_size: Fraction of the data used for testing.
        val_size: Fraction of the data used for validation.
        random_state: Seed of the split. Defaults to the global seed (`physXAI.set_seed`), else a
                      new seed is drawn. The used seed is stored in the config.
        split: Split strategy ('blocks', 'holdout_end', 'random_rows') or a `BlockSplit`.
        block_length: Block length of the split in steps or as time delta string ('7D').
        gap_steps: Embargo between splits in steps. Defaults to the maximum lag + shift.
        time_index_col: Name or index of the time column of the csv file.
        csv_delimiter, csv_encoding, csv_header, csv_skiprows: Options of `pandas.read_csv`.
        ignore_nan: If False, missing values between valid rows raise an error. If True, rows
                    with missing values are left out.
        transform: Optional function `df -> df` applied after loading, e.g. to add columns that are
                   no inputs of the model. It is not serialized.
    """

    def __init__(
        self,
        inputs: list[str] | None = None,
        output: str | list[str] | None = None,
        lags: dict[str, int] | None = None,
        columns: dict | None = None,
        shift: int = 1,
        time_step: float | None = None,
        test_size: float = 0.1,
        val_size: float = 0.1,
        random_state: int | None = None,
        split: str | BlockSplit | dict = "blocks",
        block_length: int | str | None = None,
        gap_steps: int | None = None,
        time_index_col: str | int = 0,
        csv_delimiter: str = ";",
        csv_encoding: str = "latin1",
        csv_header: int = 0,
        csv_skiprows: int | list[int] | None = None,
        ignore_nan: bool = False,
        transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        **kwargs,
    ):
        if shift < 1:
            raise ValueError("shift has to be at least 1.")
        if output is None:
            raise ValueError("An output is required.")
        self.inputs: list[str] | None = expand_lags(list(inputs), lags) if inputs is not None else None
        self._lags = lags if inputs is None else None
        self.output: list[str] = [output] if isinstance(output, str) else list(output)
        from physXAI.expr import from_config as expr_from_config

        self.columns: dict = {
            name: expr_from_config(e) if isinstance(e, dict) else e for name, e in (columns or {}).items()
        }
        self.shift: int = shift
        self.time_step: float | None = time_step
        self.random_state: int = resolve_seed(random_state)

        if isinstance(split, dict):
            split = BlockSplit.from_config(split)
        elif isinstance(split, str):
            split = BlockSplit(test_size, val_size, strategy=split, block_length=block_length, gap_steps=gap_steps)
        self.split: BlockSplit = split

        self.time_index_col = time_index_col
        self.csv_delimiter = csv_delimiter
        self.csv_encoding = csv_encoding
        self.csv_header = csv_header
        self.csv_skiprows = csv_skiprows
        self.ignore_nan: bool = ignore_nan
        self.transform = transform

    # The split sizes are part of the AgentLib-MPC contract
    @property
    def test_size(self) -> float:
        return self.split.test_size

    @property
    def val_size(self) -> float:
        return self.split.val_size

    # --- loading --------------------------------------------------------------------------------
    def load_data(self, file_path: str | Path) -> pd.DataFrame:
        """
        Loads a csv file onto a regular time grid of `time_step`.

        The time column may hold seconds or date times. Time stamps which are not on the grid are
        dropped; missing time stamps become rows with missing values.

        Args:
            file_path: Path to the csv file (absolute or relative to the working directory).

        Returns:
            pd.DataFrame: Data indexed by the regular time grid.
        """
        df = pd.read_csv(
            Path(file_path),
            delimiter=self.csv_delimiter,
            index_col=self.time_index_col,
            encoding=self.csv_encoding,
            header=self.csv_header,
            skiprows=self.csv_skiprows,
        )
        df = df[~df.index.duplicated(keep="first")].sort_index()

        index = df.index
        if not pd.api.types.is_numeric_dtype(index) and not isinstance(index, pd.DatetimeIndex):
            index = pd.to_datetime(index)
        is_datetime = isinstance(index, pd.DatetimeIndex)
        seconds = ((index - index[0]).total_seconds() if is_datetime else index - index[0]).to_numpy(dtype=float)

        data_step = _sampling_time(seconds)
        if self.time_step is None:
            self.time_step = data_step
        elif not _is_multiple(self.time_step, data_step):
            raise ValueError(f"Given time step {self.time_step} is not a multiple of data time step {data_step}.")

        positions = seconds / self.time_step
        on_grid = np.isclose(positions, np.round(positions), atol=1e-6)
        df = df.iloc[on_grid]
        positions = np.round(positions[on_grid]).astype(np.int64)
        df.index = positions
        df = df.reindex(np.arange(positions[-1] + 1))

        if is_datetime:
            df.index = index[0] + pd.to_timedelta(df.index * self.time_step, unit="s")
        else:
            df.index = float(index[0]) + df.index * self.time_step
        df.index.name = "time"
        return df

    # --- processing -----------------------------------------------------------------------------
    def _required_columns(self) -> list[str]:
        """Raw columns (besides inputs and outputs) needed in the data."""
        return [output_feature(out) for out in self.output]

    def _window_length(self) -> int:
        return 1

    def _default_gap(self) -> int:
        return max_lag(self.inputs) + self.shift

    def build_training_data(
        self,
        df: pd.DataFrame,
        extra_targets: list[str] | None = None,
        eval_horizon: int | None = None,
        eval_stride: int = 1,
    ) -> TrainingData:
        """
        Turns a loaded DataFrame into `TrainingData` (derived columns, shifted targets, split).

        Args:
            df: Data on a regular time grid, see `load_data`.
            extra_targets: Further columns to shift like the outputs (e.g. warmup features).
            eval_horizon: Steps of the multi step evaluation stored in the training data.
            eval_stride: Distance between two evaluation windows.
        """
        df = df.copy()
        if self.transform is not None:
            df = self.transform(df)

        for name, expression in self.columns.items():
            for variable in expression.free_vars():
                ensure_column(df, variable)
            df[name] = np.broadcast_to(expression.evaluate(df), (len(df),))
        shifted = list(dict.fromkeys([*self.output, *(extra_targets or [])]))
        for name in dict.fromkeys([*self.inputs, *shifted]):
            ensure_column(df, name)
        # raw features of the outputs and the variables of derived columns are kept for recursive
        # (multi step) evaluations, if available
        candidates = [*self._required_columns(), *(v for e in self.columns.values() for v in e.free_vars())]
        aux = [c for c in dict.fromkeys(candidates) if c not in self.inputs and _try_ensure_column(df, c)]

        target_columns = [TARGET_PREFIX + name for name in shifted]
        targets = pd.DataFrame({TARGET_PREFIX + name: df[name].shift(-self.shift) for name in shifted}, index=df.index)
        columns = list(dict.fromkeys([*self.inputs, *aux]))
        frame = pd.concat([df[columns], targets], axis=1)

        used = [*self.inputs, *target_columns]
        valid = frame[used].notna().all(axis=1).to_numpy()
        if not valid.any():
            raise ValueError("Data Error: No row holds valid values for all inputs and outputs.")
        first, last = np.flatnonzero(valid)[[0, -1]]
        if not valid[first : last + 1].all() and not self.ignore_nan:
            raise ValueError(
                "Data Error: The data contains missing values (or missing time stamps) in intermediate rows. "
                f"If this is intended, set ignore_nan=True in {self.__class__.__name__}."
            )

        labels = self.split.assign(
            len(frame), self.random_state, self._window_length(), self.time_step, self._default_gap()
        )
        td = TrainingData(
            frame.to_numpy(dtype=np.float64),
            list(frame.columns),
            self.inputs,
            self.output,
            labels,
            valid,
            time=frame.index.to_numpy(),
            time_step=self.time_step,
            eval_horizon=eval_horizon,
            eval_stride=eval_stride,
            targets=[TARGET_PREFIX + name for name in self.output],
            derived=self.columns,
        )
        logger.info("Split of the data: %s", {k: round(v, 3) for k, v in split_summary(labels[valid]).items()})
        return td

    def resolve_inputs(self, model=None) -> list[str]:
        """
        Determines the inputs: the given ones or, if None, the inputs required by the model.

        Raises:
            ValueError: If the model needs inputs which are not part of the given inputs.
        """
        required = getattr(model, "required_inputs", None) if model is not None else None
        if self.inputs is None:
            if required is None:
                raise ValueError("No inputs given: pass inputs or a model with an expression graph to pipeline().")
            self.inputs = order_inputs(expand_lags(required, self._lags), self.output)
        elif required is not None:
            missing = [name for name in required if name not in self.inputs]
            if missing:
                raise ValueError(
                    f"The model needs the inputs {missing}, which are not part of the inputs {self.inputs}."
                )
        return self.inputs

    @abstractmethod
    def pipeline(self, file_path: str | Path, model=None) -> TrainingDataGeneric:
        """Loads and preprocesses a csv file. Inputs can be taken from `model`."""

    # --- (de)serialization ----------------------------------------------------------------------
    def get_config(self) -> dict:
        return {
            "__class_name__": self.__class__.__name__,
            "inputs": self.inputs,
            "output": self.output,
            "columns": {name: expr_to_config(e) for name, e in self.columns.items()} or None,
            "shift": self.shift,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "time_step": self.time_step,
            "split": self.split.get_config(),
            "time_index_col": self.time_index_col,
            "csv_delimiter": self.csv_delimiter,
            "csv_encoding": self.csv_encoding,
            "csv_header": self.csv_header,
            "csv_skiprows": self.csv_skiprows,
            "ignore_nan": self.ignore_nan,
            "transform": getattr(self.transform, "__name__", None) if self.transform is not None else None,
        }

    @classmethod
    def from_config(cls, config: dict) -> PreprocessingData:
        """
        Creates a preprocessing from its config. Called on `PreprocessingData`, the class is taken
        from '__class_name__'. A `transform` hook is not serialized and has to be set again.
        """
        config = dict(config)
        class_name = config.pop("__class_name__", cls.__name__)
        target = cls if cls is not PreprocessingData else PREPROCESSING_CLASS_REGISTRY[class_name]
        config.pop("physXAI_version", None)
        transform = config.pop("transform", None)
        if isinstance(transform, str):
            logger.warning("The transform '%s' is not serialized and has to be set again.", transform)
            transform = None
        if isinstance(config.get("split"), dict):
            split = config["split"]
            config["test_size"], config["val_size"] = split["test_size"], split["val_size"]
        return target(transform=transform, **config)


def _sampling_time(seconds: np.ndarray) -> float:
    """Most frequent positive difference of the time stamps."""
    diffs = np.diff(seconds)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("Data Error: The data needs at least two time stamps.")
    values, counts = np.unique(np.round(diffs, 6), return_counts=True)
    return float(values[np.argmax(counts)])


def _is_multiple(value: float, base: float) -> bool:
    ratio = value / base
    return bool(np.isclose(ratio, round(ratio)) and round(ratio) >= 1)


def _resolve_steps(value: int | str | None, time_step: float) -> int | None:
    """Converts a number of steps or a time delta string ('24h') into steps."""
    if value is None or isinstance(value, (int, np.integer)):
        return value
    return int(round(pd.Timedelta(value).total_seconds() / time_step))


@register_preprocessing
class PreprocessingSingleStep(PreprocessingData):
    """
    Preprocessing of single step models, which predict the output of the next time step.

    Args:
        eval_horizon: Optional horizon of the multi step (recursive) evaluation of the model, in
                      steps or as time delta string (e.g. '24h').
        eval_stride: Distance between two evaluation windows, in steps or as time delta string.
                     Defaults to a quarter of the horizon.
        **kwargs: See `PreprocessingData`.
    """

    def __init__(
        self,
        inputs: list[str] | None = None,
        output: str | list[str] | None = None,
        eval_horizon: int | str | None = None,
        eval_stride: int | str | None = None,
        **kwargs,
    ):
        super().__init__(inputs, output, **kwargs)
        self.eval_horizon = eval_horizon
        self.eval_stride = eval_stride

    def _horizon_steps(self) -> tuple[int | None, int]:
        horizon = _resolve_steps(self.eval_horizon, self.time_step) if self.time_step else None
        stride = _resolve_steps(self.eval_stride, self.time_step) if self.time_step else None
        if stride is None:
            stride = max((horizon or 4) // 4, 1)
        return horizon, stride

    def _window_length(self) -> int:
        horizon, _ = self._horizon_steps()
        return horizon or 1

    def pipeline(self, file_path: str | Path, model=None) -> TrainingData:
        """
        Loads and preprocesses a csv file.

        Returns:
            TrainingData: Single step data, split leakage free.

        Args:
            file_path: Path to the csv file.
            model: Optional model; without explicit inputs, its required inputs are used.
        """
        self.resolve_inputs(model)
        df = self.load_data(file_path)
        horizon, stride = self._horizon_steps()
        td = self.build_training_data(df, eval_horizon=horizon, eval_stride=stride)
        td.add_file_path(file_path)
        return td

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"eval_horizon": self.eval_horizon, "eval_stride": self.eval_stride})
        return config


@register_preprocessing
class PreprocessingMultiStep(PreprocessingData):
    """
    Preprocessing of multi step (recurrent) models.

    A sequence holds `warmup_width` steps, on which the model warms up starting from a zero state,
    followed by `label_width` predicted steps. AgentLib-MPC warms the model up the same way before
    every MPC step, so training and MPC match. Sequences are only cut out of one split, so training
    and test sequences never overlap.

    Predicting the change ('Change(T)') is recommended: the trajectory of T is then the measured
    start value plus the accumulated changes, and the model is trained on that trajectory.

    Args:
        inputs: Input features of the model (no lags: the memory of a recurrent model is its state).
        output: Output feature(s), e.g. `px.Change('T')`.
        label_width: Number of predicted steps (training horizon), in steps or as time delta string
                     ('12h'). Choose it longer than the MPC horizon, so the model stays stable over
                     the whole horizon.
        warmup_width: Number of warmup steps, in steps or as time delta string.
        overlapping_sequences: If True, training sequences start at every step, else every
                               `label_width` steps.
        eval_horizon: Number of predicted steps of the evaluation (steps or time delta string).
                      Defaults to `label_width`; a longer horizon shows the stability beyond training.
        eval_stride: Distance between two evaluation sequences. Defaults to a quarter of the horizon.
        **kwargs: See `PreprocessingData`.
    """

    def __init__(
        self,
        inputs: list[str] | None = None,
        output: str | list[str] | None = None,
        label_width: int | str = 1,
        warmup_width: int | str = 0,
        overlapping_sequences: bool = True,
        eval_horizon: int | str | None = None,
        eval_stride: int | str | None = None,
        **kwargs,
    ):
        for legacy in ("warmup_columns_input", "warmup_columns_labels"):
            kwargs.pop(legacy, None)
        init_features = kwargs.pop("init_features", None)
        if "batch_size" in kwargs:
            kwargs.pop("batch_size")
            logger.warning("PreprocessingMultiStep(batch_size=...) is ignored, set the batch size of the model.")
        super().__init__(inputs, output, **kwargs)
        if init_features is not None and (self.inputs is None or list(init_features) != list(self.inputs)):
            raise ValueError(
                "'init_features' was removed in physXAI 1.0: there is no separate initialization model "
                "anymore, the model warms itself up on its inputs (as in AgentLib-MPC). To start from the "
                f"measured value of a feature, predict its change, e.g. output=px.Change('{init_features[0]}')."
            )
        self.label_width = label_width
        self.warmup_width = warmup_width
        self.overlapping_sequences = overlapping_sequences
        self.eval_horizon = eval_horizon
        self.eval_stride = eval_stride

    def _steps(self) -> dict[str, int]:
        """Widths and strides in steps (the time step is known after loading the data)."""
        label = _resolve_steps(self.label_width, self.time_step) if self.time_step else self.label_width
        warmup = _resolve_steps(self.warmup_width, self.time_step) if self.time_step else self.warmup_width
        if not isinstance(label, (int, np.integer)) or label < 1:
            raise ValueError(f"label_width has to be at least one step, got {self.label_width}.")
        if not isinstance(warmup, (int, np.integer)) or warmup < 0:
            raise ValueError(f"warmup_width must not be negative, got {self.warmup_width}.")
        horizon = (_resolve_steps(self.eval_horizon, self.time_step) if self.time_step else self.eval_horizon) or label
        stride = 1 if self.overlapping_sequences else label
        eval_stride = _resolve_steps(self.eval_stride, self.time_step) if self.time_step else self.eval_stride
        return {
            "label": int(label),
            "warmup": int(warmup),
            "horizon": int(horizon),
            "stride": int(stride),
            "eval_stride": int(eval_stride or max(horizon // 4, 1)),
        }

    def _window_length(self) -> int:
        steps = self._steps()
        return steps["warmup"] + max(steps["label"], steps["horizon"]) + self.shift

    def _default_gap(self) -> int:
        return super()._default_gap() + self._steps()["warmup"]

    def pipeline(self, file_path: str | Path, model=None) -> TrainingDataMultiStep:
        """
        Loads a csv file and prepares the sequences.

        Returns:
            TrainingDataMultiStep: Sequences of every split, split leakage free.

        Args:
            file_path: Path to the csv file.
            model: Optional model; without explicit inputs, its required inputs are used.
        """
        self.resolve_inputs(model)
        df = self.load_data(file_path)
        series = self.build_training_data(df)
        steps = self._steps()
        td = TrainingDataMultiStep(
            series,
            warmup_width=steps["warmup"],
            label_width=steps["label"],
            stride=steps["stride"],
            eval_horizon=steps["horizon"],
            eval_stride=steps["eval_stride"],
        )
        td.add_file_path(file_path)
        if not td.has_split("train"):
            raise ValueError(
                f"No training sequence of {steps['warmup'] + steps['label']} steps fits into the data. "
                "Reduce label_width / warmup_width or the block length of the split."
            )
        return td

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "label_width": self.label_width,
                "warmup_width": self.warmup_width,
                "overlapping_sequences": self.overlapping_sequences,
                "eval_horizon": self.eval_horizon,
                "eval_stride": self.eval_stride,
            }
        )
        return config
