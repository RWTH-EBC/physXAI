"""
Output directory of a training run.

A `Run` replaces the global `Logger` of physXAI < 1.0. It is an explicit object holding the
directory all files of a run are written to. Paths are absolute or relative to the current
working directory; they are never resolved relative to a git repository, so physXAI behaves
the same when installed as a package. Nothing is ever deleted and no interactive prompts are
shown, which makes runs safe for parameter sweeps and CI.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from physXAI import __version__

logger = logging.getLogger("physXAI")

_LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


class _ConsoleHandler(logging.StreamHandler):
    """Prints physXAI messages, unless the application configured logging itself (then the root logger does)."""

    def emit(self, record: logging.LogRecord) -> None:
        if logging.getLogger().handlers:
            return
        super().emit(record)


def set_log_level(level: str | int) -> None:
    """
    Sets the log level of physXAI. The default is 'info', which shows metrics and the progress of the training.

    Messages are printed to the console, unless the application configured logging (e.g. with
    `logging.basicConfig`), then they are passed on to the root logger.

    Args:
        level: 'debug', 'info', 'warning', 'error' (case insensitive) or a `logging` level.
    """
    if isinstance(level, str):
        try:
            level = _LOG_LEVELS[level.lower()]
        except KeyError:
            raise ValueError(f"Invalid log level '{level}'. Valid options are: {list(_LOG_LEVELS)}") from None
    logger.setLevel(level)
    if not any(isinstance(h, _ConsoleHandler) for h in logger.handlers):
        handler = _ConsoleHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s physXAI: %(message)s"))
        logger.addHandler(handler)


def keras_verbosity() -> int | str:
    """Verbosity of keras `fit` and `predict`: progress bars only on log level 'info' or lower."""
    return "auto" if logger.isEnabledFor(logging.INFO) else 0


def keras_verbosity_int() -> int:
    """Verbosity of keras callbacks: 1 on log level 'info' or lower, else 0."""
    return 1 if logger.isEnabledFor(logging.INFO) else 0


def output_path(path: str | Path, suffix: str | None = None) -> Path:
    """
    Path of a file to be written, with its directory created.

    Args:
        path: Absolute or relative to the current working directory.
        suffix: Appended if the path does not end with it yet (e.g. '.keras').
    """
    path = Path(path)
    if suffix is not None and path.suffix != suffix:
        path = path.with_name(path.name + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def input_path(path: str | Path) -> Path:
    """
    Path of a file to be read.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"'{path}' does not exist (relative paths start at '{Path.cwd()}').")
    return path


class Run:
    """
    Directory collecting all files of one training run.

    Example:
        >>> run = Run("stored_data", "TAir", overwrite=True)
        >>> run.save_json("notes.json", {"comment": "first try"})

    Args:
        base_path: Parent directory of the run. Absolute or relative to the current working directory.
        name: Name of the run directory. Defaults to a timestamp (valid on all operating systems).
        overwrite: If False, writing a file that already exists raises a `FileExistsError`.
                   Existing files are never deleted, so several models can share one run directory.
        log_level: Optional log level, see `set_log_level`.
    """

    def __init__(
        self,
        base_path: str | Path = "stored_data",
        name: str | None = None,
        overwrite: bool = False,
        log_level: str | int | None = None,
    ):
        if name is None:
            name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path: Path = Path(base_path) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.overwrite: bool = overwrite
        if log_level is not None:
            set_log_level(log_level)

    def __repr__(self) -> str:
        return f"Run('{self.path}')"

    def file(self, file_name: str | Path) -> Path:
        """
        Path of a file inside the run directory. Sub directories are created if necessary.

        Raises:
            FileExistsError: If the file exists and the run does not allow overwriting.
        """
        path = self.path / file_name
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"'{path}' already exists. Use Run(..., overwrite=True) to overwrite files.")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, file_name: str | Path, data: Any) -> Path:
        """Writes `data` as indented json into the run directory and returns the path."""
        path = self.file(file_name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, default=_json_default)
        logger.debug("Saved %s", path)
        return path

    def save(
        self,
        name: str,
        preprocessing=None,
        model=None,
        training_data=None,
        trained_model=None,
        plot: bool = True,
    ) -> dict[str, Path]:
        """
        Saves a trained model with everything needed to reproduce, compare and use it.

        Files (all optional, depending on the arguments):

        - `{name}_preprocessing.json`: config of the preprocessing (`PreprocessingData.from_config`).
        - `{name}_model.json`: config of the model (`AbstractModel.model_from_config`).
        - `{name}_training_data.json`: metrics, horizon metrics, training time and history.
        - `{name}.keras` or `{name}.joblib`: the trained model.
        - `{name}.html`: the model report (`plot=True`).

        The files follow the layout AgentLib-MPC reads (see `save_for_agentlib`), and
        `physXAI.evaluation.comparison.ModelComparison` compares saved runs.

        Args:
            name: Name of the model, the prefix of all files.
            preprocessing: The preprocessing object.
            model: The physXAI model object.
            training_data: The evaluated training data.
            trained_model: The trained model. Defaults to `model.trained_model`, set by `model.pipeline`.
            plot: Whether to save the model report (needs the model and the evaluated training data).

        Returns:
            dict: The written files by kind ('preprocessing', 'model_config', 'training_data', 'model', 'plot').
        """
        files: dict[str, Path] = dict()
        version = self.version_info()
        if preprocessing is not None:
            files["preprocessing"] = self.save_json(f"{name}_preprocessing.json", {**preprocessing.get_config(), **version})
        if model is not None:
            files["model_config"] = self.save_json(f"{name}_model.json", {**model.get_config(), **version})
        if training_data is not None:
            files["training_data"] = self.save_json(f"{name}_training_data.json", {**training_data.get_config(), **version})
        if model is not None:
            if trained_model is None:
                trained_model = getattr(model, "trained_model", None)
            if trained_model is not None:
                for suffix in (".keras", ".joblib"):
                    self.file(name + suffix)  # raises if the model exists and the run does not overwrite
                model.save_model(trained_model, str(self.path / name))
                saved = [self.path / (name + s) for s in (".keras", ".joblib") if (self.path / (name + s)).exists()]
                if saved:
                    files["model"] = saved[0]
            if plot and training_data is not None and getattr(training_data, "metrics", None) is not None:
                from physXAI.plotting.plotting import save_figure

                files["plot"] = save_figure(model.plot(training_data), self.file(f"{name}.html"))
        logger.info("Saved '%s' in %s", name, self.path)
        return files

    def save_for_agentlib(self, name: str, preprocessing, model, training_data=None, trained_model=None) -> str:
        """
        Writes all files AgentLib-MPC reads, see `physXAI.io.agentlib.save_for_agentlib`.

        Returns:
            str: `name`, which is what a training script for AgentLib-MPC has to return.
        """
        from physXAI.io.agentlib import save_for_agentlib

        return save_for_agentlib(self, name, preprocessing, model, training_data, trained_model)

    @staticmethod
    def version_info() -> dict:
        """Meta information stored with every saved config."""
        return {"physXAI_version": __version__}


def _json_default(obj):
    """Converts numpy scalars and arrays when writing json."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
