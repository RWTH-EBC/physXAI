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


def set_log_level(level: str | int) -> None:
    """
    Sets the log level of physXAI and attaches a console handler if none is configured.

    Args:
        level: 'debug', 'info', 'warning', 'error' (case insensitive) or a `logging` level.
    """
    if isinstance(level, str):
        try:
            level = _LOG_LEVELS[level.lower()]
        except KeyError:
            raise ValueError(f"Invalid log level '{level}'. Valid options are: {list(_LOG_LEVELS)}") from None
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s physXAI: %(message)s"))
        logger.addHandler(handler)


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
