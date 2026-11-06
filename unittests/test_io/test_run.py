import json
import logging

import numpy as np
import pytest

from physXAI import __version__
from physXAI.io import Run, set_log_level


def test_default_name_is_valid_on_all_platforms(tmp_path):
    run = Run(tmp_path)
    assert run.path.is_dir()
    assert ":" not in run.path.name


def test_files_are_not_overwritten_by_default(tmp_path):
    run = Run(tmp_path, "run")
    run.save_json("a.json", {"x": 1})
    with pytest.raises(FileExistsError):
        run.save_json("a.json", {"x": 2})


def test_overwrite_keeps_other_files(tmp_path):
    """Several models share one run directory in AgentLib-MPC, nothing may be deleted."""
    Run(tmp_path, "run").save_json("a.json", {"x": 1})
    run = Run(tmp_path, "run", overwrite=True)
    run.save_json("b.json", {"y": np.float32(2.5), "z": np.arange(2)})
    run.save_json("b.json", {"y": 3})
    assert json.loads((run.path / "a.json").read_text()) == {"x": 1}
    assert json.loads((run.path / "b.json").read_text()) == {"y": 3}


def test_relative_paths_use_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = Run("out", "r1")
    assert run.path.resolve() == (tmp_path / "out" / "r1").resolve()


def test_version_info():
    assert Run.version_info() == {"physXAI_version": __version__}


def test_log_level():
    set_log_level("WARNING")
    assert logging.getLogger("physXAI").level == logging.WARNING
    with pytest.raises(ValueError):
        set_log_level("loud")
