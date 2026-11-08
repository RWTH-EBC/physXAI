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
    import keras
    import numpy

    info = Run.version_info()
    assert info["physXAI_version"] == __version__
    assert info["environment"]["numpy"] == numpy.__version__
    assert info["environment"]["keras"] == keras.__version__
    assert {"python", "tensorflow", "scikit-learn", "casadi"} <= info["environment"].keys()


def test_log_level():
    set_log_level("WARNING")
    assert logging.getLogger("physXAI").level == logging.WARNING
    with pytest.raises(ValueError):
        set_log_level("loud")


def test_keras_verbosity_follows_the_log_level():
    from physXAI.io.run import keras_verbosity, keras_verbosity_int

    set_log_level("info")
    assert keras_verbosity() == "auto" and keras_verbosity_int() == 1
    set_log_level("warning")
    assert keras_verbosity() == 0 and keras_verbosity_int() == 0
    set_log_level("info")


def test_messages_go_to_the_root_logger_if_configured(monkeypatch):
    """physXAI prints its messages itself, unless the application configured logging."""
    import io

    from physXAI.io.run import _ConsoleHandler

    stream = io.StringIO()
    handler = _ConsoleHandler(stream)
    record = logging.LogRecord("physXAI", logging.INFO, __file__, 1, "message", None, None)
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    handler.emit(record)
    monkeypatch.setattr(root, "handlers", [logging.NullHandler()])
    handler.emit(record)
    assert stream.getvalue().count("message") == 1


def test_save(tmp_path, one_room_csv):
    from physXAI.models import LinearRegressionModel
    from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

    prep = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)", csv_delimiter=",")
    td = prep.pipeline(one_room_csv)
    m = LinearRegressionModel()
    m.pipeline(td)
    run = Run(tmp_path, "run")
    files = run.save("lr", prep, m, td)
    assert {p.name for p in files.values()} == {
        "lr_preprocessing.json",
        "lr_model.json",
        "lr_training_data.json",
        "lr.joblib",
        "lr.html",
    }
    saved = json.loads(files["training_data"].read_text())
    assert saved["physXAI_version"] == __version__ and "by_output" in saved["metrics"]
    # the saved configs, with their meta information, create the objects again
    from physXAI.models.models import AbstractModel
    from physXAI.preprocessing.preprocessing import PreprocessingData

    saved_prep = json.loads(files["preprocessing"].read_text())
    assert "environment" in saved_prep
    assert PreprocessingData.from_config(saved_prep).get_config() == prep.get_config()
    loaded = AbstractModel.model_from_config(json.loads(files["model_config"].read_text()))
    assert isinstance(loaded, LinearRegressionModel)
    with pytest.raises(FileExistsError):
        run.save("lr", prep, m, td)
    # only configs, without a trained model
    assert set(Run(tmp_path, "configs").save("x", prep, LinearRegressionModel())) == {"preprocessing", "model_config"}
