"""Comparison of several models with several runs each."""

import json

import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.evaluation.comparison import ModelComparison, repeat_training
from physXAI.io import Run
from physXAI.models import LinearRegressionModel
from physXAI.models.ann.ann_design import ClassicalANNModel, RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


@pytest.fixture(scope="module")
def prep():
    return PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", eval_horizon=6,
                                   random_state=7)


@pytest.fixture(scope="module")
def saved(csv, prep, tmp_path_factory):
    """Linear regression and a small ANN, three and two runs, saved into one run directory."""
    run = Run(tmp_path_factory.mktemp("runs"), "compare", overwrite=True)
    linreg = repeat_training(csv, prep, LinearRegressionModel(), seeds=3, run=run, name="linreg")
    ann = repeat_training(csv, prep, ClassicalANNModel(hidden_layer_size=[4], epochs=1), seeds=[0, 1], run=run,
                          name="ann", plot=True)
    return run, linreg + ann


def test_repeated_runs_are_saved(saved, prep):
    run, _ = saved
    for seed in range(3):
        folder = run.path / "linreg" / f"seed_{seed}"
        assert {p.name for p in folder.iterdir()} == {
            "linreg_preprocessing.json", "linreg_model.json", "linreg_training_data.json", "linreg.joblib"
        }
        assert json.loads((folder / "linreg_preprocessing.json").read_text())["random_state"] == seed
    folder = run.path / "ann" / "seed_1"
    assert json.loads((folder / "ann_model.json").read_text())["random_seed"] == 1
    assert (folder / "ann.keras").exists() and (folder / "ann.html").exists()
    # the given objects are not changed
    assert prep.random_state == 7


def test_runs_differ_and_share_splits(saved):
    _, comparison = saved
    assert repr(comparison) == "ModelComparison({'linreg': 3, 'ann': 2})"
    runs = comparison.runs()
    assert list(runs["split_seed"]) == [0, 1, 2, 0, 1]
    scores = comparison.scores()
    linreg = scores[scores["model"] == "linreg"]["RMSE Change(T)"]
    assert linreg.nunique() == 3  # other splits, other errors
    assert {"Horizon RMSE T", "RMSE T @ 0.5 h", "Bias T @ 0.5 h", "R2 Change(T)"} <= set(scores.columns)


def test_loaded_runs_equal_runs_in_memory(saved):
    run, in_memory = saved
    loaded = ModelComparison.load(run.path)
    assert loaded.models == ["ann", "linreg"]  # sorted by directory
    a = in_memory.scores().set_index(["model", "run"]).sort_index()
    b = loaded.scores().set_index(["model", "run"]).sort_index()
    np.testing.assert_allclose(a.to_numpy(dtype=float), b[a.columns].to_numpy(dtype=float))
    assert loaded.runs()["model_class"].tolist().count("LinearRegressionModel") == 3

    # explicit labels and a list of directories
    labelled = ModelComparison.load({"LR": run.path / "linreg"})
    assert labelled.models == ["LR"] and sorted(labelled.runs()["run"]) == ["seed_0", "seed_1", "seed_2"]
    assert ModelComparison.load([run.path / "ann"]).models == ["ann"]
    with pytest.raises(FileNotFoundError):
        ModelComparison.load(run.path / "missing")


def test_summary(saved):
    _, comparison = saved
    summary = comparison.summary()
    row = summary.loc[("RMSE Change(T)", "linreg")]
    values = comparison.scores().query("model == 'linreg'")["RMSE Change(T)"]
    assert row["runs"] == 3
    assert row["mean"] == pytest.approx(values.mean()) and row["std"] == pytest.approx(values.std())
    assert summary.loc[("Horizon RMSE T", "ann"), "runs"] == 2


def test_plot_and_save(saved, tmp_path):
    _, comparison = saved
    fig = comparison.plot()
    boxes = [t for t in fig.data if t.type == "box"]
    bands = [t for t in fig.data if t.type == "scatter" and t.fill == "toself"]
    assert {t.name for t in boxes} == {"linreg", "ann"}
    assert len(bands) == 4  # rmse and bias band for both models
    assert sum(t.showlegend is not False for t in fig.data if t.type != "table") == 2  # one entry per model
    assert not [t for t in comparison.plot(band=None).data if t.type == "scatter" and t.fill == "toself"]
    with pytest.raises(ValueError):
        comparison.plot(band="sigma")

    files = comparison.save(Run(tmp_path, "cmp"))
    assert all(p.exists() for p in files.values())
    assert files["plot"].name == "comparison.html"


def test_single_step_models_and_rnn_on_a_common_horizon(csv):
    """An RNN evaluated over 8 steps and a single step model over 6 steps are compared over 6 steps."""
    single = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", eval_horizon=6)
    multi = PreprocessingMultiStep(inputs=["mDot", "load", "T_in", "T"], output=px.Change("T"), label_width=4,
                                   warmup_width=2, eval_horizon=8)
    comparison = (repeat_training(csv, single, LinearRegressionModel(), seeds=2, name="linreg")
                  + repeat_training(csv, multi, RNNModel(rnn_units=2, epochs=1, early_stopping_epochs=None), seeds=2,
                                    name="rnn"))
    assert comparison.common_horizon() == ("hours", 0.5)
    scores = comparison.scores()
    assert scores["RMSE T @ 0.5 h"].notna().all()
    assert scores.query("model == 'rnn'")["RMSE T"].notna().all()  # trajectory of the RNN
    assert "RMSE T @ 0.25 h" in comparison.scores(horizon=0.25).columns
    comparison.plot()


def test_runs_of_physxai_below_1(tmp_path):
    """Runs saved by the Logger of physXAI < 1.0 (flat metrics, no horizon)."""
    for i, rmse in enumerate([0.1, 0.2]):
        folder = tmp_path / "old" / f"run{i}"
        folder.mkdir(parents=True)
        metrics = {"train_kpis": {"MSE Train": rmse**2, "RMSE Train": rmse, "R2 Train": 0.9},
                   "val_kpis": None, "test_kpis": {"MSE Test": rmse**2, "RMSE Test": rmse, "R2 Test": 0.8}}
        (folder / "training_data.json").write_text(json.dumps({"metrics": metrics, "training_time": 1.0}))
        (folder / "model_config.json").write_text(json.dumps({"__class_name__": "LinearRegressionModel"}))
    comparison = ModelComparison.load(tmp_path)
    assert comparison.models == ["old"]
    assert comparison.scores()["RMSE output"].tolist() == [0.1, 0.2]
    assert comparison.common_horizon() is None
    assert comparison.runs()["model_class"].tolist() == ["LinearRegressionModel"] * 2
    comparison.plot()


def test_invalid_vary(csv, prep):
    with pytest.raises(ValueError):
        repeat_training(csv, prep, LinearRegressionModel(), seeds=1, vary=("weather",))
