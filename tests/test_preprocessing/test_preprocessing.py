import json

import numpy as np
import pandas as pd
import pytest
from conftest import simulate_one_room

import physXAI
from physXAI.preprocessing.preprocessing import PreprocessingData, PreprocessingMultiStep, PreprocessingSingleStep
from physXAI.preprocessing.training_data import TrainingData


@pytest.fixture
def frame():
    return simulate_one_room(n_steps=2000, dt=300)


def _csv(df, tmp_path, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, sep=";")
    return str(path)


def test_lags_and_change_are_derived(frame, tmp_path):
    prep = PreprocessingSingleStep(inputs=["mDot", "T"], lags={"T": 2}, output="Change(T)", random_state=0)
    td = prep.pipeline(_csv(frame, tmp_path))
    assert prep.inputs == ["mDot", "T", "T_lag1", "T_lag2"]

    rows = td.rows("train")
    T = frame["T"].to_numpy()
    X, y = td.X("train"), td.y("train")
    np.testing.assert_allclose(X[:, 2], T[rows - 1], rtol=1e-6)
    np.testing.assert_allclose(X[:, 3], T[rows - 2], rtol=1e-6)
    # the target of row t is the change to the next step
    np.testing.assert_allclose(y[:, 0], T[rows + 1] - T[rows], rtol=1e-4, atol=1e-6)


def test_lag_names_in_inputs_are_derived(frame, tmp_path):
    td = PreprocessingSingleStep(inputs=["mDot", "mDot_lag1", "T"], output="T").pipeline(_csv(frame, tmp_path))
    rows = td.rows("test")
    np.testing.assert_allclose(td.X("test")[:, 1], frame["mDot"].to_numpy()[rows - 1], rtol=1e-6)


def test_no_pairs_across_missing_time_stamps(frame, tmp_path):
    """Regression: with ignore_nan the target must never be taken from the other side of a gap."""
    gappy = frame.drop(frame.index[1000:1010])
    path = _csv(gappy, tmp_path)
    with pytest.raises(ValueError, match="ignore_nan"):
        PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)").pipeline(path)

    prep = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)", ignore_nan=True, random_state=0)
    td = prep.pipeline(path)
    assert len(td.data) == len(frame)  # missing time stamps become invalid rows
    T = frame["T"].to_numpy()
    for split in ("train", "val", "test"):
        rows = td.rows(split)
        assert not np.any((rows >= 999) & (rows < 1010))
        np.testing.assert_allclose(td.y(split)[:, 0], T[rows + 1] - T[rows], rtol=1e-4, atol=1e-6)


def test_time_step_subsampling_and_datetime_index(frame, tmp_path):
    df = frame.copy()
    df.index = pd.Timestamp("2025-01-01") + pd.to_timedelta(df.index, unit="s")
    df.index.name = "time"
    prep = PreprocessingSingleStep(inputs=["mDot", "T"], output="T", time_step=900)
    td = prep.pipeline(_csv(df, tmp_path))
    assert len(td.data) == int(np.ceil(len(frame) / 3))
    assert np.issubdtype(td.time.dtype, np.datetime64)
    with pytest.raises(ValueError, match="multiple"):
        PreprocessingSingleStep(inputs=["mDot"], output="T", time_step=400).pipeline(_csv(df, tmp_path, "b.csv"))


def test_transform_hook(frame, tmp_path):
    def heat_flow(df):
        df["Q"] = df["mDot"] * (df["T_in"] - df["T"])
        return df

    td = PreprocessingSingleStep(inputs=["Q", "T"], output="Change(T)", transform=heat_flow).pipeline(
        _csv(frame, tmp_path)
    )
    rows = td.rows("train")
    expected = (frame["mDot"] * (frame["T_in"] - frame["T"])).to_numpy()[rows]
    np.testing.assert_allclose(td.X("train")[:, 0], expected, rtol=1e-5)


def test_config_roundtrip_reproduces_the_split(frame, tmp_path):
    path = _csv(frame, tmp_path)
    prep = PreprocessingSingleStep(
        inputs=["mDot", "T"], lags={"T": 1}, output="Change(T)", eval_horizon="2h", gap_steps=3
    )
    td = prep.pipeline(path)
    config = json.loads(json.dumps(prep.get_config()))
    assert config["inputs"] == ["mDot", "T", "T_lag1"]
    for key in ("__class_name__", "inputs", "output", "shift", "time_step", "test_size", "val_size", "random_state"):
        assert key in config  # AgentLib-MPC contract

    restored = PreprocessingData.from_config(config)
    assert isinstance(restored, PreprocessingSingleStep)
    td2 = restored.pipeline(path)
    assert restored.inputs == prep.inputs
    np.testing.assert_array_equal(td.labels, td2.labels)
    assert td.eval_horizon == td2.eval_horizon == 24


def test_global_seed():
    physXAI.set_seed(5)
    try:
        assert PreprocessingSingleStep(inputs=["a"], output="b").random_state == 5
        assert PreprocessingSingleStep(inputs=["a"], output="b", random_state=7).random_state == 7
    finally:
        physXAI.set_seed(None)
    seeds = {PreprocessingSingleStep(inputs=["a"], output="b").random_state for _ in range(3)}
    assert len(seeds) > 1  # without a seed, every object draws (and stores) a new one


def test_training_data_save_and_load(frame, tmp_path):
    td = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)").pipeline(_csv(frame, tmp_path))
    td.save(tmp_path / "td")
    loaded = TrainingData.load(tmp_path / "td")
    np.testing.assert_array_equal(loaded.X("test"), td.X("test"))
    np.testing.assert_array_equal(loaded.y("val"), td.y("val"))
    assert loaded.inputs == td.inputs
    assert loaded.outputs == td.outputs


def test_multi_step_sequences_do_not_leak(frame, tmp_path):
    prep = PreprocessingMultiStep(inputs=["mDot", "load"], output="T", label_width=6, warmup_width=4, random_state=0)
    td = prep.pipeline(_csv(frame, tmp_path))
    # one sequence of warmup and prediction, starting from a zero state
    assert td.X_train.shape[1:] == (10, 2)
    assert td.y_train.shape[1:] == (6, 1)

    # 'load' is random, so its values identify the rows of a sequence
    rows = {split: set(np.round(td.X(split)[:, :, 1].ravel(), 6).tolist()) for split in ("train", "val", "test")}
    assert not rows["train"] & rows["test"]
    assert not rows["train"] & rows["val"]

    # label k is T at the step after the k-th predicted input
    T = frame["T"].to_numpy()
    first = int(np.argmin(np.abs(frame["load"].to_numpy() - td.X_train[0, 4, 1])))
    np.testing.assert_allclose(td.y_train[0, :, 0], T[first + 1 : first + 7], rtol=1e-5)


def test_multi_step_trajectory_of_changes(frame, tmp_path):
    prep = PreprocessingMultiStep(inputs=["mDot", "load"], output="Change(T)", label_width=5, warmup_width=3)
    td = prep.pipeline(_csv(frame, tmp_path))
    assert td.absolute and td.difference == [True]
    T = frame["T"].to_numpy()
    first = int(np.argmin(np.abs(frame["load"].to_numpy() - td.X_train[0, 3, 1])))
    # the targets are the changes, the trajectory starts at the measured value T(t0)
    np.testing.assert_allclose(td.targets("train")[0, :, 0], np.diff(T[first : first + 6]), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(td.initial("train")[0], T[first], rtol=1e-6)
    np.testing.assert_allclose(td.y_train[0, :, 0], T[first + 1 : first + 6], rtol=1e-5)


def test_multi_step_without_validation(frame, tmp_path):
    td = PreprocessingMultiStep(
        inputs=["mDot", "T"], output="Change(T)", label_width=3, warmup_width=0, val_size=0
    ).pipeline(_csv(frame, tmp_path))
    assert td.X_val is None
    assert td.X_train.shape[1:] == (3, 2)


def test_unknown_arguments_are_rejected():
    """Arguments of physXAI < 1.0 (e.g. init_features, batch_size) are not silently ignored."""
    with pytest.raises(TypeError):
        PreprocessingMultiStep(inputs=["mDot"], output="T", label_width=3, init_features=["T"])
    with pytest.raises(TypeError):
        PreprocessingSingleStep(inputs=["mDot"], output="T", batch_size=4)
    prep = PreprocessingMultiStep(inputs=["mDot"], output="T", label_width=3)
    assert PreprocessingData.from_config(prep.get_config()).label_width == 3
