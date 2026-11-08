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
    from physXAI import expr as px

    frame = frame.set_index(pd.Timestamp("2024-03-01") + pd.to_timedelta(frame.index, unit="s"))
    frame.index.name = "time"
    v = px.Vars()
    prep = PreprocessingSingleStep(
        inputs=["mDot", "q", "T"],
        output="Change(T)",
        columns={"q": v.mDot * v.T},
        loss_columns=["y:load"],
        eval_horizon=6,
    )
    td = prep.pipeline(_csv(frame, tmp_path))
    td.save(tmp_path / "td")
    loaded = TrainingData.load(tmp_path / "td")
    np.testing.assert_array_equal(loaded.X("test"), td.X("test"))
    np.testing.assert_array_equal(loaded.y("val"), td.y("val"))
    np.testing.assert_array_equal(loaded.labels, td.labels)
    np.testing.assert_array_equal(loaded.valid, td.valid)
    np.testing.assert_array_equal(loaded.time, td.time)
    assert np.issubdtype(loaded.time.dtype, np.datetime64)
    assert loaded.inputs == td.inputs
    assert loaded.outputs == td.outputs
    assert loaded.loss_columns == ["y:load"] and loaded.eval_horizon == 6 and loaded.shift == 1
    assert list(loaded.derived) == ["q"] and loaded.derived["q"].free_vars() == ["mDot", "T"]


@pytest.mark.parametrize("shift", [2, -1, True])
def test_shift_is_zero_or_one(shift):
    with pytest.raises(ValueError, match="shift"):
        PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)", shift=shift)


def test_recursive_predictions_need_shift_one():
    with pytest.raises(ValueError, match="eval_horizon"):
        PreprocessingSingleStep(inputs=["mDot", "T_lag1"], output="T", shift=0, eval_horizon=4)
    with pytest.raises(ValueError, match="Multi step"):
        PreprocessingMultiStep(inputs=["mDot"], output="T", label_width=3, shift=0)


def test_shift_zero_predicts_the_output_at_the_time_of_the_inputs(frame, tmp_path):
    """An algebraic output of point sampled data, the target is taken at the time of the inputs."""
    from physXAI.io import AgentLibCompatibilityError, check_agentlib_compatibility
    from physXAI.models import LinearRegressionModel
    from physXAI.training import RecursiveTraining

    prep = PreprocessingSingleStep(inputs=["mDot", "T_lag1"], output="T", shift=0, random_state=0)
    td = prep.pipeline(_csv(frame, tmp_path))
    rows = td.rows("train")
    T = frame["T"].to_numpy()
    np.testing.assert_allclose(td.y("train")[:, 0], T[rows], rtol=1e-6)
    np.testing.assert_allclose(td.X("train")[:, 1], T[rows - 1], rtol=1e-6)
    assert td.shift == 0 and prep._default_gap() == 1  # embargo: largest lag + shift
    # no recursive evaluation, no multi step training, no export to AgentLib-MPC
    LinearRegressionModel().pipeline(td, plot=False)
    assert td.horizon_metrics is None
    from physXAI.models import ClassicalANNModel

    with pytest.raises(ValueError, match="shift=1"):
        ClassicalANNModel(epochs=1, recursive_training=RecursiveTraining(horizon=4)).pipeline(td, plot=False)
    with pytest.raises(AgentLibCompatibilityError, match="shift"):
        check_agentlib_compatibility(prep)


@pytest.mark.parametrize(
    "settings",
    [
        dict(inputs=["mDot", "T"], output="T"),  # the target itself is an input
        dict(inputs=["mDot", "Change(T)"], output="T"),  # the change holds the target as well
        dict(inputs=["mDot", "T_lag1"], output="Change(T)"),  # a change is defined to the next step
        dict(inputs=["mDot", "q"], output="T", columns="q"),  # a derived input depends on the target
        dict(inputs=["mDot", "Change(q)"], output="T", columns="q"),  # its change holds q(t) as well
        dict(inputs=["mDot", "q2"], output="T", columns="q"),  # a nested derived column
    ],
)
def test_shift_zero_rejects_inputs_holding_the_target(frame, tmp_path, settings):
    from physXAI import expr as px

    settings = dict(settings)
    if settings.pop("columns", None):
        v = px.Vars()
        settings["columns"] = {"q": v.mDot * v.T, "q2": 2 * v.q}
    with pytest.raises(ValueError, match="shift"):
        PreprocessingSingleStep(shift=0, **settings).pipeline(_csv(frame, tmp_path))


def test_shift_zero_accepts_lags_of_derived_columns(frame, tmp_path):
    """Lags of the target, also inside derived columns, only hold past values."""
    from physXAI import expr as px

    v = px.Vars()
    prep = PreprocessingSingleStep(
        inputs=["mDot", "q_lag1", "Change(T_lag1)"], output="T", shift=0, columns={"q": v.mDot * v.T}, random_state=0
    )
    td = prep.pipeline(_csv(frame, tmp_path))
    assert td.inputs == ["mDot", "q_lag1", "Change(T_lag1)"]


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


def test_sub_sampling_takes_the_values_at_the_time_steps_and_warns(frame, tmp_path, caplog):
    path = _csv(frame, tmp_path)
    with caplog.at_level("WARNING", logger="physXAI"):
        data = PreprocessingSingleStep(inputs=["load", "T"], output="Change(T)", time_step=900).load_data(path)
    assert "sub-sampled" in caplog.text and "one of every 3 values" in caplog.text
    np.testing.assert_allclose(data["load"].to_numpy(), frame["load"].to_numpy()[::3][: len(data)])
    caplog.clear()
    with caplog.at_level("WARNING", logger="physXAI"):
        PreprocessingSingleStep(inputs=["load", "T"], output="Change(T)").load_data(path)
    assert "sub-sampled" not in caplog.text


def test_dropped_time_stamps_are_logged(frame, tmp_path, caplog):
    frame = pd.concat([frame.iloc[:10], frame.iloc[[5]], frame.iloc[10:]])  # a duplicate
    shifted = frame.iloc[[20]].copy()
    shifted.index = shifted.index + 17.0  # a time stamp off the grid of 300 s
    frame = pd.concat([frame, shifted])
    with caplog.at_level("WARNING", logger="physXAI"):
        PreprocessingSingleStep(inputs=["T"], output="Change(T)").load_data(_csv(frame, tmp_path))
    assert "1 duplicate time stamps" in caplog.text
    assert "1 time stamps" in caplog.text and "not on the grid" in caplog.text


def test_closed_form_data_in_float64(frame, tmp_path):
    td = PreprocessingSingleStep(inputs=["mDot", "T"], output="T", random_state=0).pipeline(_csv(frame, tmp_path))
    assert td.X("train").dtype == np.float32
    X = td.X("train", np.float64)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[:, 1], frame["T"].to_numpy()[td.rows("train")])
