"""Cross validation of the number of epochs and the minimum number of epochs of the early stopping."""

import numpy as np
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.models import ModularANN
from physXAI.models.ann.ann_design import RNNModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep
from physXAI.preprocessing.splitting import BlockSplit
from physXAI.training import CrossValidation, RecursiveTraining, cross_validate, fit_cross_validated
from physXAI.training.cross_validation import RECURSIVE, TRAINING, phases


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


def _multi_step():
    return PreprocessingMultiStep(
        inputs=["mDot", "load", "T_in", "T"], output=px.Change("T"), label_width=4, warmup_width=3, random_state=0
    )


def test_rnn_epochs_from_the_mean_curve(csv):
    prep = _multi_step()
    model = RNNModel(rnn_units=4, epochs=12, early_stopping_epochs=2, random_seed=1)
    result = fit_cross_validated(prep, model, csv, CrossValidation(folds=3))
    cv = result.td.cross_validation
    assert set(cv["epochs"]) == {TRAINING} and len(cv["curves"][TRAINING]) == 3
    common = min(len(c) for c in cv["curves"][TRAINING])
    assert cv["epochs"][TRAINING] == int(np.argmin(cv["mean_curves"][TRAINING][:common])) + 1
    # the final model trains for exactly E* epochs on all data outside the test blocks
    assert result.model.epochs == cv["epochs"][TRAINING]
    assert len(result.td.training_record.history["loss"]) == cv["epochs"][TRAINING]
    assert result.preprocessing.split.folds == 3 and result.preprocessing.split.fold is None
    assert not result.td.has_split("val")
    # the given model and preprocessing are not changed
    assert model.epochs == 12 and prep.split.folds is None
    # the test blocks are the ones of a training without cross validation
    plain = prep.pipeline(csv)
    assert result.td.summary()["fingerprint"]["test"] == plain.summary()["fingerprint"]["test"]
    assert "cross_validation" in result.td._results_config()


def test_min_epochs(csv):
    model = RNNModel(rnn_units=4, epochs=10, early_stopping_epochs=1, random_seed=1)
    cv = cross_validate(_multi_step(), model, csv, CrossValidation(folds=2, min_epochs=5))
    assert cv["epochs"][TRAINING] >= 5
    assert all(len(c) >= 6 for c in cv["curves"][TRAINING])


def test_reused_cross_validation(csv, monkeypatch):
    model = RNNModel(rnn_units=4, epochs=12, early_stopping_epochs=2, random_seed=1)
    earlier = {"epochs": {TRAINING: 3}, "settings": CrossValidation(folds=3).get_config()}
    import physXAI.training.cross_validation as module

    def no_folds(*args, **kwargs):
        raise AssertionError("no new cross validation expected")

    monkeypatch.setattr(module, "cross_validate", no_folds)
    result = fit_cross_validated(_multi_step(), model, csv, CrossValidation(folds=3), earlier)
    assert len(result.td.training_record.history["loss"]) == 3
    assert result.td.cross_validation["reused"] and "reused" not in earlier
    with pytest.raises(ValueError, match="phases"):
        fit_cross_validated(_multi_step(), model, csv, CrossValidation(folds=3), {"epochs": {RECURSIVE: 3}})


def test_phases_of_the_model_kinds():
    v = px.Vars()
    linear = px.Linear([v.mDot * (v.T_in - v.T), v.load])
    assert phases(RNNModel()) == [TRAINING]
    assert phases(ModularANN(linear, solver="ols")) == []
    assert phases(ModularANN(linear, solver="ols", recursive_training=RecursiveTraining(horizon=3))) == [RECURSIVE]
    mlp = px.MLP([v.mDot, v.load, v.T_in, v.T], n_neurons=2)
    assert phases(ModularANN(mlp, recursive_training=RecursiveTraining(horizon=3))) == [TRAINING, RECURSIVE]
    assert phases(ModularANN(mlp, recursive_training=RecursiveTraining(horizon=3, mode="multi_step"))) == [RECURSIVE]


def test_recursive_single_step_model(tmp_path):
    path = tmp_path / "one_room.csv"
    simulate_one_room(n_steps=800).to_csv(path, sep=";")
    v = px.Vars()
    model = ModularANN(
        px.MLP([v.mDot, v.load, v.T_in, v.T], n_neurons=3),
        epochs=4,
        early_stopping_epochs=1,
        random_seed=0,
        recursive_training=RecursiveTraining(horizon=3, epochs=3, stride=4),
    )
    prep = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", random_state=0)
    result = fit_cross_validated(prep, model, str(path), CrossValidation(folds=2))
    epochs = result.td.cross_validation["epochs"]
    assert set(epochs) == {TRAINING, RECURSIVE}
    assert result.model.epochs == epochs[TRAINING] and result.model.recursive_training.epochs == epochs[RECURSIVE]
    assert len(result.td.recursive_training_record.history["loss"]) == epochs[RECURSIVE]


def test_needs_a_patience_and_epochs(csv):
    with pytest.raises(ValueError, match="patience"):
        cross_validate(_multi_step(), RNNModel(rnn_units=2, epochs=2, early_stopping_epochs=None), csv)
    v = px.Vars()
    with pytest.raises(ValueError, match="closed form"):
        cross_validate(_multi_step(), ModularANN(px.Linear([v.load]), solver="ols"), csv)


def test_early_stopping_start(csv):
    """Without a minimum the patience of 1 ends the training early, with it the training runs at least that long."""
    prep = _multi_step()
    prep.split = BlockSplit(0.1, 0.1)
    td = prep.pipeline(csv)
    assert td.has_split("val")
    m = RNNModel(rnn_units=4, epochs=15, early_stopping_epochs=1, early_stopping_start=8, random_seed=1)
    m.pipeline(td, plot=False, save_model=False)
    assert len(td.training_record.history["loss"]) >= 9
    assert RNNModel.from_config(m.get_config()).early_stopping_start == 8


def _recursive_model():
    v = px.Vars()
    return ModularANN(
        px.MLP([v.mDot, v.load, v.T_in, v.T], n_neurons=3),
        epochs=4,
        early_stopping_epochs=1,
        random_seed=0,
        recursive_training=RecursiveTraining(horizon=3, epochs=3, stride=4),
    )


def test_fold_without_multi_step_validation(tmp_path, monkeypatch):
    """A fold whose validation data holds no multi step window counts only for the single step phase."""
    import physXAI.training.cross_validation as cross_validation

    path = tmp_path / "one_room.csv"
    simulate_one_room(n_steps=800).to_csv(path, sep=";")
    prep = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", random_state=0)
    calls = []

    def first_fold_without(td, model):
        calls.append(model.recursive_training is not None)
        return len(calls) > 1

    monkeypatch.setattr(cross_validation, "_has_recursive_validation", first_fold_without)
    result = cross_validate(prep, _recursive_model(), str(path), CrossValidation(folds=3))
    assert result["folds_used"] == {TRAINING: [0, 1, 2], RECURSIVE: [1, 2]}
    assert len(result["curves"][TRAINING]) == 3 and len(result["curves"][RECURSIVE]) == 2
    assert set(result["epochs"]) == {TRAINING, RECURSIVE}

    monkeypatch.setattr(cross_validation, "_has_recursive_validation", lambda td, model: False)
    with pytest.raises(RuntimeError, match="No fold holds validation data for the recursive phase"):
        cross_validate(prep, _recursive_model(), str(path), CrossValidation(folds=2))


def test_has_recursive_validation(tmp_path):
    """Short validation blocks hold no window of a long horizon."""
    from physXAI.training.cross_validation import _has_recursive_validation

    path = tmp_path / "one_room.csv"
    simulate_one_room(n_steps=400).to_csv(path, sep=";")
    prep = PreprocessingSingleStep(inputs=["mDot", "load", "T_in", "T"], output="Change(T)", random_state=0)
    prep.split = BlockSplit(test_size=0.2, val_size=0.0, block_length=20, min_blocks=1).with_fold(5, 0)
    model = _recursive_model()
    td = prep.pipeline(str(path), model=model)
    assert _has_recursive_validation(td, model)
    model.recursive_training.horizon = 200
    assert not _has_recursive_validation(td, model)
