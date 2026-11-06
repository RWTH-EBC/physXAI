"""Feature selection by backward elimination."""

import json

import pytest
from conftest import simulate_one_room

from physXAI.feature_selection.recursive_feature_elimination import recursive_feature_elimination_pipeline
from physXAI.feature_selection.selection import FeatureSelection
from physXAI.models import LinearRegressionModel
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep

INPUTS = ["mDot", "load", "load_lag1", "T_in", "T", "T_lag1"]


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


def _prep(**kwargs):
    return PreprocessingSingleStep(inputs=INPUTS, output="Change(T)", random_state=0, **kwargs)


def test_removable_respects_lags_and_fixed_inputs():
    fs = FeatureSelection(_prep(eval_horizon=6), LinearRegressionModel(), controls={"mDot": -1})
    # the highest lag first, the recursive feature and the controls never
    assert fs.removable(INPUTS) == ["load_lag1", "T_in", "T_lag1"]
    assert FeatureSelection(_prep(), LinearRegressionModel(), ascending_lag_order=False).removable(INPUTS) == INPUTS


def test_selection_on_the_horizon_error(csv):
    prep = _prep(eval_horizon=6)
    fs = FeatureSelection(prep, LinearRegressionModel(), controls={"load": 1}, n_repeats=2, mpc_horizon=3)
    result = fs.run(csv)
    assert prep.inputs == INPUTS  # the preprocessing is not changed
    assert result.criterion == "horizon"
    assert {"load", "T"} <= set(result.selected.inputs)
    full = result.steps[len(INPUTS)][0]
    assert full.horizon_rmse is not None and full.horizon_bias is not None
    assert full.score_std >= 0
    # the heat load warms the room: the sensitivity has the expected sign everywhere
    assert full.sensitivity["load"] == pytest.approx(1.0)
    assert full.plausible
    assert len(result.table()) == sum(len(c) for c in result.steps.values())
    json.dumps(result.get_config())
    result.plot()


def test_implausible_sensitivity_is_flagged(csv):
    fs = FeatureSelection(_prep(), LinearRegressionModel(), controls={"load": -1}, ascending_lag_order=False)
    candidate = fs.evaluate(csv, INPUTS)
    assert candidate.sensitivity["load"] == pytest.approx(0.0)
    assert not candidate.plausible


def test_tolerance_prefers_fewer_inputs(csv):
    exact = FeatureSelection(_prep(), LinearRegressionModel()).run(csv)
    tolerant = FeatureSelection(_prep(), LinearRegressionModel(), tolerance=10.0).run(csv)
    assert len(tolerant.selected.inputs) <= len(exact.selected.inputs)
    assert len(tolerant.selected.inputs) == 1


def test_invalid_settings():
    with pytest.raises(ValueError, match="eval_horizon"):
        FeatureSelection(_prep(), LinearRegressionModel(), criterion="horizon")
    with pytest.raises(ValueError, match="validation"):
        FeatureSelection(_prep(val_size=0), LinearRegressionModel())
    with pytest.raises(ValueError, match="controls"):
        FeatureSelection(_prep(), LinearRegressionModel(), controls={"unknown": 1})
    with pytest.raises(NotImplementedError):
        FeatureSelection(PreprocessingMultiStep(inputs=["mDot"], output="T", label_width=2), LinearRegressionModel())


def test_legacy_interface(csv):
    runs, inputs = recursive_feature_elimination_pipeline(csv, _prep(eval_horizon=4), LinearRegressionModel())
    # down to the recursive feature, which is never removed
    assert sorted(runs) == list(range(1, len(INPUTS) + 1))
    assert runs[1][0]["inputs"] == ["T"]
    assert "T" in inputs
    assert all({"inputs", "kpi", "kpi_single_step"} <= set(run) for candidates in runs.values() for run in candidates)
