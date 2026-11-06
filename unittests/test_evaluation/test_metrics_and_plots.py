"""Metrics per output, plots of the model report and saving of the plots."""

import json
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from conftest import simulate_one_room

from physXAI import expr as px
from physXAI.evaluation.metrics import Metrics, regression_kpis
from physXAI.models import LinearRegressionModel
from physXAI.models.ann.ann_design import RNNModel
from physXAI.plotting.plotting import binned_median, model_report, plot_horizon_metrics, plot_prediction_correlation
from physXAI.preprocessing.preprocessing import PreprocessingMultiStep, PreprocessingSingleStep


@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "one_room.csv"
    simulate_one_room(n_steps=1500).to_csv(path, sep=";")
    return str(path)


@pytest.fixture(scope="module")
def linreg(csv):
    prep = PreprocessingSingleStep(
        inputs=["mDot", "load", "T_in", "T"], output="Change(T)", eval_horizon=6, random_state=0
    )
    td = prep.pipeline(csv)
    model = LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    return td, model


def test_regression_kpis():
    kpis = regression_kpis(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0]))
    assert kpis["bias"] == pytest.approx(1.0)
    assert kpis["mae"] == pytest.approx(1.0)
    assert kpis["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert kpis["n"] == 3


def test_metrics_per_output(linreg):
    td, _ = linreg
    metrics = td.metrics
    assert list(metrics.by_output) == ["Change(T)"]
    test = metrics.by_output["Change(T)"]["test"]
    assert test["bias"] == pytest.approx(float(np.mean(td.y_test_pred - td.y_test)), abs=1e-9)
    # legacy keys
    assert metrics.test_kpis["RMSE Test"] == pytest.approx(test["rmse"])
    config = json.loads(json.dumps(td.get_config()))
    assert set(config["metrics"]["by_output"]["Change(T)"]) == {"train", "val", "test"}
    assert len(config["horizon_metrics"]["test"]["std"]) == 6


def test_metrics_of_several_outputs(csv):
    """Several outputs are evaluated separately, also over the horizon of a recurrent model."""
    prep = PreprocessingMultiStep(
        inputs=["mDot", "load"], output=[px.Change("T"), "T_in"], label_width=4, warmup_width=2, random_state=0
    )
    td = prep.pipeline(csv)
    RNNModel(rnn_units=4, epochs=1, early_stopping_epochs=None).pipeline(td, plot=False, save_model=False)
    assert list(td.metrics.by_output) == ["T", "T_in"]
    t, t_in = td.metrics.by_output["T"]["test"], td.metrics.by_output["T_in"]["test"]
    assert t["rmse"] != pytest.approx(t_in["rmse"])
    assert td.metrics.test_kpis["MSE Test"] == pytest.approx((t["mse"] + t_in["mse"]) / 2)
    assert set(td.horizon_metrics.by_feature) == {"T", "T_in"}
    titles = [a.text for a in model_report(td, "RNN").layout.annotations]
    assert "Prediction Correlation – T" in titles and "Prediction Correlation – T_in" in titles
    assert "Prediction Error over the Horizon – T_in" in titles


def test_binned_median():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, 5000)
    prediction = y + rng.normal(0, 1, 5000)
    x, median, low, high = binned_median(y, prediction, bins=20)
    assert len(x) == 20
    np.testing.assert_allclose(median, x, atol=0.3)
    assert np.all(low < median) and np.all(median < high)
    # quartiles of a normal distribution: +- 0.674 sigma
    np.testing.assert_allclose(high - low, 1.35, atol=0.25)


def test_prediction_correlation_style(linreg):
    td, _ = linreg
    fig = plot_prediction_correlation(td)
    points = [t for t in fig.data if t.mode == "markers"]
    medians = [t for t in fig.data if t.name and t.name.endswith("median")]
    bands = [t for t in fig.data if t.fill == "toself"]
    assert {t.name for t in points} == {"Train", "Val", "Test"}
    assert all(t.marker.size == 3 and t.marker.opacity == 0.3 for t in points)
    assert len(medians) == 3 and all(t.line.width == 4 for t in medians)
    assert len(bands) == 3
    assert not [t for t in plot_prediction_correlation(td, iqr=False).data if t.fill == "toself"]


def test_horizon_plot_has_band(linreg):
    td, _ = linreg
    fig = plot_horizon_metrics(td)
    assert any(t.fill == "toself" for t in fig.data)
    assert fig.layout.xaxis.title.text == "Prediction horizon [h]"


def test_pipeline_saves_but_does_not_show_plots(linreg, tmp_path):
    td, _ = linreg
    m = LinearRegressionModel()
    with patch.object(go.Figure, "show") as show:
        m.pipeline(td, save_path=str(tmp_path / "linreg.joblib"))
        assert not show.called
        assert (tmp_path / "linreg.html").exists() and (tmp_path / "linreg.joblib").exists()

        m.pipeline(td, save_model=False, show_plot=True)
        assert show.called

    # without a path nothing is written
    assert m.save_plot(m.plot(td)) is None
    assert m.trained_model is not None


def test_metrics_without_validation(csv):
    td = PreprocessingSingleStep(inputs=["mDot", "T"], output="Change(T)", val_size=0).pipeline(csv)
    LinearRegressionModel().pipeline(td, plot=False, save_model=False)
    assert td.metrics.val_kpis is None
    assert "val" not in td.metrics.by_output["Change(T)"]
    assert isinstance(Metrics(td).get_metrics()[0], list)
    model_report(td, "no validation")


def test_large_plots_are_thinned_out():
    from physXAI.plotting.plotting import MAX_POINTS, _thin

    shown = _thin(100_000)
    assert len(shown) == MAX_POINTS and shown[0] == 0 and shown[-1] == 99_999
    assert len(_thin(10)) == 10
