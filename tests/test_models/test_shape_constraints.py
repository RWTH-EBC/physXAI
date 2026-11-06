"""Constrained least squares with shape constraints (`ModularANN(solver='qp', shape=ShapeConstraints(...))`)."""

import json
import logging
import time

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from scipy.optimize import nnls

from physXAI import expr as px
from physXAI.io import Run
from physXAI.models import AbstractModel, ModularANN, ShapeConstraints
from physXAI.models.ann.pinn.physics import PhysicsResidual
from physXAI.models.modular.shape_constraints import fit_shape_constrained
from physXAI.preprocessing.preprocessing import PreprocessingSingleStep
from physXAI.training import RecursiveTraining

v = px.Vars()
TERMS = [v.u, v.u * v.T_amb, v.u * v.T, v.T_amb]  # derivatives with respect to u and T_amb are multilinear


def heat_pump(n=2000, seed=0, conflict=0.0):
    """P = u * (3000 + 80 * (T - T_amb)) + conflict * T_amb + noise: dP/du > 0, dP/dT_amb < 0 unless conflict > 0."""
    rng = np.random.default_rng(seed)
    u = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.2, 1.0, n))
    t_amb, t = rng.uniform(263.15, 288.15, n), rng.uniform(291.15, 297.15, n)
    power = u * (3000.0 + 80.0 * (t - t_amb)) + conflict * t_amb + rng.normal(0.0, 50.0, n)
    return {"u": u, "T_amb": t_amb, "T": t}, power


def prediction(terms, result, values):
    n = len(next(iter(values.values())))
    return result.bias + sum(w * np.broadcast_to(t.evaluate(values), (n,)) for w, t in zip(result.weights, terms, strict=True))


def box_points(result, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    return {c: rng.uniform(low, high, n) for c, (low, high) in result.domain["box"].items()}


def derivative(terms, result, values, column, h=1e-3):
    shifted = {**values, column: values[column] + h}
    return (prediction(terms, result, shifted) - prediction(terms, result, values)) / h


# --- core -------------------------------------------------------------------------------------------------------
def test_inactive_constraints_give_least_squares():
    values, y = heat_pump()  # dP/du = 3000 + 80 * (T - T_amb) > 0 with a wide margin
    result = fit_shape_constrained(TERMS, values, y, shape=ShapeConstraints(monotonicities={"u": 1}))
    features = np.column_stack([t.evaluate(values) for t in TERMS] + [np.ones(len(y))])
    ols = np.linalg.lstsq(features, y, rcond=None)[0]
    np.testing.assert_allclose(prediction(TERMS, result, values), features @ ols, rtol=1e-6, atol=1e-3)
    assert result.rmse_train == pytest.approx(result.rmse_train_unconstrained, rel=1e-6)
    assert result.constraints[0]["certificate"] == "exact"
    assert result.constraints[0]["active_share_data"] == 0.0


def test_sign_bounds_give_nonnegative_least_squares():
    values, y = heat_pump()
    terms = [v.u, v.T_amb, v.T]
    result = fit_shape_constrained(terms, values, y - 3000.0, bias=False, weight_signs=[1, 1, 1])
    features = np.column_stack([t.evaluate(values) for t in terms])
    expected = nnls(features, y - 3000.0)[0]
    np.testing.assert_allclose(result.weights, expected, rtol=1e-5, atol=1e-6)
    assert np.all(result.weights >= 0)


def test_conflicting_data_respect_the_constraint_on_the_whole_box():
    values, y = heat_pump(conflict=120.0)  # the data say: the power rises with T_amb at low u
    result = fit_shape_constrained(TERMS, values, y, shape=ShapeConstraints(monotonicities={"u": 1, "T_amb": -1}))
    constraint = {c["column"]: c for c in result.constraints}["T_amb"]
    assert constraint["certificate"] == "exact"
    assert result.rmse_train > result.rmse_train_unconstrained  # the cost of the physics
    points = box_points(result)
    assert derivative(TERMS, result, points, "T_amb").max() <= 1e-6 * 3000
    assert derivative(TERMS, result, points, "u").min() >= -1e-6 * 3000


def test_nonlinear_terms_are_verified():
    values, y = heat_pump(conflict=120.0)
    terms = [v.u, v.u * v.T_amb / (v.T + 15.0), v.T_amb]
    result = fit_shape_constrained(terms, values, y, shape=ShapeConstraints(monotonicities={"T_amb": -1}))
    assert result.constraints[0]["certificate"] == "verified"
    assert derivative(terms, result, box_points(result), "T_amb").max() <= 1e-6 * 3000


def _gap_data():
    """y = u^3 - 1.5 u^2 + 0.6 u is decreasing around u = 0.5, where there is no data."""
    rng = np.random.default_rng(0)
    u = np.concatenate([rng.uniform(0.0, 0.3, 300), rng.uniform(0.7, 1.0, 300)])
    y = u ** 3 - 1.5 * u ** 2 + 0.6 * u + rng.normal(0.0, 1e-3, len(u))
    return [v.u, v.u ** 2, v.u ** 3], {"u": u}, y


def test_exchange_finds_violations_between_the_data():
    terms, values, y = _gap_data()
    result = fit_shape_constrained(terms, values, y, shape=ShapeConstraints(monotonicities={"u": 1}))
    assert result.constraints[0]["certificate"] == "verified"
    assert result.iterations >= 2
    grid = {"u": np.linspace(0.0, 1.0, 2001)}
    assert derivative(terms, result, grid, "u", h=1e-6).min() >= -1e-4


def test_domain_data_only_constrains_the_samples():
    terms, values, y = _gap_data()
    result = fit_shape_constrained(terms, values, y,
                                   shape=ShapeConstraints(monotonicities={"u": 1}, domain="data"))
    assert result.constraints[0]["certificate"] == "data"
    assert derivative(terms, result, values, "u", h=1e-6).min() >= -1e-4


def test_no_convergence_is_reported(caplog):
    terms, values, y = _gap_data()
    with caplog.at_level(logging.WARNING, logger="physXAI"):
        result = fit_shape_constrained(terms, values, y, shape=ShapeConstraints(monotonicities={"u": 1}, max_iter=1))
    assert result.constraints[0]["certificate"] == "violated"
    assert not result.certified
    assert "violated" in caplog.text


def _corner_case():
    """y = x * (1 - 1.2 z): dy/dx > 0 on the data (z in [0.4, 0.6]), but < 0 at the corner z = 1 of the box."""
    rng = np.random.default_rng(0)
    values = {"x": rng.uniform(0.0, 1.0, 400), "z": rng.uniform(0.4, 0.6, 400)}
    y = values["x"] * (1.0 - 1.2 * values["z"]) + rng.normal(0.0, 1e-3, 400)
    shape = ShapeConstraints(monotonicities={"x": 1}, bounds={"z": (0.0, 1.0)}, n_samples=0)
    return [v.x, v.x * v.z], values, y, shape


def test_corners_are_checked_on_the_final_solution(monkeypatch):
    """A solution violating the constraint only at a corner (no candidate) is not certified."""
    from physXAI.models.modular import shape_constraints

    terms, values, y, shape = _corner_case()
    assert fit_shape_constrained(terms, values, y, shape=shape).constraints[0]["certificate"] == "exact"
    original = shape_constraints._LeastSquares.solve
    monkeypatch.setattr(shape_constraints._LeastSquares, "solve",
                        lambda self, rows: original(self, np.zeros((0, rows.shape[1]))))  # ignores the constraints
    result = fit_shape_constrained(terms, values, y, shape=shape)
    assert result.constraints[0]["certificate"] == "violated"


def test_failed_qp_is_at_most_verified(monkeypatch):
    from physXAI.models.modular import shape_constraints

    terms, values, y, shape = _corner_case()
    original = shape_constraints._LeastSquares.solve
    monkeypatch.setattr(shape_constraints._LeastSquares, "solve", lambda self, rows: (original(self, rows)[0], False))
    assert fit_shape_constrained(terms, values, y, shape=shape).constraints[0]["certificate"] == "verified"


def test_bounds_define_the_box_and_singular_points_are_left_out(caplog):
    values, y = heat_pump()
    values = {**values, "on": (values["u"] > 0).astype(float)}
    terms = [v.u / (v.on + 1e-12), v.T_amb]
    shape = ShapeConstraints(monotonicities={"T_amb": -1}, bounds={"T_amb": (250.0, 310.0), "on": (-1.0, 1.0)})
    with caplog.at_level(logging.WARNING, logger="physXAI"):
        result = fit_shape_constrained(terms, values, y, shape=shape)
    assert result.domain["box"]["T_amb"] == [250.0, 310.0]
    assert result.constraints[0]["certificate"] in ("exact", "verified")


def test_trivial_convexity_of_linear_terms():
    values, y = heat_pump()
    result = fit_shape_constrained(TERMS, values, y, shape=ShapeConstraints(convexities={"u": -1}))
    assert result.constraints[0]["certificate"] == "exact"
    assert result.constraints[0]["active_share_data"] is None


@pytest.mark.parametrize("kwargs, match", [
    ({}, "at least one"),
    ({"monotonicities": {"u": 2}}, "monotonicities"),
    ({"convexities": {"u": 0}}, "convexities"),
    ({"monotonicities": {"u": 1}, "domain": "everywhere"}, "domain"),
    ({"monotonicities": {"u": 1}, "bounds": {"u": (1.0, 0.0)}}, "bounds"),
    ({"monotonicities": {"u": 1}, "qp_solver": "unknown"}, "QP solver"),
])
def test_invalid_shape_constraints(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ShapeConstraints(**kwargs)


def test_unknown_column():
    values, y = heat_pump()
    with pytest.raises(ValueError, match="mDot"):
        fit_shape_constrained(TERMS, values, y, shape=ShapeConstraints(monotonicities={"mDot": 1}))


def test_fit_is_fast():
    """Coarse upper bound, see tests/benchmark_shape_constraints.py for the benchmark."""
    values, y = heat_pump(n=10000, conflict=120.0)
    start = time.perf_counter()
    fit_shape_constrained(TERMS, values, y, shape=ShapeConstraints(monotonicities={"u": 1, "T_amb": -1}))
    assert time.perf_counter() - start < 5.0


# --- ModularANN -------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def csv(tmp_path_factory):
    values, y = heat_pump(n=1500, conflict=120.0)
    frame = pd.DataFrame(values)
    frame["P"] = np.concatenate([[0.0], y[:-1]])  # the model predicts the next step
    frame.index = pd.Index(np.arange(len(frame)) * 300.0, name="time")
    path = tmp_path_factory.mktemp("data") / "heat_pump.csv"
    frame.to_csv(path, sep=";")
    return str(path)


def _model(**kwargs):
    shape = kwargs.pop("shape", ShapeConstraints(monotonicities={"u": 1, "T_amb": -1}))
    return ModularANN(px.Linear(TERMS), solver="qp", shape=shape, **kwargs)


def test_modular_ann_qp_end_to_end(csv, tmp_path):
    model = _model()
    prep = PreprocessingSingleStep(output="P", random_state=0)
    td = prep.pipeline(csv, model=model)
    model.pipeline(td, plot=False)
    shape = td.physics["shape_constraints"]
    assert [c["certificate"] for c in shape["constraints"]] == ["exact", "exact"]
    assert "Shape Constraints" in str(model.plot(td).layout.annotations)

    files = Run(tmp_path, "qp", overwrite=True).save("model", prep, model, td)
    saved = keras.saving.load_model(files["model"])
    np.testing.assert_allclose(saved.predict(td.X_test, verbose=0), model.trained_model.predict(td.X_test, verbose=0),
                               rtol=1e-6)
    # the exported keras model itself keeps the constraint on random points of the box (automatic differentiation)
    rng = np.random.default_rng(2)
    box = shape["domain"]["box"]
    x = tf.constant(np.column_stack([rng.uniform(*box[c], 5000) for c in td.inputs]).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = saved(x)
    gradient = tape.gradient(y, x).numpy()
    assert gradient[:, td.inputs.index("T_amb")].max() <= 1e-3
    assert gradient[:, td.inputs.index("u")].min() >= -1e-3


def test_modular_ann_qp_config_roundtrip():
    model = _model(rescale_output=True)
    config = json.loads(json.dumps(model.get_config()))
    loaded = AbstractModel.model_from_config(config)
    assert loaded.solver == "qp"
    assert loaded.shape.get_config() == model.shape.get_config()


def test_sign_bounds_with_qp(csv):
    model = ModularANN(px.Linear([v.u, v.T_amb, v.T], monotonicities={"T_amb": 1}), solver="qp")
    td = PreprocessingSingleStep(output="P", random_state=0).pipeline(csv, model=model)
    trained = model.pipeline(td, plot=False)
    assert trained.layers[-1].get_weights()[0][1, 0] >= 0


def test_modular_ann_qp_requirements():
    with pytest.raises(ValueError, match="solver='qp'"):
        ModularANN(px.Linear(TERMS), shape=ShapeConstraints(monotonicities={"u": 1}))
    with pytest.raises(ValueError, match="recursive_training"):
        _model(recursive_training=RecursiveTraining(horizon=4))
    with pytest.raises(ValueError, match="physics"):
        _model(physics=PhysicsResidual(px.Prediction() - v.u, name="r"))
    with pytest.raises(ValueError, match="solver='qp'"):
        ModularANN(px.Linear([v.u], monotonicities={"u": 1}), solver="ols")
    with pytest.raises(ValueError, match="Linear"):
        ModularANN(px.MLP([v.u]), solver="qp")
    # sign bounds alone are kept by the keras constraint, so the multi step training is allowed
    ModularANN(px.Linear([v.u], monotonicities={"u": 1}), solver="qp", recursive_training=RecursiveTraining(horizon=4))


@pytest.mark.parametrize("solver", ["ols", "qp"])
def test_online_learning_of_closed_form_solvers(csv, tmp_path, solver):
    model = ModularANN(px.Linear(TERMS), solver=solver)
    td = PreprocessingSingleStep(output="P", random_state=0).pipeline(csv, model=model)
    model.pipeline(td, plot=False, save_path=str(tmp_path / "m"))
    with pytest.raises(NotImplementedError, match="closed form"):
        model.online_pipeline(td, str(tmp_path / "m.keras"))
