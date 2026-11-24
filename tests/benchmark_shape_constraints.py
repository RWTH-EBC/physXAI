"""
Benchmark of the constrained least squares fit (`ModularANN(solver='qp', shape=ShapeConstraints(...))`).

Synthetic heat pump data with N samples, 4 terms and 2 monotonicity constraints on raw columns. The time of
`model.pipeline` (graph, fit and evaluation) is measured for every N. Not part of pytest (manual check, see
`tests/test_models/test_shape_constraints.py` for a coarse upper bound in the test suite).

Usage:

    python tests/benchmark_shape_constraints.py                     # N = 1e3, 1e4, 1e5
    python tests/benchmark_shape_constraints.py --sizes 1000 10000
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd


def synthetic_heat_pump(n: int, seed: int = 0) -> pd.DataFrame:
    """
    Electrical power of a heat pump in the next time step: P(t+1) = u * (3000 + 80 * (T - T_amb)) (t) + noise, so
    dP/du > 0 and dP/dT_amb < 0. The model predicts the next step from the inputs of the current step (shift = 1).
    """
    rng = np.random.default_rng(seed)
    u = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.2, 1.0, n))
    t_amb = rng.uniform(263.15, 288.15, n)
    t = rng.uniform(291.15, 297.15, n)
    power = np.empty(n)
    power[0] = 0.0
    power[1:] = (u * (3000.0 + 80.0 * (t - t_amb)))[:-1] + rng.normal(0.0, 50.0, n - 1)
    index = pd.Index(np.arange(n) * 300.0, name="time")
    return pd.DataFrame({"u": u, "T_amb": t_amb, "T": t, "P": power}, index=index)


def build_model():
    from physXAI import expr as px
    from physXAI.models import ModularANN, ShapeConstraints

    v = px.Vars()
    terms = [v.u, v.u * v.T_amb, v.u * v.T, v.T_amb]
    return ModularANN(px.Linear(terms), solver="qp", shape=ShapeConstraints(monotonicities={"u": 1, "T_amb": -1}))


def run(sizes: list[int]) -> list[tuple[int, float]]:
    import physXAI
    from physXAI.preprocessing.preprocessing import PreprocessingSingleStep

    physXAI.set_log_level("warning")
    results = list()
    with tempfile.TemporaryDirectory() as directory:
        for n in sizes:
            path = Path(directory) / f"heat_pump_{n}.csv"
            synthetic_heat_pump(n).to_csv(path, sep=";")
            model = build_model()
            td = PreprocessingSingleStep(output="P", random_state=0).pipeline(path, model=model)
            start = time.perf_counter()
            model.pipeline(td, plot=False)
            seconds = time.perf_counter() - start
            results.append((n, seconds))
            print(f"N = {n:>7d}  {seconds:8.2f} s  (test RMSE {td.metrics.kpi('test', 'rmse'):.1f} W)", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    arguments = parser.parse_args()
    run(arguments.sizes)
