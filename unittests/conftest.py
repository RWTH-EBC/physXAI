"""Shared pytest fixtures for the physXAI test-suite."""

import os
from unittest.mock import patch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def disable_plotly_show():
    """Never open a browser window from tests."""
    with patch("plotly.graph_objects.Figure.show"):
        yield


def simulate_one_room(n_steps: int = 576, dt: float = 300.0, seed: int = 0) -> pd.DataFrame:
    """
    Simulates the one room system of the AgentLib-MPC example with known dynamics.

        T[k+1] = T[k] + dt / C * (mDot * cp * (T_in - T[k]) + load)

    Args:
        n_steps: Number of time steps.
        dt: Sampling time in seconds.
        seed: Seed of the random excitation.

    Returns:
        DataFrame indexed by time in seconds with columns mDot, load, T_in, T.
    """
    rng = np.random.default_rng(seed)
    cp, capacity = 1000.0, 100000.0
    m_dot = rng.uniform(0.0, 0.05, n_steps)
    load = rng.uniform(0.0, 300.0, n_steps)
    t_in = 290.15 + rng.normal(0.0, 0.5, n_steps)
    temperature = np.empty(n_steps)
    temperature[0] = 293.15
    for k in range(n_steps - 1):
        temperature[k + 1] = temperature[k] + dt / capacity * (m_dot[k] * cp * (t_in[k] - temperature[k]) + load[k])
    index = pd.Index(np.arange(n_steps) * dt, name="time")
    return pd.DataFrame({"mDot": m_dot, "load": load, "T_in": t_in, "T": temperature}, index=index)


@pytest.fixture
def one_room_csv(tmp_path) -> str:
    """Path to a comma separated csv file of the simulated one room system."""
    path = tmp_path / "one_room.csv"
    simulate_one_room().to_csv(path, sep=",")
    return str(path)
