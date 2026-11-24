"""
Cutting windows out of a split time series.

A window of length L starting at row s covers the rows s, ..., s + L - 1. It is only taken if
all of these rows are valid (no missing values) and belong to the same split, so windows never
cross a split boundary or a gap in the data.
"""

from __future__ import annotations

import numpy as np


def window_starts(labels: np.ndarray, valid: np.ndarray, split: int, length: int, stride: int = 1) -> np.ndarray:
    """
    Start rows of all windows of one split.

    Args:
        labels: Split label per row (see `physXAI.preprocessing.splitting`).
        valid: Whether a row holds no missing values.
        split: The split to take the windows from.
        length: Length of a window in rows.
        stride: Distance between the starts of two windows within a contiguous segment.

    Returns:
        np.ndarray: Sorted start rows.
    """
    if length < 1 or stride < 1:
        raise ValueError("Window length and stride have to be at least 1.")
    ok = (labels == split) & valid
    n = len(ok) - length + 1
    if n <= 0:
        return np.array([], dtype=np.int64)
    counts = np.concatenate([[0], np.cumsum(ok, dtype=np.int64)])
    starts = np.flatnonzero(counts[length : length + n] - counts[:n] == length)
    if stride > 1 and len(starts) > 0:
        # count the offset within each run of consecutive starts
        new_run = np.concatenate([[True], np.diff(starts) != 1])
        run_first = starts[np.maximum.accumulate(np.where(new_run, np.arange(len(starts)), 0))]
        starts = starts[(starts - run_first) % stride == 0]
    return starts.astype(np.int64)


def gather_windows(data: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """
    Stacks the windows into an array.

    Args:
        data: Array of shape (rows, features).
        starts: Start rows of the windows.
        length: Length of the windows.

    Returns:
        np.ndarray: Array of shape (windows, length, features).
    """
    if len(starts) == 0:
        return np.empty((0, length, data.shape[1]), dtype=data.dtype)
    return data[starts[:, None] + np.arange(length)[None, :]]
