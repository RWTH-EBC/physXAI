"""
Leakage free splitting of time series into training, validation and test data.

Rows of a time series are strongly autocorrelated and multi step windows span many rows.
Splitting single rows (or overlapping windows) randomly therefore leaks information from the
training into the test data. `BlockSplit` instead assigns contiguous blocks of the time series
to the splits (blocked holdout, see Bergmeir et al. 2018 and Cerqueira et al. 2020):

- **Stratified:** val and test blocks are drawn from equally sized strata of the time axis, so
  every split covers the whole period (e.g. all seasons), which a purely random draw does not
  guarantee.
- **Embargo:** `gap_steps` rows at every boundary between two splits are not used at all
  (purging, López de Prado 2018). The default (the reach of a sample into the past, i.e. lags, changes and derived
  columns, plus `shift`) guarantees that no value of the data is part of a training and a test sample.
- Windows are later only taken from within one split (see `physXAI.preprocessing.windows`),
  so no window ever crosses a boundary.

The embargo removes shared samples, not the autocorrelation of the process: a test block lies between training
blocks of the same period (weather, season, operation), so the test error measures the interpolation within the
period of the data. It is optimistic for a new period (e.g. another season or year). For that question use
`'holdout_end'` (chronological: train, val, test), a longer `gap_steps` or a separate data set.
`'random_rows'` (random rows, only to compare with, single step only) shows how much a random split overestimates
the performance.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("physXAI")

TRAIN, VAL, TEST, UNUSED = 0, 1, 2, -1
SPLITS = {"train": TRAIN, "val": VAL, "test": TEST}

Strategy = Literal["blocks", "holdout_end", "random_rows"]


class BlockSplit:
    """
    Assigns every row of a time series to the training, validation or test split.

    Args:
        test_size: Fraction of the rows used for testing.
        val_size: Fraction of the rows used for validation.
        strategy: 'blocks' (default), 'holdout_end' or 'random_rows', see the module docstring.
        block_length: Length of a block in time steps (int) or as a pandas time delta string,
                      e.g. '7D'. Defaults to max(4 * window length, 1 day).
        gap_steps: Number of rows left out at each boundary between two splits. Defaults to the
                   value given by the preprocessing (maximum lag + warmup).
        min_blocks: The block length is reduced if the data holds fewer blocks, so that small data
                    sets can still be split.
    """

    def __init__(
        self,
        test_size: float = 0.1,
        val_size: float = 0.1,
        strategy: Strategy = "blocks",
        block_length: int | str | None = None,
        gap_steps: int | None = None,
        min_blocks: int = 20,
    ):
        if not (0 <= test_size < 1 and 0 <= val_size < 1 and test_size + val_size < 1):
            raise ValueError(f"Invalid split sizes: test_size={test_size}, val_size={val_size}.")
        if strategy not in ("blocks", "holdout_end", "random_rows"):
            raise ValueError(f"Unknown split strategy '{strategy}'.")
        if gap_steps is not None and gap_steps < 0:
            raise ValueError("gap_steps must not be negative.")
        self.test_size = test_size
        self.val_size = val_size
        self.strategy: Strategy = strategy
        self.block_length = block_length
        self.gap_steps = gap_steps
        self.min_blocks = min_blocks

    def get_config(self) -> dict:
        return {
            "test_size": self.test_size,
            "val_size": self.val_size,
            "strategy": self.strategy,
            "block_length": self.block_length,
            "gap_steps": self.gap_steps,
            "min_blocks": self.min_blocks,
        }

    @classmethod
    def from_config(cls, config: dict) -> BlockSplit:
        return cls(**config)

    def resolve_block_length(self, n_rows: int, window_length: int = 1, time_step: float | None = None) -> int:
        """Block length in rows used for `n_rows` rows."""
        if isinstance(self.block_length, str):
            if time_step is None:
                raise ValueError("A block length given as time delta requires the time step of the data.")
            length = int(round(pd.Timedelta(self.block_length).total_seconds() / time_step))
        elif self.block_length is not None:
            length = int(self.block_length)
        else:
            day = int(round(86400 / time_step)) if time_step else 1
            length = max(4 * window_length, day)
        length = max(length, window_length, 1)

        if n_rows // length < self.min_blocks:
            reduced = max(window_length, n_rows // self.min_blocks, 1)
            if reduced < length:
                logger.warning(
                    "Only %d rows: block length reduced from %d to %d rows to get enough blocks for the split.",
                    n_rows,
                    length,
                    reduced,
                )
                length = reduced
        return length

    def assign(
        self,
        n_rows: int,
        seed: int,
        window_length: int = 1,
        time_step: float | None = None,
        default_gap: int = 0,
    ) -> np.ndarray:
        """
        Assigns the rows to the splits.

        Args:
            n_rows: Number of rows of the (regular) time series.
            seed: Seed of the random assignment.
            window_length: Length of the windows that are cut out of each split later.
            time_step: Sampling time in seconds, needed for block lengths given as time delta.
            default_gap: Embargo used if `gap_steps` is None.

        Returns:
            np.ndarray: One label per row: TRAIN (0), VAL (1), TEST (2) or UNUSED (-1, embargo).
        """
        rng = np.random.default_rng(seed)
        if self.strategy == "random_rows":
            return self._random_rows(n_rows, rng)

        if self.strategy == "holdout_end":
            n_test = int(round(n_rows * self.test_size))
            n_val = int(round(n_rows * self.val_size))
            labels = np.full(n_rows, TRAIN, dtype=np.int8)
            labels[n_rows - n_test - n_val : n_rows - n_test] = VAL
            labels[n_rows - n_test :] = TEST
        else:
            labels = self._blocks(n_rows, rng, window_length, time_step)

        gap = self.gap_steps if self.gap_steps is not None else default_gap
        if gap > 0:
            labels = apply_embargo(labels, gap)
        return labels

    def _random_rows(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        labels = np.full(n_rows, TRAIN, dtype=np.int8)
        order = rng.permutation(n_rows)
        n_test = int(round(n_rows * self.test_size))
        n_val = int(round(n_rows * self.val_size))
        labels[order[:n_test]] = TEST
        labels[order[n_test : n_test + n_val]] = VAL
        return labels

    def _blocks(self, n_rows: int, rng: np.random.Generator, window_length: int, time_step: float | None):
        length = self.resolve_block_length(n_rows, window_length, time_step)
        n_blocks = max(n_rows // length, 1)
        n_test = _n_blocks(n_blocks, self.test_size)
        n_val = _n_blocks(n_blocks - n_test, self.val_size * n_blocks / max(n_blocks - n_test, 1))
        if n_test + n_val >= n_blocks:
            logger.warning("Too few blocks (%d) for a split, all data is used for training.", n_blocks)
            return np.full(n_rows, TRAIN, dtype=np.int8)

        block_labels = np.full(n_blocks, TRAIN, dtype=np.int8)
        test_blocks = _stratified_choice(np.arange(n_blocks), n_test, rng)
        block_labels[test_blocks] = TEST
        val_blocks = _stratified_choice(np.flatnonzero(block_labels == TRAIN), n_val, rng)
        block_labels[val_blocks] = VAL

        # the last block absorbs the remainder of the rows
        block_of_row = np.minimum(np.arange(n_rows) // length, n_blocks - 1)
        return block_labels[block_of_row]


def _n_blocks(n_blocks: int, size: float) -> int:
    """Number of blocks for a split of `size`, at least one if the size is not zero."""
    if size <= 0:
        return 0
    return max(1, int(round(n_blocks * size)))


def _stratified_choice(candidates: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Draws `k` candidates, one out of each of `k` consecutive, equally sized strata."""
    if k <= 0:
        return np.array([], dtype=int)
    strata = np.array_split(candidates, k)
    return np.array([rng.choice(stratum) for stratum in strata if len(stratum) > 0], dtype=int)


def apply_embargo(labels: np.ndarray, gap: int) -> np.ndarray:
    """
    Marks `gap` rows at every boundary between two splits as UNUSED.

    The rows are taken from the training side of a boundary, so validation and test data stay
    complete. At a boundary between validation and test data, the validation side is used.
    Only rows of the block at the boundary are purged: a block shorter than `gap` is purged completely,
    but the rows of the neighbouring block are kept.
    """
    labels = labels.copy()
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    original = labels.copy()
    n = len(labels)
    for b in boundaries:
        before, after = original[b - 1], original[b]
        if UNUSED in (before, after):
            continue
        purge_before = before == TRAIN or (before == VAL and after == TEST)
        if purge_before:
            start = b
            while start > max(b - gap, 0) and original[start - 1] == before:
                start -= 1
            labels[start:b] = UNUSED
        else:
            stop = b
            while stop < min(b + gap, n) and original[stop] == after:
                stop += 1
            labels[b:stop] = UNUSED
    return labels


def split_summary(labels: np.ndarray) -> dict[str, float]:
    """Fractions of rows per split, e.g. for logging."""
    n = max(len(labels), 1)
    return {name: float(np.sum(labels == value)) / n for name, value in {**SPLITS, "unused": UNUSED}.items()}
