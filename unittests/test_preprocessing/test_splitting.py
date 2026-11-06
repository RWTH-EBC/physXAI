import numpy as np
import pytest

from physXAI.preprocessing.splitting import TEST, TRAIN, UNUSED, VAL, BlockSplit, apply_embargo
from physXAI.preprocessing.windows import gather_windows, window_starts

N_YEAR = 365 * 24  # hourly data of one year


def test_sizes_and_determinism():
    split = BlockSplit(test_size=0.1, val_size=0.2)
    labels = split.assign(N_YEAR, seed=3, time_step=3600)
    assert np.isclose(np.mean(labels == TEST), 0.1, atol=0.03)
    assert np.isclose(np.mean(labels == VAL), 0.2, atol=0.03)
    np.testing.assert_array_equal(labels, split.assign(N_YEAR, seed=3, time_step=3600))
    assert not np.array_equal(labels, split.assign(N_YEAR, seed=4, time_step=3600))


def test_test_blocks_cover_every_season():
    """Stratification: test and validation data lie in every quarter of the year."""
    labels = BlockSplit(0.1, 0.1, block_length="1D").assign(N_YEAR, seed=0, time_step=3600)
    for quarter in np.array_split(labels, 4):
        assert np.any(quarter == TEST)
        assert np.any(quarter == VAL)


def test_blocks_have_the_given_length():
    labels = BlockSplit(0.1, 0.1, block_length="7D").assign(N_YEAR, seed=0, time_step=3600)
    runs = np.split(labels, np.flatnonzero(np.diff(labels)) + 1)
    test_runs = [len(r) for r in runs if r[0] == TEST]
    assert all(length >= 7 * 24 for length in test_runs)


def test_embargo_separates_splits():
    gap = 5
    labels = BlockSplit(0.2, 0.2, gap_steps=gap).assign(2000, seed=1, time_step=60)
    train = np.flatnonzero(labels == TRAIN)
    others = np.flatnonzero((labels == TEST) | (labels == VAL))
    distance = np.min(np.abs(train[:, None] - others[None, :]))
    assert distance > gap


def test_embargo_keeps_test_data():
    labels = np.array([TRAIN] * 5 + [TEST] * 3 + [TRAIN] * 5)
    purged = apply_embargo(labels, 2)
    np.testing.assert_array_equal(purged == TEST, labels == TEST)
    assert list(purged[3:5]) == [UNUSED, UNUSED]
    assert list(purged[8:10]) == [UNUSED, UNUSED]


def test_holdout_end_is_chronological():
    labels = BlockSplit(0.1, 0.1, strategy="holdout_end").assign(100, seed=0)
    assert list(labels[:80]) == [TRAIN] * 80
    assert list(labels[80:90]) == [VAL] * 10
    assert list(labels[90:]) == [TEST] * 10


def test_random_rows_legacy():
    labels = BlockSplit(0.1, 0.1, strategy="random_rows").assign(1000, seed=0)
    assert np.sum(labels == TEST) == 100
    assert np.sum(labels == VAL) == 100


def test_small_data_is_still_split():
    labels = BlockSplit(0.1, 0.1).assign(60, seed=0, time_step=300)
    assert np.any(labels == TEST)
    assert np.any(labels == VAL)


@pytest.mark.parametrize("config", [dict(test_size=0.6, val_size=0.5), dict(strategy="nope")])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        BlockSplit(**config)


def test_windows_stay_within_one_split():
    labels = BlockSplit(0.2, 0.2, block_length=50).assign(1000, seed=2)
    valid = np.ones(1000, dtype=bool)
    valid[[100, 555]] = False
    seen = dict()
    for split in (TRAIN, VAL, TEST):
        starts = window_starts(labels, valid, split, length=12)
        rows = (starts[:, None] + np.arange(12)).ravel()
        assert np.all(labels[rows] == split)
        assert np.all(valid[rows])
        seen[split] = set(rows.tolist())
    assert not seen[TRAIN] & seen[TEST]
    assert not seen[TRAIN] & seen[VAL]
    assert not seen[VAL] & seen[TEST]


def test_window_stride_and_gather():
    labels = np.zeros(20, dtype=np.int8)
    valid = np.ones(20, dtype=bool)
    valid[10] = False
    starts = window_starts(labels, valid, TRAIN, length=3, stride=2)
    assert list(starts) == [0, 2, 4, 6, 11, 13, 15, 17]
    data = np.arange(40.0).reshape(20, 2)
    windows = gather_windows(data, starts[:2], 3)
    assert windows.shape == (2, 3, 2)
    assert windows[1, 0, 0] == 4.0
