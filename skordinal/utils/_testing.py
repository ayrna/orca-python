"""Testing utilities for the skordinal test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TEST_RANDOM_STATE = 0

_BALANCE_SCALE_TEST_IDX_FILE = Path(__file__).parent / "_balance_scale_test_idx.csv"


def _make_balance_scale_split_pinned() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return a pinned train/test split of the Balance Scale dataset.

    Test indices are frozen in ``_balance_scale_test_idx.csv`` (one integer
    per line) so golden-file tests remain stable across scikit-learn updates.

    Returns
    -------
    X_train : ndarray of shape (437, 4)
        Training features.

    X_test : ndarray of shape (188, 4)
        Test features.

    y_train : ndarray of shape (437,)
        Training targets (0-indexed).

    y_test : ndarray of shape (188,)
        Test targets (0-indexed).

    Examples
    --------
    >>> from skordinal.utils._testing import _make_balance_scale_split_pinned
    >>> X_train, X_test, y_train, y_test = _make_balance_scale_split_pinned()
    >>> X_train.shape, X_test.shape
    ((437, 4), (188, 4))
    """
    from skordinal.datasets import load_balance_scale

    bunch = load_balance_scale()
    test_idx = np.loadtxt(_BALANCE_SCALE_TEST_IDX_FILE, dtype=np.int64)
    mask = np.zeros(bunch.target.shape[0], dtype=bool)
    mask[test_idx] = True
    return (
        bunch.data[~mask],
        bunch.data[mask],
        bunch.target[~mask],
        bunch.target[mask],
    )
