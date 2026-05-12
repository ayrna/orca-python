"""Testing utilities for the skordinal test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TEST_RANDOM_STATE = 0

_BALANCE_SCALE_DIR = (
    Path(__file__).parent.parent / "datasets" / "data" / "balance-scale"
)
_BALANCE_SCALE_TEST_IDX_FILE = Path(__file__).parent / "_balance_scale_test_idx.csv"


def make_balance_scale_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the seed-``TEST_RANDOM_STATE`` train/test split of Balance Scale.

    Returns
    -------
    X_train : ndarray of shape (468, 4)
        Training features.

    X_test : ndarray of shape (157, 4)
        Test features.

    y_train : ndarray of shape (468,)
        Training targets (1-indexed).

    y_test : ndarray of shape (157,)
        Test targets (1-indexed).

    Examples
    --------
    >>> from skordinal.utils._testing import make_balance_scale_split
    >>> X_train, X_test, y_train, y_test = make_balance_scale_split()
    >>> X_train.shape, X_test.shape
    ((468, 4), (157, 4))
    """
    train_file = _BALANCE_SCALE_DIR / f"train_balance-scale_{TEST_RANDOM_STATE}.csv"
    test_file = _BALANCE_SCALE_DIR / f"test_balance-scale_{TEST_RANDOM_STATE}.csv"
    train = np.loadtxt(train_file, delimiter=",")
    test = np.loadtxt(test_file, delimiter=",")
    X_train, y_train = train[:, :-1], train[:, -1].astype(int)
    X_test, y_test = test[:, :-1], test[:, -1].astype(int)
    return X_train, X_test, y_train, y_test


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
