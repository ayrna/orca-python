"""Testing utilities for the skordinal test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TEST_RANDOM_STATE = 0

_BALANCE_SCALE_DIR = (
    Path(__file__).parent.parent / "datasets" / "data" / "balance-scale"
)


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
