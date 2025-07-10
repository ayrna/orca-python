"""Tests for the metrics module."""

import numpy as np
import numpy.testing as npt
from sklearn.metrics import recall_score

from orca_python.metrics import (
    accuracy_off1,
    amae,
    ccr,
    gm,
    gmsec,
    greater_is_better,
    mae,
    mmae,
    ms,
    mze,
    rps,
    spearman,
    tkendall,
    wkappa,
)


def test_greater_is_better():
    """Test the greater_is_better function."""
    assert greater_is_better("accuracy_off1")
    assert greater_is_better("ccr")
    assert greater_is_better("gm")
    assert greater_is_better("gmsec")
    assert not greater_is_better("mae")
    assert not greater_is_better("mmae")
    assert not greater_is_better("amae")
    assert greater_is_better("ms")
    assert not greater_is_better("mze")
    assert not greater_is_better("rps")
    assert greater_is_better("tkendall")
    assert greater_is_better("wkappa")
    assert greater_is_better("spearman")


def test_accuracy_off1():
    """Test the Accuracy that allows errors in adjacent classes."""
    y_true = np.array([0, 1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5, 0])
    expected = 0.8333333333333334
    actual = accuracy_off1(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 2, 1, 4, 3])
    expected = 1.0
    actual = accuracy_off1(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_ccr():
    """Test the Correctly Classified Ratio (CCR) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.8000
    actual = ccr(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = ccr(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_amae():
    """Test the Average Mean Absolute Error (AMAE) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 0.0
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected = 1.0
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected = 2.0
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected = 0.125
    actual = amae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_gm():
    """Test the Geometric Mean (GM) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.7991
    actual = gm(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = gm(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_gmsec():
    """Test the Geometric Mean of Sensitivity and Specificity (GMSEC) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected = np.sqrt(sensitivities[0] * sensitivities[-1])
    actual = gmsec(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    sensitivities = recall_score(y_true, y_pred, average=None)
    expected = np.sqrt(sensitivities[0] * sensitivities[-1])
    actual = gmsec(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_mae():
    """Test the Mean Absolute Error (MAE) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.3000
    actual = mae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = mae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_mmae():
    """Test the Maximum Mean Absolute Error (MMAE) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 0.0
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected = 2.0
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected = 2.0
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected = 0.5
    actual = mmae(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_ms():
    """Test the Mean Sensitivity (MS) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = ms(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 1.0
    actual = ms(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = ms(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_mze():
    """Test the Mean Zero-one Error (MZE) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.2000
    actual = mze(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = mze(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_rps():
    """Test the ranked probability score (RPS) metric."""
    y_true = np.array([0, 0, 3, 2])
    y_pred = np.array(
        [
            [0.2, 0.4, 0.2, 0.2],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.05, 0.1, 0.35],
            [0.1, 0.05, 0.65, 0.2],
        ]
    )
    expected = 0.506875
    actual = rps(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_tkendall():
    """Test the Kendall's Tau metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6240
    actual = tkendall(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = tkendall(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_wkappa():
    """Test the Weighted Kappa metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6703
    actual = wkappa(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = wkappa(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_spearman():
    """Test the Spearman's rank correlation coefficient metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6429
    actual = spearman(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = spearman(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)
