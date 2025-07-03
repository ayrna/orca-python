"""Tests for the metrics module."""

import numpy.testing as npt
import pytest
from numpy import array

from orca_python.metrics import (
    amae,
    ccr,
    gm,
    mae,
    mmae,
    ms,
    mze,
    spearman,
    tkendall,
    wkappa,
)


@pytest.fixture
def real_y():
    return array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])


@pytest.fixture
def predicted_y():
    return array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])


def test_ccr(real_y, predicted_y):
    """Test the Correctly Classified Ratio (CCR) metric."""
    expected = 0.8000
    actual = ccr(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_amae(real_y, predicted_y):
    """Test the Average Mean Absolute Error (AMAE) metric."""
    expected = 0.2937
    actual = amae(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_gm(real_y, predicted_y):
    """Test the Geometric Mean (GM) metric."""
    expected = 0.7991
    actual = gm(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_mae(real_y, predicted_y):
    """Test the Mean Absolute Error (MAE) metric."""
    expected = 0.3000
    actual = mae(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_mmae(real_y, predicted_y):
    """Test the Mean Mean Absolute Error (MMAE) metric."""
    expected = 0.4286
    actual = mmae(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_ms(real_y, predicted_y):
    """Test the Mean Sensitivity (MS) metric."""
    expected = 0.7143
    actual = ms(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_mze(real_y, predicted_y):
    """Test the Mean Zero-one Error (MZE) metric."""
    expected = 0.2000
    actual = mze(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_tkendall(real_y, predicted_y):
    """Test the Kendall's Tau metric."""
    expected = 0.6240
    actual = tkendall(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_wkappa(real_y, predicted_y):
    """Test the Weighted Kappa metric."""
    expected = 0.6703
    actual = wkappa(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)


def test_spearman(real_y, predicted_y):
    """Test the Spearman's rank correlation coefficient metric."""
    expected = 0.6429
    actual = spearman(real_y, predicted_y)
    npt.assert_almost_equal(expected, actual, decimal=4)
