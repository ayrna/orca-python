"""Tests for the metrics module."""

import pytest
from numpy import array
import numpy.testing as npt

# path.append('..')

# import metrics
import orca_python.metrics as metrics


@pytest.fixture
def real_y():
    return array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])


@pytest.fixture
def predicted_y():
    return array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])


def test_ccr(real_y, predicted_y):
    real_ccr = 0.8000
    predicted_ccr = metrics.ccr(real_y, predicted_y)
    npt.assert_almost_equal(real_ccr, predicted_ccr, decimal=4)


def test_amae(real_y, predicted_y):
    real_amae = 0.2937
    predicted_amae = metrics.amae(real_y, predicted_y)
    npt.assert_almost_equal(real_amae, predicted_amae, decimal=4)


def test_gm(real_y, predicted_y):
    real_gm = 0.7991
    predicted_gm = metrics.gm(real_y, predicted_y)
    npt.assert_almost_equal(real_gm, predicted_gm, decimal=4)


def test_mae(real_y, predicted_y):
    real_mae = 0.3000
    predicted_mae = metrics.mae(real_y, predicted_y)
    npt.assert_almost_equal(real_mae, predicted_mae, decimal=4)


def test_mmae(real_y, predicted_y):
    real_mmae = 0.4286
    predicted_mmae = metrics.mmae(real_y, predicted_y)
    npt.assert_almost_equal(real_mmae, predicted_mmae, decimal=4)


def test_ms(real_y, predicted_y):
    real_ms = 0.7143
    predicted_ms = metrics.ms(real_y, predicted_y)
    npt.assert_almost_equal(real_ms, predicted_ms, decimal=4)


def test_mze(real_y, predicted_y):
    real_mze = 0.2000
    predicted_mze = metrics.mze(real_y, predicted_y)
    npt.assert_almost_equal(real_mze, predicted_mze, decimal=4)


def test_tkendall(real_y, predicted_y):
    real_tkendall = 0.6240
    predicted_tkendall = metrics.tkendall(real_y, predicted_y)
    npt.assert_almost_equal(real_tkendall, predicted_tkendall, decimal=4)


def test_wkappa(real_y, predicted_y):
    real_wkappa = 0.6703
    predicted_wkappa = metrics.wkappa(real_y, predicted_y)
    npt.assert_almost_equal(real_wkappa, predicted_wkappa, decimal=4)


def test_spearman(real_y, predicted_y):
    real_spearman = 0.6429
    predicted_spearman = metrics.spearman(real_y, predicted_y)
    npt.assert_almost_equal(real_spearman, predicted_spearman, decimal=4)
