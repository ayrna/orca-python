"""Tests for the metrics module."""

import numpy as np
import numpy.testing as npt
import pytest
from sklearn.metrics import recall_score

from skordinal.metrics import (
    accuracy_off1,
    accuracy_score,
    average_mean_absolute_error,
    geometric_mean,
    gmsec,
    kendalls_tau,
    maximum_mean_absolute_error,
    mean_absolute_error,
    mean_zero_one_error,
    minimum_sensitivity,
    ranked_probability_score,
    spearmans_rho,
    weighted_kappa,
)


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


def test_accuracy_score():
    """Test the Correctly Classified Ratio (accuracy_score) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.8000
    actual = accuracy_score(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = accuracy_score(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_average_mean_absolute_error():
    """Test the Average Mean Absolute Error (average_mean_absolute_error) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 0.0
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected = 1.0
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected = 2.0
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected = 0.125
    actual = average_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_geometric_mean():
    """Test the Geometric Mean (geometric_mean) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.7991
    actual = geometric_mean(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = geometric_mean(y_true, y_pred)
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


def test_mean_absolute_error():
    """Test the Mean Absolute Error (mean_absolute_error) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.3000
    actual = mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_maximum_mean_absolute_error():
    """Test the Maximum Mean Absolute Error (maximum_mean_absolute_error) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 0.0
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    expected = 2.0
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 2, 1, 3])
    y_pred = np.array([2, 2, 0, 3, 1])
    expected = 2.0
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 1, 2, 3, 3])
    y_pred = np.array([0, 1, 2, 3, 4])
    expected = 0.5
    actual = maximum_mean_absolute_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_minimum_sensitivity():
    """Test the Minimum Sensitivity (minimum_sensitivity) metric."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    expected = 0.5
    actual = minimum_sensitivity(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    expected = 1.0
    actual = minimum_sensitivity(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = minimum_sensitivity(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_mean_zero_one_error():
    """Test the Mean Zero-one Error (mean_zero_one_error) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.2000
    actual = mean_zero_one_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.5
    actual = mean_zero_one_error(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_ranked_probability_score():
    """Test the ranked probability score (ranked_probability_score) metric."""
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
    actual = ranked_probability_score(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_kendalls_tau():
    """Test the Kendall's Tau (kendalls_tau) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6240
    actual = kendalls_tau(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = kendalls_tau(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_weighted_kappa():
    """Test the Weighted Kappa (weighted_kappa) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6703
    actual = weighted_kappa(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = weighted_kappa(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


def test_spearmans_rho():
    """Test the Spearman's rank correlation coefficient (spearmans_rho) metric."""
    y_true = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    y_pred = np.array([1, 3, 3, 1, 2, 3, 1, 2, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3, 1, 3])
    expected = 0.6429
    actual = spearmans_rho(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=4)

    # Test using one-hot and probabilities
    y_true = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    y_pred = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    expected = 0.0
    actual = spearmans_rho(y_true, y_pred)
    npt.assert_almost_equal(expected, actual, decimal=6)


@pytest.mark.parametrize(
    "name",
    [
        "ccr",
        "amae",
        "gm",
        "mae",
        "mmae",
        "ms",
        "mze",
        "tkendall",
        "wkappa",
        "spearman",
    ],
)
def test_deprecated_alias_warns(name):
    """Calling a deprecated short-name alias emits DeprecationWarning."""
    import skordinal.metrics as m

    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 2, 0])
    fn = getattr(m, name)
    with pytest.warns(DeprecationWarning, match=name):
        fn(y_true, y_pred)


def test_deprecated_rps_warns():
    """Calling the deprecated rps alias emits DeprecationWarning."""
    import skordinal.metrics as m

    y_true = np.array([0, 1, 2])
    y_proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    with pytest.warns(DeprecationWarning, match="rps"):
        m.rps(y_true, y_proba)


def test_deprecated_alias_not_in_all():
    """Deprecated short names are not present in skordinal.metrics.__all__."""
    import skordinal.metrics as m

    deprecated = [
        "ccr",
        "amae",
        "gm",
        "mae",
        "mmae",
        "ms",
        "mze",
        "tkendall",
        "wkappa",
        "spearman",
        "rps",
    ]
    for name in deprecated:
        assert name not in m.__all__, f"{name!r} should not be in __all__"


def test_metric_names_in_all():
    """All public metric names are present in skordinal.metrics.__all__."""
    import skordinal.metrics as m

    expected = [
        "accuracy_score",
        "average_mean_absolute_error",
        "geometric_mean",
        "mean_absolute_error",
        "maximum_mean_absolute_error",
        "minimum_sensitivity",
        "mean_zero_one_error",
        "kendalls_tau",
        "weighted_kappa",
        "spearmans_rho",
        "ranked_probability_score",
        "gmsec",
        "accuracy_off1",
    ]
    for name in expected:
        assert name in m.__all__, f"{name!r} missing from __all__"
