"""Tests for the REDSVM classifier."""

import numpy as np
import numpy.testing as npt
import pytest

from orca_python.classifiers.REDSVM import REDSVM
from orca_python.datasets import load_dataset
from orca_python.testing import TEST_DATASETS_DIR, TEST_PREDICTIONS_DIR


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 2]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([0, 1, 1, 0, 1])


@pytest.mark.parametrize(
    "kernel, degree, gamma, coef0, C, cache_size, tol, shrinking, expected_file",
    [
        (0, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.0"),
        (1, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.1"),
        (2, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.2"),
        (3, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.3"),
        (4, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.4"),
        (5, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.5"),
        (6, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.6"),
        (7, 2, 0.1, 0.5, 0.1, 150, 0.005, 0, "expectedPredictions.7"),
    ],
)
def test_redsvm_predict_matches_expected(
    kernel, degree, gamma, coef0, C, cache_size, tol, shrinking, expected_file
):
    """Test that predictions match expected values."""
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    classifier = REDSVM(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        C=C,
        cache_size=cache_size,
        tol=tol,
        shrinking=shrinking,
    )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(TEST_PREDICTIONS_DIR / "REDSVM" / expected_file)

    npt.assert_equal(
        y_pred, y_expected, "The prediction doesnt match with the desired values"
    )


@pytest.mark.parametrize(
    "param_name, invalid_value, error_msg",
    [
        ("kernel", -1, "unknown kernel type"),
        ("cache_size", -1, "cache_size <= 0"),
        ("tol", -1, "eps <= 0"),
        ("shrinking", 2, "shrinking != 0 and shrinking != 1"),
        ("kernel", 8, "Wrong input format: sample_serial_number out of range"),
    ],
)
def test_redsvm_fit_hyperparameters_validation(
    X, y, param_name, invalid_value, error_msg
):
    """Test that hyperparameters are validated."""
    classifier = REDSVM(**{param_name: invalid_value})

    with pytest.raises(ValueError, match=error_msg):
        model = classifier.fit(X, y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = REDSVM()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The REDSVM fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The REDSVM fit method doesnt return Null on error"


def test_redsvm_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_redsvm_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = REDSVM()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
