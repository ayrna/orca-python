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
    "kernel, expected_file",
    [
        (0, "predictions_linear_0.csv"),
        (1, "predictions_poly_0.csv"),
        (2, "predictions_rbf_0.csv"),
        (3, "predictions_sigmoid_0.csv"),
        (4, "predictions_stump_0.csv"),
        (5, "predictions_perceptron_0.csv"),
        (6, "predictions_laplacian_0.csv"),
        (7, "predictions_exponential_0.csv"),
    ],
)
def test_redsvm_predict_matches_expected(kernel, expected_file):
    """Test that predictions match expected values."""
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    classifier = REDSVM(
        C=0.1,
        kernel=kernel,
        degree=2,
        gamma=0.1,
        coef0=0.5,
        shrinking=0,
        tol=0.005,
        cache_size=150,
    )

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(
        TEST_PREDICTIONS_DIR / "REDSVM" / expected_file, delimiter=",", usecols=1
    )

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
