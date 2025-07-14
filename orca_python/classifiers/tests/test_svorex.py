"""Tests for the SVOREX classifier."""

import numpy as np
import numpy.testing as npt
import pytest

from orca_python.classifiers.SVOREX import SVOREX
from orca_python.datasets import load_dataset
from orca_python.testing import TEST_DATASETS_DIR, TEST_PREDICTIONS_DIR


@pytest.fixture
def X():
    """Create sample feature patterns for testing."""
    return np.array([[1, 2], [2, 1], [2, 2], [1, 1], [2, 3]])


@pytest.fixture
def y():
    """Create sample target variables for testing."""
    return np.array([1, 2, 2, 1, 2])


@pytest.mark.parametrize(
    "kernel, expected_file",
    [
        ("gaussian", "predictions_gaussian_0.csv"),
        ("linear", "predictions_linear_0.csv"),
        ("poly", "predictions_poly_0.csv"),
    ],
)
def test_svorex_predict_matches_expected(kernel, expected_file):
    """Test that predictions match expected values."""
    X_train, y_train, X_test, _ = load_dataset(
        dataset_name="balance-scale", data_path=TEST_DATASETS_DIR
    )

    classifier = SVOREX(C=0.5, kernel=kernel, degree=4, tol=0.002, kappa=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_expected = np.loadtxt(
        TEST_PREDICTIONS_DIR / "SVOREX" / expected_file, delimiter=",", usecols=1
    )

    npt.assert_equal(
        y_pred, y_expected, "The prediction doesnt match with the desired values"
    )


@pytest.mark.parametrize(
    "params, error_msg",
    [
        ({"tol": 0}, "- T is invalid"),
        ({"C": 0}, "- C is invalid"),
        ({"kappa": 0}, "- K is invalid"),
        ({"kernel": "poly", "degree": 0}, "- P is invalid"),
        ({"kappa": -1}, "-1 is invalid"),
    ],
)
def test_svorex_fit_hyperparameters_validation(X, y, params, error_msg):
    """Test that hyperparameters are validated."""
    classifier = SVOREX(**params)

    with pytest.raises(ValueError, match=error_msg):
        model = classifier.fit(X, y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"


def test_svorex_fit_input_validation(X, y):
    """Test that input data is validated."""
    X_invalid = X[:-1, :-1]
    y_invalid = y[:-1]

    classifier = SVOREX()
    with pytest.raises(ValueError):
        model = classifier.fit(X, y_invalid)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit([], y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X, [])
        assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
        model = classifier.fit(X_invalid, y)
        assert model is None, "The SVOREX fit method doesnt return Null on error"


def test_svorex_validates_internal_model_format(X, y):
    """Test that internal model format is validated."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(TypeError, match="Model should be a dictionary!"):
        classifier.model_ = 1
        classifier.predict(X)


def test_svorex_predict_invalid_input_raises_error(X, y):
    """Test that invalid input raises an error."""
    classifier = SVOREX()
    classifier.fit(X, y)

    with pytest.raises(ValueError):
        classifier.predict([])
