"""Tests for the scaling functions in the preprocessing module."""

import numpy as np
import numpy.testing as npt
import pytest

from orca_python.preprocessing import apply_scaling, minmax_scale, standardize
from orca_python.preprocessing.scalers import _validate_and_align
from orca_python.testing import TEST_RANDOM_STATE


@pytest.fixture
def X_train():
    """Create synthetic training data for testing."""
    return np.random.RandomState(TEST_RANDOM_STATE).randn(100, 5)


@pytest.fixture
def X_test():
    """Create synthetic test data for testing."""
    return np.random.RandomState(TEST_RANDOM_STATE).randn(50, 5)


def test_validate_and_align_valid_inputs(X_train, X_test):
    """Test _validate_and_align with valid matching inputs."""
    X_train_valid, X_test_valid = _validate_and_align(X_train, X_test)

    assert X_train_valid.shape == X_train.shape
    assert X_test_valid.shape == X_test.shape
    npt.assert_array_equal(X_train_valid, X_train)
    npt.assert_array_equal(X_test_valid, X_test)


def test_validate_and_align_none_test(X_train):
    """Test _validate_and_align with None test data."""
    X_train_valid, X_test_valid = _validate_and_align(X_train, None)

    assert X_train_valid.shape == X_train.shape
    assert X_test_valid is None
    npt.assert_array_equal(X_train_valid, X_train)


def test_validate_and_align_mismatched_features(X_train, X_test):
    """Test _validate_and_align raises error for mismatched feature counts."""
    X_invalid = X_test[:, :-1]

    error_msg = "X_test has 4 features but X_train has 5."
    with pytest.raises(ValueError, match=error_msg):
        _validate_and_align(X_train, X_invalid)


def test_validate_and_align_invalid_input():
    """Test _validate_and_align raises error for invalid input types."""
    with pytest.raises((ValueError, TypeError)):
        _validate_and_align("invalid", None)


def test_minmax_scale_data(X_train, X_test):
    """Test that minmax_scale function correctly scales input data to [0,1] range."""
    X_train_scaled, X_test_scaled = minmax_scale(X_train, X_test)

    assert np.all(X_train_scaled >= 0) and np.all(X_train_scaled <= 1)
    assert np.all(X_test_scaled >= 0) and np.all(X_test_scaled <= 1)


def test_minmax_scale_return_transformer(X_train, X_test):
    """Test that minmax_scale returns transformer when requested."""
    _, expected_X_test, scaler = minmax_scale(X_train, X_test, return_transformer=True)

    X_test_scaled = scaler.transform(X_test)
    npt.assert_array_almost_equal(X_test_scaled, expected_X_test)


def test_standardize_data(X_train, X_test):
    """Test that standardize function correctly produces output with zero mean
    and unit variance."""
    X_train_scaled, _ = standardize(X_train, X_test)

    npt.assert_almost_equal(np.mean(X_train_scaled), 0, decimal=6)
    npt.assert_almost_equal(np.std(X_train_scaled), 1, decimal=6)


def test_standardize_return_transformer(X_train, X_test):
    """Test that standardize returns transformer when requested."""
    _, expected_X_test, scaler = standardize(X_train, X_test, return_transformer=True)

    X_test_scaled = scaler.transform(X_test)
    npt.assert_array_almost_equal(X_test_scaled, expected_X_test)


@pytest.mark.parametrize(
    "method, scaling_func", [("norm", minmax_scale), ("std", standardize)]
)
def test_apply_scaling_correctly(X_train, X_test, method, scaling_func):
    """Test that different preprocessing methods work as expected."""
    expected_X_train, expected_X_test = scaling_func(X_train, X_test)
    X_train_scaled, X_test_scaled = apply_scaling(X_train, X_test, method)

    npt.assert_array_almost_equal(X_train_scaled, expected_X_train)
    npt.assert_array_almost_equal(X_test_scaled, expected_X_test)


def test_apply_scaling_none_method(X_train, X_test):
    """Test that scaling function handles None input correctly."""
    post_X_train, post_X_test = apply_scaling(X_train, X_test, None)

    npt.assert_array_equal(post_X_train, X_train)
    npt.assert_array_equal(post_X_test, X_test)


def test_apply_scaling_return_transformer(X_train, X_test):
    """Test that the transformer returned by apply_scaling works as expected."""
    _, _, scaler = apply_scaling(X_train, X_test, "norm", return_transformer=True)
    X_test_scaled = scaler.transform(X_test)
    _, expected_X_test = minmax_scale(X_train, X_test)
    npt.assert_array_almost_equal(X_test_scaled, expected_X_test)

    _, _, scaler = apply_scaling(X_train, X_test, "std", return_transformer=True)
    X_test_scaled = scaler.transform(X_test)
    _, expected_X_test = standardize(X_train, X_test)
    npt.assert_array_almost_equal(X_test_scaled, expected_X_test)

    _, _, scaler = apply_scaling(X_train, X_test, None, return_transformer=True)
    assert scaler is None


def test_apply_scaling_case_insensitive(X_train, X_test):
    """Test that apply_scaling handles different case variations."""
    X_train_lower, X_test_lower = apply_scaling(X_train, X_test, "norm")
    X_train_upper, X_test_upper = apply_scaling(X_train, X_test, "NORM")

    npt.assert_array_equal(X_train_lower, X_train_upper)
    npt.assert_array_equal(X_test_lower, X_test_upper)


def test_apply_scaling_invalid_method_type(X_train, X_test):
    """Test that an invalid scaling method type raises a ValueError."""
    error_msg = "Scaling method must be a string or None."
    with pytest.raises(ValueError, match=error_msg):
        apply_scaling(X_train, X_test, 123)


def test_apply_scaling_unknown_method(X_train, X_test):
    """Test that an unknown scaling method raises a ValueError."""
    error_msg = "Unknown scaling method 'invalid'. Valid options: 'norm', 'std', None."
    with pytest.raises(ValueError, match=error_msg):
        apply_scaling(X_train, X_test, "invalid")


def test_apply_scaling_inconsistent_features(X_train, X_test):
    """Test that scaling with inconsistent feature dimensions raises ValueError."""
    X_invalid = X_test[:, :-1]

    with pytest.raises(ValueError):
        apply_scaling(X_train, X_invalid, "norm")


def test_minmax_scale_inconsistent_features(X_train, X_test):
    """Test that minmax_scale raises ValueError for mismatched features."""
    X_invalid = X_test[:, :-1]

    with pytest.raises(ValueError):
        minmax_scale(X_train, X_invalid)


def test_standardize_inconsistent_features(X_train, X_test):
    """Test that standardize raises ValueError for mismatched features."""
    X_invalid = X_test[:, :-1]

    with pytest.raises(ValueError):
        standardize(X_train, X_invalid)
