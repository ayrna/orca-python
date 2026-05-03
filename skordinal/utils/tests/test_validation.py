"""Tests for the validation utilities."""

import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import skordinal.utils.validation as val_mod
from skordinal.utils.validation import (
    check_monotonic_probabilities,
    check_ordinal_targets,
    validate_thresholds,
)

_ALL_HELPERS = [
    check_ordinal_targets,
    validate_thresholds,
    check_monotonic_probabilities,
]
_HELPER_NAMES = [f.__name__ for f in _ALL_HELPERS]


@pytest.mark.parametrize("func", _ALL_HELPERS, ids=_HELPER_NAMES)
def test_api_no_mutable_defaults(func):
    """No helper has a mutable default argument."""
    for default in func.__defaults__ or ():
        assert not isinstance(default, (list, dict, np.ndarray)), (
            f"{func.__name__} has a mutable default: {default!r}"
        )


def test_api_check_monotonic_probabilities_default_repair():
    """The default value of ``repair`` is True."""
    sig = inspect.signature(check_monotonic_probabilities)
    assert sig.parameters["repair"].default is True


def test_integration_all_names_in_module_all():
    """``__all__`` lists exactly the three public helpers and is re-exported."""
    import skordinal.utils as utils_pkg

    expected = set(_HELPER_NAMES)
    # Canonical module: skordinal.utils.validation
    assert set(val_mod.__all__) == expected
    for name in expected:
        assert callable(getattr(val_mod, name))
    # Subpackage shortcut: skordinal.utils re-exports the same surface.
    assert set(utils_pkg.__all__) == expected
    for name in expected:
        assert getattr(utils_pkg, name) is getattr(val_mod, name)


def test_cot_known_encoding():
    """Exact encoding, class order, dtype, round-trip, and contiguity."""
    y = [3, 1, 2, 1, 3]
    classes, y_encoded = check_ordinal_targets(y)
    assert_array_equal(classes, [1, 2, 3])
    assert_array_equal(y_encoded, [2, 0, 1, 0, 2])
    assert y_encoded.dtype == np.intp
    assert_array_equal(classes[y_encoded], y)
    assert set(y_encoded.tolist()) == set(range(len(classes)))
    assert np.all(np.diff(classes) > 0)


def test_cot_integer_valued_floats_accepted():
    """Float labels with integer values are accepted; classes dtype stays float."""
    classes, y_encoded = check_ordinal_targets(np.array([1.0, 2.0, 3.0]))
    assert classes.dtype.kind == "f"
    assert y_encoded.dtype == np.intp


def test_cot_non_contiguous_labels():
    """Gap labels are allowed; encoding is still 0-based."""
    classes, y_encoded = check_ordinal_targets([10, 20, 30])
    assert_array_equal(y_encoded, [0, 1, 2])
    assert_array_equal(classes, [10, 20, 30])


@pytest.mark.parametrize(
    "y, match",
    [
        ([], None),
        (np.array([[1, 2], [3, 4]]), r"y must be a 1D array"),
        ([1, 1, 1], r"y must contain at least 2 unique classes"),
        (np.array(["a", "b"], dtype=object), None),
        ([np.nan, 1.0, 2.0], None),
    ],
    ids=["empty", "2d", "single-class", "object-dtype", "nan"],
)
def test_cot_invalid_input_raises(y, match):
    """Each invalid-input shape / dtype / cardinality raises ValueError."""
    expected = ValueError if match is not None else (ValueError, TypeError)
    with pytest.raises(expected, match=match):
        check_ordinal_targets(y)


@pytest.mark.parametrize(
    "thresholds",
    [
        [-1.0, 0.0, 1.0, 2.0],
        [0.0],
        [0.0, np.nextafter(0.0, 1.0)],
    ],
    ids=["generic", "binary-edge", "smallest-gap"],
)
def test_vt_valid_returns_none(thresholds):
    """Valid threshold vectors return None."""
    assert validate_thresholds(thresholds) is None


@pytest.mark.parametrize(
    "thresholds, match",
    [
        ([0.0, 0.0, 1.0], r"strictly increasing"),
        ([1.0, 0.0], r"strictly increasing"),
        ([0.0, np.inf], r"finite"),
        ([np.nan, 1.0], r"finite"),
        ([], r"length >= 1"),
        ([[0.0, 1.0]], r"1D array"),
    ],
    ids=["equal", "decreasing", "inf", "nan", "empty", "2d"],
)
def test_vt_invalid_raises(thresholds, match):
    """Each invalid threshold vector raises ValueError with a specific message."""
    with pytest.raises(ValueError, match=match):
        validate_thresholds(thresholds)


def test_cmp_valid_input_output_values():
    """Output values, row sums, and non-negativity for a valid monotonic input."""
    cumproba = np.array([[0.2, 0.5, 0.9]])
    class_proba = check_monotonic_probabilities(cumproba)
    assert_allclose(class_proba, [[0.2, 0.3, 0.4, 0.1]], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_cmp_repair_true_repairs_violation():
    """Monotonicity violations are repaired silently when ``repair=True``."""
    cumproba = np.array([[0.5, 0.3, 0.9]])
    class_proba = check_monotonic_probabilities(cumproba, repair=True)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_cmp_repair_false_valid_input():
    """``repair=False`` happy path with multiple rows."""
    cumproba = np.array([[0.2, 0.5, 0.9], [0.1, 0.4, 0.8]])
    class_proba = check_monotonic_probabilities(cumproba, repair=False)
    assert class_proba.shape == (2, 4)
    assert_allclose(class_proba[0], [0.2, 0.3, 0.4, 0.1], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)


def test_cmp_repair_false_raises_on_violation():
    """A non-monotonic row raises ValueError when ``repair=False``."""
    with pytest.raises(ValueError, match=r"cumproba rows must be non-decreasing"):
        check_monotonic_probabilities(np.array([[0.5, 0.3]]), repair=False)


@pytest.mark.parametrize("repair", [True, False])
def test_cmp_k2_single_column(repair):
    """K=2 (single input column) produces a 2-column output for both branches."""
    cumproba = np.array([[0.3], [0.7]])
    class_proba = check_monotonic_probabilities(cumproba, repair=repair)
    assert class_proba.shape == (2, 2)
    assert_allclose(class_proba[0], [0.3, 0.7], atol=1e-12)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "cumproba",
    [np.array([[-0.1, 0.5]]), np.array([[0.5, 1.2]])],
    ids=["negative-entry", "above-one-entry"],
)
def test_cmp_out_of_range_raises(cumproba):
    """Entries outside [0, 1] raise ValueError."""
    with pytest.raises(ValueError, match=r"cumproba entries must lie in \[0, 1\]"):
        check_monotonic_probabilities(cumproba)


@pytest.mark.parametrize(
    "cumproba, mass_index",
    [
        (np.array([[0.0, 0.0, 0.0]]), -1),
        (np.array([[1.0, 1.0, 1.0]]), 0),
    ],
    ids=["all-zero-row", "all-one-row"],
)
def test_cmp_special_rows_concentrate_mass(cumproba, mass_index):
    """All-zero and all-one rows place all mass on a single class."""
    class_proba = check_monotonic_probabilities(cumproba)
    assert_allclose(class_proba.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(class_proba >= 0.0)
    assert class_proba[0, mass_index] == pytest.approx(1.0)
