"""Utilities for ordinal classification."""

from skordinal.utils.validation import (
    check_monotonic_probabilities,
    check_ordinal_targets,
    validate_thresholds,
)

__all__ = [
    "check_monotonic_probabilities",
    "check_ordinal_targets",
    "validate_thresholds",
]
