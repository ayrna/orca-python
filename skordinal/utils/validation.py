"""Validation utilities for ordinal classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import isotonic_regression
from sklearn.utils import check_array
from sklearn.utils.multiclass import check_classification_targets

__all__ = [
    "check_ordinal_targets",
    "validate_thresholds",
    "check_monotonic_probabilities",
]


def check_ordinal_targets(
    y: ArrayLike,
) -> tuple[NDArray, NDArray[np.intp]]:
    """Validate an ordinal target vector and return its integer encoding.

    Accepts integer or integer-valued float labels. Rejects string/object
    arrays, continuous floats, arrays with fewer than 2 unique classes,
    and empty or multi-dimensional inputs.

    The return order ``(classes, y_encoded)`` mirrors
    ``np.unique(y, return_inverse=True)`` so callers can write
    ``self.classes_, y_enc = check_ordinal_targets(y)`` in ``fit()``.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels. Must be 1-D and have a numeric dtype. Labels need
        not form a contiguous range; gaps are allowed (e.g. ``[3, 5, 7]``
        is mapped to ``[0, 1, 2]``). Integer-valued floats (e.g.
        ``[1.0, 2.0]``) are accepted; continuous floats (e.g.
        ``[0.5, 1.5]``) are rejected by the upstream sklearn check.

    Returns
    -------
    classes : ndarray of shape (n_classes,)
        Unique labels sorted in ascending order. Dtype matches the
        original dtype of ``y``.

    y_encoded : ndarray of shape (n_samples,), dtype np.intp
        Zero-based contiguous encoding such that
        ``classes[y_encoded[i]] == y[i]`` for every sample ``i``.

    Raises
    ------
    ValueError
        If ``y`` is empty, multi-dimensional, has a non-numeric dtype,
        or contains fewer than 2 unique classes. Upstream ``ValueError``
        from ``check_array`` (e.g. NaN inputs, object arrays) and from
        ``check_classification_targets`` (e.g. continuous targets) are
        propagated unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.validation import check_ordinal_targets
    >>> classes, y_enc = check_ordinal_targets(np.array([3, 1, 2, 1, 3]))
    >>> classes
    array([1, 2, 3])
    >>> y_enc
    array([2, 0, 1, 0, 2])
    """
    y = check_array(
        y, ensure_2d=False, dtype="numeric", ensure_min_samples=1, input_name="y"
    )

    if y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y.shape}")

    check_classification_targets(y)

    classes, y_encoded = np.unique(y, return_inverse=True)

    if classes.size < 2:
        raise ValueError(
            f"y must contain at least 2 unique classes, got {classes.size}"
        )

    return classes, y_encoded.astype(np.intp)


def validate_thresholds(thresholds: ArrayLike) -> None:
    """Check that thresholds are strictly increasing and finite.

    A valid threshold vector must be 1-D, contain only finite values,
    have at least one entry, and have strictly positive consecutive
    differences.

    Parameters
    ----------
    thresholds : array-like of shape (n_classes - 1,)
        Threshold values defining the boundaries between ordinal classes.
        Must be strictly increasing. Length must be at least 1, i.e.
        ``n_classes >= 2``. Length 1 (binary case) trivially satisfies
        the monotonicity check.

    Raises
    ------
    ValueError
        If ``thresholds`` is not 1-D, is empty, contains non-finite
        values, or is not strictly increasing.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.validation import validate_thresholds
    >>> validate_thresholds(np.array([-1.0, 0.0, 1.0]))  # returns None
    """
    thresholds = np.asarray(thresholds, dtype=float)

    if thresholds.ndim != 1:
        raise ValueError(f"thresholds must be a 1D array, got shape {thresholds.shape}")

    if thresholds.size < 1:
        raise ValueError("thresholds must have length >= 1, got 0")

    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be finite, got non-finite values")

    diffs = np.diff(thresholds)
    if (diffs <= 0).any():
        raise ValueError(
            f"thresholds must be strictly increasing, got differences {diffs!r}"
        )


def check_monotonic_probabilities(
    cumproba: ArrayLike,
    repair: bool = True,
) -> NDArray[np.float64]:
    """Convert cumulative class probabilities to class-wise probabilities.

    Takes a matrix of cumulative probabilities ``P(Y <= k | x)`` for
    ``k = 1, ..., n_classes - 1`` and returns a matrix of class-wise
    probabilities ``P(Y = k | x)`` for ``k = 1, ..., n_classes``.

    When ``repair=True``, monotonicity violations in each row are
    silently fixed via isotonic regression before differencing. When
    ``repair=False``, any non-monotonic row triggers a ``ValueError``.

    Special row behaviors:

    - An all-zero row becomes ``[0, 0, ..., 1]``; the final class absorbs
      all probability mass.
    - An all-one row becomes ``[1, 0, ..., 0]``; the first class absorbs
      all probability mass.

    Parameters
    ----------
    cumproba : array-like of shape (n_samples, n_classes - 1)
        Cumulative probabilities. Each row should be a non-decreasing
        sequence of values in ``[0, 1]``.

    repair : bool, default=True
        If ``True``, apply isotonic regression row-wise to enforce
        monotonicity before differencing, then clip and renormalise.
        If ``False``, raise ``ValueError`` when any row is non-monotonic.

    Returns
    -------
    class_proba : ndarray of shape (n_samples, n_classes), dtype np.float64
        Class-wise probabilities. Each row is non-negative and sums to
        exactly ``1.0``.

    Raises
    ------
    ValueError
        If ``cumproba`` is not 2-D, has zero columns, contains NaN / inf
        values, contains values outside ``[0, 1]``, or — when
        ``repair=False`` — has any row whose entries are not
        non-decreasing. Upstream ``ValueError`` from ``check_array``
        (e.g. NaN inputs) is propagated unchanged.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.utils.validation import check_monotonic_probabilities
    >>> cumproba = np.array([[0.2, 0.5, 0.9]])
    >>> check_monotonic_probabilities(cumproba)
    array([[0.2, 0.3, 0.4, 0.1]])
    """
    cumproba = check_array(
        cumproba, ensure_2d=True, dtype=np.float64, input_name="cumproba"
    )

    if cumproba.min() < 0.0 or cumproba.max() > 1.0:
        raise ValueError(
            f"cumproba entries must lie in [0, 1], got range "
            f"[{cumproba.min():.4g}, {cumproba.max():.4g}]"
        )

    n_samples, n_thresholds = cumproba.shape
    class_proba = np.empty((n_samples, n_thresholds + 1), dtype=np.float64)

    if not repair:
        diffs = np.diff(cumproba, axis=1)
        if (diffs < 0.0).any():
            raise ValueError(
                f"cumproba rows must be non-decreasing, got minimum diff "
                f"{diffs.min():.4g}"
            )
        class_proba[:, 0] = cumproba[:, 0]
        class_proba[:, 1:-1] = diffs
        class_proba[:, -1] = 1.0 - cumproba[:, -1]
        return class_proba

    for i in range(n_samples):
        row_iso = isotonic_regression(
            cumproba[i], y_min=0.0, y_max=1.0, increasing=True
        )
        class_proba[i, 0] = row_iso[0]
        class_proba[i, 1:-1] = np.diff(row_iso)
        class_proba[i, -1] = 1.0 - row_iso[-1]

    np.clip(class_proba, 0.0, None, out=class_proba)
    row_sums = class_proba.sum(axis=1, keepdims=True)
    # Defensive: y_min=0 prevents all-zero rows; clip is a safety net.
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    class_proba /= row_sums
    return class_proba
