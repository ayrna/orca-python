"""scikit-learn cross-version compatibility helpers."""

from __future__ import annotations

try:
    from sklearn.utils.validation import validate_data as _sk_validate_data

    _HAS_VALIDATE_DATA = True
except ImportError:  # scikit-learn < 1.6
    _HAS_VALIDATE_DATA = False


def validate_data(estimator, X, y=None, *, reset=True, **check_params):
    """Validate ``X`` (and optionally ``y``) for a scikit-learn estimator.

    Thin wrapper around the API split introduced in scikit-learn 1.6.
    On 1.6+ it forwards to :func:`sklearn.utils.validation.validate_data`;
    on 1.3-1.5 it falls back to ``estimator._validate_data``. The helper
    lives in a dedicated module so it can be removed in one place once
    the minimum supported scikit-learn version is bumped to 1.6.

    Parameters
    ----------
    estimator : estimator instance
        The estimator to validate the input for. Mutates the estimator
        and sets ``n_features_in_`` (and ``feature_names_in_`` when
        applicable) when ``reset=True``.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples.

    y : array-like of shape (n_samples,), default=None
        The target vector. If ``None``, only ``X`` is validated.

    reset : bool, default=True
        Whether to reset the ``n_features_in_`` attribute. If ``False``,
        the input will be checked for consistency with data provided when
        ``reset`` was last ``True``. Set to ``False`` at predict /
        transform time.

    **check_params : kwargs
        Parameters passed to :func:`~sklearn.utils.check_array` or
        :func:`~sklearn.utils.check_X_y`. Ignored if ``validate_separately``
        is not False.

    Returns
    -------
    out : {ndarray, sparse matrix} or tuple of these
        The validated input. A ``(X, y)`` tuple is returned when ``y``
        is provided, otherwise the validated ``X`` alone.

    Notes
    -----
    Passing ``y=None`` is rewritten internally to scikit-learn's
    ``"no_validation"`` sentinel before forwarding to the upstream
    validator on scikit-learn 1.6+. This means the estimator's
    ``requires_y`` tag is silently bypassed; classifiers that need a
    target should pass ``y`` explicitly during ``fit``.
    """
    if _HAS_VALIDATE_DATA:
        y_arg = "no_validation" if y is None else y
        return _sk_validate_data(estimator, X, y_arg, reset=reset, **check_params)
    if y is None:
        return estimator._validate_data(X, reset=reset, **check_params)
    return estimator._validate_data(X, y, reset=reset, **check_params)
