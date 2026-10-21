"""Class-imbalance metrics for ordinal classification targets."""

from __future__ import annotations

from collections import Counter

from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import check_array


def _class_counts(y) -> dict[object, int]:
    """Return the count of each class in a one-dimensional target array."""
    targets = check_array(y, ensure_2d=False, dtype=None, input_name="y")
    if targets.ndim != 1:
        raise ValueError(f"y must be one-dimensional; got shape {targets.shape}.")

    counts = dict(Counter(targets.tolist()))
    if len(counts) < 2:
        raise ValueError("y must contain at least two distinct classes.")
    return counts


def _class_imbalance_ratios(class_counts: dict[object, int]) -> list[float]:
    """Return the per-class imbalance ratios for validated class counts."""
    total_samples = sum(class_counts.values())
    n_classes = len(class_counts)
    return [
        (total_samples - count) / (n_classes * count) for count in class_counts.values()
    ]


@validate_params({"y": ["array-like"]}, prefer_skip_nested_validation=True)
def compute_dataset_air(y) -> float:
    """Compute the Average Imbalance Ratio (AIR) from class labels.

    For each class :math:`i`, its imbalance ratio is
    ``(N - n_i) / (K * n_i)``, where ``N`` is the number of target values,
    ``n_i`` is the count for class ``i``, and ``K`` is the number of classes.
    AIR is the mean of these per-class ratios.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels. ``y`` can be taken from a dataset returned by
        :func:`~skordinal.datasets.load_dataset` or supplied directly from
        any other source.

    Returns
    -------
    air : float
        Mean of the per-class imbalance ratios.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or contains fewer than two distinct
        classes.

    Examples
    --------
    >>> from skordinal.datasets import compute_dataset_air, load_dataset
    >>> dataset = load_dataset("balance_scale")
    >>> air = compute_dataset_air(dataset.target)
    """
    class_counts = _class_counts(y)
    class_irs = _class_imbalance_ratios(class_counts)
    return sum(class_irs) / len(class_irs)


@validate_params({"y": ["array-like"]}, prefer_skip_nested_validation=True)
def compute_dataset_mir(y) -> float:
    """Compute the Maximum Imbalance Ratio (MIR) from class labels.

    For each class :math:`i`, its imbalance ratio is
    ``(N - n_i) / (K * n_i)``, where ``N`` is the number of target values,
    ``n_i`` is the count for class ``i``, and ``K`` is the number of classes.
    MIR is the maximum per-class ratio divided by the minimum. A perfectly
    balanced dataset has MIR equal to 1.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels. ``y`` can be taken from a dataset returned by
        :func:`~skordinal.datasets.load_dataset` or supplied directly from
        any other source.

    Returns
    -------
    mir : float
        Maximum per-class imbalance ratio divided by the minimum per-class
        imbalance ratio.

    Raises
    ------
    ValueError
        If ``y`` is not one-dimensional or contains fewer than two distinct
        classes.

    Examples
    --------
    >>> from skordinal.datasets import compute_dataset_mir, load_dataset
    >>> dataset = load_dataset("balance_scale")
    >>> mir = compute_dataset_mir(dataset.target)
    """
    class_counts = _class_counts(y)
    class_irs = _class_imbalance_ratios(class_counts)
    return max(class_irs) / min(class_irs)
