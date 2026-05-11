"""Metrics for ordinal classification."""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    recall_score,
)
from sklearn.utils import check_array, check_consistent_length


def _check_metric_inputs(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Coerce metric inputs to 1-D arrays and validate length consistency.

    Two-dimensional inputs are interpreted as one-hot encoded labels and
    collapsed via ``argmax`` along the last axis. Centralises the input
    coercion that every public ordinal metric needs.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_classes)
        Predicted labels or class scores.

    Returns
    -------
    y_true : ndarray of shape (n_samples,)
    y_pred : ndarray of shape (n_samples,)

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim > 1:
        y_true_arr = y_true_arr.argmax(axis=-1)
    if y_pred_arr.ndim > 1:
        y_pred_arr = y_pred_arr.argmax(axis=-1)
    check_consistent_length(y_true_arr, y_pred_arr)
    return y_true_arr, y_pred_arr


def _check_proba_inputs(
    y_true: ArrayLike, y_proba: ArrayLike, *, sum_atol: float = 1e-6
) -> tuple[NDArray, NDArray[np.float64]]:
    """Validate inputs for probabilistic ordinal metrics.

    ``y_true`` may be 1-D class labels or a 2-D one-hot matrix.
    ``y_proba`` must be a 2-D matrix coercible to ``float64`` whose rows
    sum to approximately one.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_classes)
        Ground truth labels.

    y_proba : array-like of shape (n_samples, n_classes)
        Predicted class probability matrix.

    sum_atol : float, default=1e-6
        Absolute tolerance for the row-sum check.

    Returns
    -------
    y_true : ndarray of shape (n_samples,)
    y_proba : ndarray of shape (n_samples, n_classes), dtype float64

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_proba`` have inconsistent length, or if any
        row of ``y_proba`` does not sum to 1 within ``sum_atol``.
    """
    y_true_arr = np.asarray(y_true)
    if y_true_arr.ndim > 1:
        y_true_arr = y_true_arr.argmax(axis=-1)
    y_proba_arr = check_array(
        y_proba, ensure_2d=True, dtype="float64", input_name="y_proba"
    )
    check_consistent_length(y_true_arr, y_proba_arr)
    row_sums = y_proba_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=sum_atol):
        raise ValueError(
            f"y_proba rows must sum to 1 (atol={sum_atol}); got row-sum "
            f"range [{row_sums.min():.6g}, {row_sums.max():.6g}]"
        )
    return y_true_arr, y_proba_arr


def _recall_per_class(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
) -> NDArray[np.float64]:
    """Return per-class recall as a 1-D float64 ndarray.

    Thin wrapper around :func:`sklearn.metrics.recall_score` with
    ``average=None`` and ``zero_division=0``. Centralises the call so
    public sensitivity-based metrics share one implementation.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.

    y_pred : ndarray of shape (n_samples,)
        Predicted labels.

    labels : array-like of shape (n_classes,), default=None
        Labels in the order to score. If ``None``, all unique labels are
        used.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    sensitivities : ndarray of shape (n_classes,), dtype float64
    """
    return np.asarray(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            sample_weight=sample_weight,
            zero_division=0,
        ),
        dtype=np.float64,
    )


def _per_class_mae(
    y_true: NDArray,
    y_pred: NDArray,
    *,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
) -> NDArray[np.float64]:
    """Return per-class mean absolute error as a 1-D float64 ndarray.

    Drops rows of the confusion matrix with no support (zero true
    samples for that class) so divisions remain finite. Shared by
    :func:`average_mean_absolute_error` and
    :func:`maximum_mean_absolute_error`.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.

    y_pred : ndarray of shape (n_samples,)
        Predicted labels.

    labels : array-like of shape (n_classes,), default=None
        Labels to index the confusion matrix.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    per_class_mae : ndarray of shape (n_classes_with_support,), dtype float64
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)
    n_class = cm.shape[0]
    costs = np.abs(np.arange(n_class)[:, None] - np.arange(n_class)[None, :])
    errors = costs * cm
    support = cm.sum(axis=1).astype(np.float64)
    non_zero = support > 0
    return errors[non_zero].sum(axis=1) / support[non_zero]


def average_mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Average MAE.

    Mean of the MAE metric among classes.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    average_mean_absolute_error : float
        Average mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import average_mean_absolute_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> average_mean_absolute_error(y_true, y_pred)
    0.125

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).mean())


def geometric_mean(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Geometric mean of the sensitivity (accuracy) for each class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    geometric_mean : float
        Geometric mean of the sensitivities.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import geometric_mean
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> geometric_mean(y_true, y_pred)
    0.8408964152537145

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    sum_by_class = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivities = np.diag(cm) / sum_by_class.astype("double")
    sensitivities[sum_by_class == 0] = 1
    return float(pow(np.prod(sensitivities), 1.0 / cm.shape[0]))


def gmsec(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the Geometric Mean of the Sensitivity of the Extreme Classes (GMSEC).

    Proposed in (:footcite:t:`vargas2024improving`) to assess the classification
    performance for the first and the last classes.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    gmsec : float
        Geometric mean of the sensitivities of the extreme classes.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import gmsec
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gmsec(y_true, y_pred)
    0.7071067811865476

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.sqrt(sensitivities[0] * sensitivities[-1]))


def maximum_mean_absolute_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Maximum Mean Absolute Error.

    MAE value of the class with higher distance from the true values to the predicted
    ones.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    maximum_mean_absolute_error : float
        Maximum mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import maximum_mean_absolute_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> maximum_mean_absolute_error(y_true, y_pred)
    0.5

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    return float(_per_class_mae(y_true, y_pred, sample_weight=sample_weight).max())


def minimum_sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Minimum Sensitivity.

    Lowest percentage of patterns correctly predicted as belonging to each class, with
    respect to the total number of examples in the corresponding class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    minimum_sensitivity : float
        Minimum sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import minimum_sensitivity
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> minimum_sensitivity(y_true, y_pred)
    0.5

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    sensitivities = _recall_per_class(y_true, y_pred, sample_weight=sample_weight)
    return float(np.min(sensitivities))


def mean_zero_one_error(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Mean Zero-one Error.

    Better known as error rate, is the complementary measure of CCR.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    mean_zero_one_error : float
        Mean zero-one error.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import mean_zero_one_error
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mean_zero_one_error(y_true, y_pred)
    0.2857142857142857

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    return float(1 - np.diagonal(cm).sum() / cm.sum())


def kendalls_tau(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate Kendall's tau.

    A statistic used to measure the association between two measured quantities. It is
    a measure of rank correlation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.

    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    kendalls_tau : float
        Kendall's tau.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import kendalls_tau
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> kendalls_tau(y_true, y_pred)
    0.8140915784106943

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    corr, _ = scipy.stats.kendalltau(y_true, y_pred)
    return float(corr)


def weighted_kappa(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Calculate the Weighted Kappa.

    A modified version of the Kappa statistic calculated to allow assigning
    different weights to different levels of aggregation between two variables.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    weighted_kappa : float
        Weighted Kappa.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import weighted_kappa
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> weighted_kappa(y_true, y_pred)
    0.7586206896551724

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    n_class = cm.shape[0]
    costs = np.abs(np.arange(n_class)[:, None] - np.arange(n_class)[None, :])
    f = 1 - costs

    n = cm.sum()
    x = cm / n

    r = x.sum(axis=1)
    s = x.sum(axis=0)
    Ex = r.reshape(-1, 1) * s
    po = (x * f).sum()
    pe = (Ex * f).sum()
    return float((po - pe) / (1 - pe))


def spearmans_rho(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calculate the Spearman's rank correlation coefficient.

    A non-parametric measure of statistical dependence between two variables.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    spearmans_rho : float
        Spearman rank correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import spearmans_rho
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> spearmans_rho(y_true, y_pred)
    0.9165444688834581

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    y_true_centered = y_true - np.mean(y_true)
    y_pred_centered = y_pred - np.mean(y_pred)
    num = (y_true_centered * y_pred_centered).sum()
    div = np.sqrt((y_true_centered**2).sum() * (y_pred_centered**2).sum())

    if num == 0:
        return 0.0
    return float(num / div)


def ranked_probability_score(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Compute the ranked probability score.

    As presented in :footcite:t:`janitza2016random`.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted probability distribution across different classes.

    Returns
    -------
    ranked_probability_score : float
        The ranked probability score.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import ranked_probability_score
    >>> y_true = np.array([0, 0, 3, 2])
    >>> y_pred = np.array(
    ...     [[0.2, 0.4, 0.2, 0.2],
    ...      [0.7, 0.1, 0.1, 0.1],
    ...      [0.5, 0.05, 0.1, 0.35],
    ...      [0.1, 0.05, 0.65, 0.2]])
    >>> ranked_probability_score(y_true, y_pred)
    0.5068750000000001

    """
    y_true, y_proba = _check_proba_inputs(y_true, y_proba)
    y_true = y_true.astype(np.intp)
    n_samples, n_classes = y_proba.shape

    in_range = (y_true >= 0) & (y_true < n_classes)
    y_oh = np.zeros_like(y_proba)
    rows = np.arange(n_samples)[in_range]
    y_oh[rows, y_true[in_range]] = 1.0

    y_oh_cum = y_oh.cumsum(axis=1)
    y_proba_cum = y_proba.cumsum(axis=1)

    per_sample = np.power(y_proba_cum - y_oh_cum, 2).sum(axis=1)
    per_sample[~in_range] = 1.0

    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(np.average(per_sample, weights=weights))


def accuracy_off1(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[ArrayLike] = None,
    sample_weight: Optional[ArrayLike] = None,
) -> float:
    """Computes the accuracy of the predictions.

    Allows errors if they occur in an adjacent class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    labels : np.ndarray, shape (n_classes,) or None, default=None
        Labels of the classes. If None, the labels are inferred from the data.

    Returns
    -------
    acc : float
        1-off accuracy.

    Examples
    --------
    >>> import numpy as np
    >>> from skordinal.metrics import accuracy_off1
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> accuracy_off1(y_true, y_pred)
    0.8571428571428571

    """
    y_true, y_pred = _check_metric_inputs(y_true, y_pred)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight
    )
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1), +np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return float(np.sum(correct) / np.sum(conf_mat))


def ccr(y_true, y_pred):
    """Deprecated alias for :func:`accuracy_score`."""
    warnings.warn(
        "ccr is deprecated, use accuracy_score instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return accuracy_score(y_true, y_pred)


def amae(y_true, y_pred):
    """Deprecated alias for :func:`average_mean_absolute_error`."""
    warnings.warn(
        "amae is deprecated, use average_mean_absolute_error instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return average_mean_absolute_error(y_true, y_pred)


def gm(y_true, y_pred):
    """Deprecated alias for :func:`geometric_mean`."""
    warnings.warn(
        "gm is deprecated, use geometric_mean instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return geometric_mean(y_true, y_pred)


def mae(y_true, y_pred):
    """Deprecated alias for :func:`mean_absolute_error`."""
    warnings.warn(
        "mae is deprecated, use mean_absolute_error instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return mean_absolute_error(y_true, y_pred)


def mmae(y_true, y_pred):
    """Deprecated alias for :func:`maximum_mean_absolute_error`."""
    warnings.warn(
        "mmae is deprecated, use maximum_mean_absolute_error instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return maximum_mean_absolute_error(y_true, y_pred)


def ms(y_true, y_pred):
    """Deprecated alias for :func:`minimum_sensitivity`."""
    warnings.warn(
        "ms is deprecated, use minimum_sensitivity instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return minimum_sensitivity(y_true, y_pred)


def mze(y_true, y_pred):
    """Deprecated alias for :func:`mean_zero_one_error`."""
    warnings.warn(
        "mze is deprecated, use mean_zero_one_error instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return mean_zero_one_error(y_true, y_pred)


def tkendall(y_true, y_pred):
    """Deprecated alias for :func:`kendalls_tau`."""
    warnings.warn(
        "tkendall is deprecated, use kendalls_tau instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return kendalls_tau(y_true, y_pred)


def wkappa(y_true, y_pred):
    """Deprecated alias for :func:`weighted_kappa`."""
    warnings.warn(
        "wkappa is deprecated, use weighted_kappa instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return weighted_kappa(y_true, y_pred)


def spearman(y_true, y_pred):
    """Deprecated alias for :func:`spearmans_rho`."""
    warnings.warn(
        "spearman is deprecated, use spearmans_rho instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return spearmans_rho(y_true, y_pred)


def rps(y_true, y_proba):
    """Deprecated alias for :func:`ranked_probability_score`."""
    warnings.warn(
        "rps is deprecated, use ranked_probability_score instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return ranked_probability_score(y_true, y_proba)
