"""Metrics for ordinal classification."""

from __future__ import division

import warnings

import numpy as np
import scipy.stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    recall_score,
)


def average_mean_absolute_error(y_true, y_pred):
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
    np.float64(0.125)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return np.mean(per_class_maes)


def geometric_mean(y_true, y_pred):
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
    np.float64(0.8408964152537145)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    sum_by_class = np.sum(cm, axis=1)
    sensitivities = np.diag(cm) / sum_by_class.astype("double")
    sensitivities[sum_by_class == 0] = 1
    gm = pow(np.prod(sensitivities), 1.0 / cm.shape[0])
    return gm


def gmsec(y_true, y_pred):
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
    np.float64(0.7071067811865476)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.sqrt(sensitivities[0] * sensitivities[-1])


def maximum_mean_absolute_error(y_true, y_pred):
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
    np.float64(0.5)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    errors = costs * cm

    # Remove rows with all zeros in the confusion matrix
    non_zero_cm_rows = ~np.all(cm == 0, axis=1)
    errors = errors[non_zero_cm_rows]
    cm = cm[non_zero_cm_rows]

    per_class_maes = np.sum(errors, axis=1) / np.sum(cm, axis=1).astype("double")
    return per_class_maes.max()


def minimum_sensitivity(y_true, y_pred):
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
    np.float64(0.5)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    sensitivities = recall_score(y_true, y_pred, average=None)
    return np.min(sensitivities)


def mean_zero_one_error(y_true, y_pred):
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
    np.float64(0.2857142857142857)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    confusion = confusion_matrix(y_true, y_pred)
    return 1 - np.diagonal(confusion).sum() / confusion.sum()


def kendalls_tau(y_true, y_pred):
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
    np.float64(0.8140915784106943)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    corr, pvalue = scipy.stats.kendalltau(y_true, y_pred)
    return corr


def weighted_kappa(y_true, y_pred):
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
    np.float64(0.7586206896551724)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    n_class = cm.shape[0]
    costs = np.reshape(np.tile(range(n_class), n_class), (n_class, n_class))
    costs = np.abs(costs - np.transpose(costs))
    f = 1 - costs

    n = cm.sum()
    x = cm / n

    r = x.sum(axis=1)  # Row sum
    s = x.sum(axis=0)  # Col sum
    Ex = r.reshape(-1, 1) * s
    po = (x * f).sum()
    pe = (Ex * f).sum()
    return (po - pe) / (1 - pe)


def spearmans_rho(y_true, y_pred):
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
    np.float64(0.9165444688834581)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    n = len(y_true)
    num = (
        (y_true - np.repeat(np.mean(y_true), n))
        * (y_pred - np.repeat(np.mean(y_pred), n))
    ).sum()
    div = np.sqrt(
        (pow(y_true - np.repeat(np.mean(y_true), n), 2)).sum()
        * (pow(y_pred - np.repeat(np.mean(y_pred), n), 2)).sum()
    )

    if num == 0:
        return 0
    else:
        return num / div


def ranked_probability_score(y_true, y_proba):
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
    np.float64(0.5068750000000001)

    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    y_oh = np.zeros(y_proba.shape)
    y_oh[np.arange(len(y_true)), y_true] = 1

    y_oh = y_oh.cumsum(axis=1)
    y_proba = y_proba.cumsum(axis=1)

    rps = 0
    for i in range(len(y_true)):
        if y_true[i] in np.arange(y_proba.shape[1]):
            rps += np.power(y_proba[i] - y_oh[i], 2).sum()
        else:
            rps += 1
    return rps / len(y_true)


def accuracy_off1(y_true, y_pred, labels=None):
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
    np.float64(0.8571428571428571)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if labels is None:
        labels = np.unique(y_true)

    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    n = conf_mat.shape[0]
    mask = np.eye(n, n) + np.eye(n, n, k=1), +np.eye(n, n, k=-1)
    correct = mask * conf_mat

    return 1.0 * np.sum(correct) / np.sum(conf_mat)


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
