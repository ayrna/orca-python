"""Metrics for ordinal classification."""

from __future__ import division

import numpy as np
import scipy.stats
from sklearn.metrics import confusion_matrix, recall_score


def ccr(y_true, y_pred):
    """Calculate the Correctly Classified Ratio.

    Also named Accuracy, it's the percentage of well classified patterns among all
    patterns from a set.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    ccr : float
        Correctly classified ratio.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import ccr
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 1])
    >>> ccr(y_true, y_pred)
    0.5714285714285714

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return np.count_nonzero(y_true == y_pred) / float(len(y_true))


def amae(y_true, y_pred):
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
    amae : float
        Average mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import amae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> amae(y_true, y_pred)
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


def gm(y_true, y_pred):
    """Calculate the Geometric mean of the sensitivity (accuracy) for each class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    gm : float
        Geometric mean of the sensitivities.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import gm
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> gm(y_true, y_pred)
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
    >>> from orca_python.metrics import gmsec
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


def mae(y_true, y_pred):
    """Calculate the Mean Absolute Error.

    Average absolute deviation of the predicted class from the actual true class.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels.

    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels.

    Returns
    -------
    mae : float
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mae(y_true, y_pred)
    np.float64(0.2857142857142857)

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return abs(y_true - y_pred).sum() / len(y_true)


def mmae(y_true, y_pred):
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
    mmae : float
        Maximum mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mmae
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mmae(y_true, y_pred)
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


def ms(y_true, y_pred):
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
    ms : float
        Minimum sensitivity.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import ms
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> ms(y_true, y_pred)
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


def mze(y_true, y_pred):
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
    mze : float
        Mean zero-one error.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import mze
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> mze(y_true, y_pred)
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


def tkendall(y_true, y_pred):
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
    tkendall : float
        Kendall's tau.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import tkendall
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> tkendall(y_true, y_pred)
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


def wkappa(y_true, y_pred):
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
    wkappa : float
        Weighted Kappa.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import wkappa
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> wkappa(y_true, y_pred)
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


def spearman(y_true, y_pred):
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
    spearman : float
        Spearman rank correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import spearman
    >>> y_true = np.array([0, 0, 1, 2, 3, 0, 0])
    >>> y_pred = np.array([0, 1, 1, 2, 3, 0, 1])
    >>> spearman(y_true, y_pred)
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


def rps(y_true, y_proba):
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
    rps : float
        The ranked probability score.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.metrics import rps
    >>> y_true = np.array([0, 0, 3, 2])
    >>> y_pred = np.array(
    ...     [[0.2, 0.4, 0.2, 0.2],
    ...      [0.7, 0.1, 0.1, 0.1],
    ...      [0.5, 0.05, 0.1, 0.35],
    ...      [0.1, 0.05, 0.65, 0.2]])
    >>> rps(y_true, y_pred)
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
    >>> from orca_python.metrics import accuracy_off1
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
