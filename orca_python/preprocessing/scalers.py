"""Data scaling functions."""

from scipy import sparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_array


def _validate_and_align(X_train, X_test):
    """Validate arrays as numeric 2D matrices and ensure matching feature counts.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix used for validation reference.

    X_test : array-like of shape (m_samples, n_features), optional
        Test feature matrix to validate against training matrix.

    Returns
    -------
    (X_train_valid, X_test_valid) : tuple
        Validated arrays. X_test_valid is None if X_test is None.

    Raises
    ------
    ValueError
        If X_test has different number of features than X_train.

    """
    X_train = check_array(X_train, accept_sparse=True, dtype="numeric")
    if X_test is not None:
        X_test = check_array(X_test, accept_sparse=True, dtype="numeric")
        if X_test.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"X_test has {X_test.shape[1]} features but X_train has {X_train.shape[1]}."
            )
    return X_train, X_test


def apply_scaling(X_train, X_test=None, method=None, return_transformer=False):
    """Apply normalization or standardization to the input data.

    The preprocessing is fit on the training data and then applied to both
    training and test data (if provided).

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Feature matrix used specifically for model training.

    X_test : array-like of shape (m_samples, n_features), optional
        Feature matrix used for model evaluation and prediction.

    method : {"norm", "std", None}, optional
        - "norm": Min-Max scaling to [0, 1]
        - "std" : Standardization (mean=0, std=1)
        - None  : No preprocessing

    return_transformer : bool, default=False
        If True, also return the fitted scaling object.

    Returns
    -------
    (X_train_scaled, X_test_scaled) or (X_train_scaled, X_test_scaled, scaler)
        Scaled arrays; X_test_scaled is None if X_test is None.
        If return_transformer=True and method=None, scaler is None.

    Raises
    ------
    ValueError
        If an unknown scaling method is specified.

    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing.scalers import apply_scaling
    >>> X_train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    >>> X_test = np.array([[2.5, 5.0]])
    >>> X_train_scaled, X_test_scaled = apply_scaling(X_train, X_test, method="norm")
    >>> X_train_scaled
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    >>> X_test_scaled
    array([[0.75, 0.75]])
    >>> X_train_scaled, X_test_scaled = apply_scaling(X_train, X_test, method="std")
    >>> X_train_scaled.round(3)
    array([[-1.225, -1.225],
           [ 0.   ,  0.   ],
           [ 1.225,  1.225]])
    >>> X_test_scaled.round(3)
    array([[0.612, 0.612]])

    """
    if method is None:
        return (X_train, X_test, None) if return_transformer else (X_train, X_test)

    if not isinstance(method, str):
        raise ValueError("Scaling method must be a string or None.")

    key = method.lower()
    if key == "norm":
        return minmax_scale(X_train, X_test, return_transformer)
    elif key == "std":
        return standardize(X_train, X_test, return_transformer)
    else:
        raise ValueError(
            f"Unknown scaling method '{method}'. Valid options: 'norm', 'std', None."
        )


def minmax_scale(X_train, X_test=None, return_transformer=False):
    """Scale features to a fixed range between 0 and 1.

    Fits scaling parameters on training data and applies the same transformation
    to both training and test sets.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature matrix used to fit scaling parameters.

    X_test : array-like of shape (m_samples, n_features), optional
        Test feature matrix to transform using fitted parameters.

    return_transformer : bool, default=False
        If True, also return the fitted scaling object.

    Returns
    -------
    (X_train_scaled, X_test_scaled) or (X_train_scaled, X_test_scaled, scaler)
        Scaled arrays; X_test_scaled is None if X_test is None.

    Raises
    ------
    ValueError
        If X_test has a different number of features than X_train.

    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing.scalers import minmax_scale
    >>> X_train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    >>> X_test = np.array([[2.5, 5.0]])
    >>> X_train_scaled, X_test_scaled = minmax_scale(X_train, X_test)
    >>> X_train_scaled
    array([[0. , 0. ],
           [0.5, 0.5],
           [1. , 1. ]])
    >>> X_test_scaled
    array([[0.75, 0.75]])

    """
    X_train, X_test = _validate_and_align(X_train, X_test)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    if return_transformer:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled


def standardize(X_train, X_test=None, return_transformer=False):
    """Standardize features to have zero mean and unit variance.

    Fits scaling parameters on training data and applies the same transformation
    to both training and test sets. For sparse matrices, centering is disabled
    to preserve sparsity.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Feature matrix used specifically for model training.

    X_test : array-like of shape (m_samples, n_features), optional
        Test feature matrix to transform using fitted parameters.

    return_transformer: bool, default=False
        If True, also return the fitted scaling object.

    Returns
    -------
    (X_train_scaled, X_test_scaled) or (X_train_scaled, X_test_scaled, scaler)
        Scaled arrays; X_test_scaled is None if X_test is None.

    Raises
    ------
    ValueError
        If X_test has a different number of features than X_train.

    Examples
    --------
    >>> import numpy as np
    >>> from preprocessing.scalers import standardize
    >>> X_train = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    >>> X_test = np.array([[2.5, 5.0]])
    >>> X_train_scaled, X_test_scaled = standardize(X_train, X_test)
    >>> X_train_scaled.round(3)
    array([[-1.225, -1.225],
           [ 0.   ,  0.   ],
           [ 1.225,  1.225]])
    >>> X_test_scaled.round(3)
    array([[0.612, 0.612]])

    """
    X_train, X_test = _validate_and_align(X_train, X_test)

    scaler = (
        StandardScaler(with_mean=False)
        if sparse.issparse(X_train)
        else StandardScaler()
    )
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    if return_transformer:
        return X_train_scaled, X_test_scaled, scaler
    else:
        return X_train_scaled, X_test_scaled
