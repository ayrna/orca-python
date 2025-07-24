"""Reduction from ordinal regression to binary SVM (REDSVM)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from orca_python.classifiers.libsvmRank.python import svm


class REDSVM(BaseEstimator, ClassifierMixin):
    """Reduction from ordinal regression to binary SVM classifiers.

    The configuration used is the identity coding matrix, the absolute cost matrix and
    the standard binary soft-margin SVM. This class uses libsvm-rank-2.81
    implementation: (http://www.work.caltech.edu/~htlin/program/libsvm/)

    Parameters
    ----------
    C : float, default=1
        Set the parameter C.

    kernel : str, default="rbf"
        Set type of kernel function.
        - linear: u'*v
        - polynomial: (gamma*u'*v + coef0)^degree
        - rbf: exp(-gamma*|u-v|^2)
        - sigmoid: tanh(gamma*u'*v + coef0)
        - stump: -|u-v|_1 + coef0
        - perceptron: -|u-v|_2 + coef0
        - laplacian: exp(-gamma*|u-v|_1)
        - exponential: exp(-gamma*|u-v|_2)
        - precomputed: kernel values in training_instance_matrix

    degree : int, default=3
        Set degree in kernel function.

    gamma : {'scale', 'auto'} or float, default=1.0
        Kernel coefficient determining the influence of individual training samples:
        - 'scale': 1 / (n_features * X.var())
        - 'auto': 1 / n_features
        - float: Must be non-negative.

    coef0 : float, default=0
        Set coef0 in kernel function.

    shrinking : bool, default=True
        Set whether to use the shrinking heuristics.

    tol : float, default=0.001
        Set tolerance of termination criterion.

    cache_size : int, default=100
        Set cache memory size in MB.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array that contains all different class labels found in the original dataset.

    model_ : object
        Fitted estimator.

    References
    ----------
    .. [1] H.-T. Lin and L. Li, "Reduction from cost-sensitive ordinal ranking to
           weighted binary classification", Neural Computation, vol. 24, no. 5, pp.
           1329-1367, 2012, http://10.1162/NECO_a_00265

    .. [2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data Engineering,
           Vol. 28. Issue 1, 2016,
           https://doi.org/10.1109/TKDE.2015.2457911

    """

    _parameter_constraints: dict = {
        "C": [Interval(Real, 0.0, None, closed="neither")],
        "kernel": [
            StrOptions(
                {
                    "linear",
                    "poly",
                    "rbf",
                    "sigmoid",
                    "stump",
                    "perceptron",
                    "laplacian",
                    "exponential",
                    "precomputed",
                }
            )
        ],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "gamma": [
            StrOptions({"scale", "auto"}),
            Interval(Real, 0.0, None, closed="neither"),
        ],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "shrinking": ["boolean"],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "cache_size": [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(
        self,
        C=1,
        kernel="rbf",
        degree=3,
        gamma="auto",
        coef0=0,
        shrinking=True,
        tol=0.001,
        cache_size=100,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model with the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or data has wrong format.

        """
        # Additional strict validation for boolean parameters
        if not isinstance(self.shrinking, bool):
            raise ValueError(
                f"The 'shrinking' parameter must be of type bool. "
                f"Got {type(self.shrinking).__name__} instead."
            )

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Set default gamma value if not specified
        gamma_value = self.gamma
        if self.gamma == "auto":
            gamma_value = 1.0 / X.shape[1]
        elif self.gamma == "scale":
            gamma_value = 1.0 / (X.shape[1] * X.var())

        # Map kernel type
        kernel_type_mapping = {
            "linear": 0,
            "poly": 1,
            "rbf": 2,
            "sigmoid": 3,
            "stump": 4,
            "perceptron": 5,
            "laplacian": 6,
            "exponential": 7,
            "precomputed": 8,
        }
        kernel_type = kernel_type_mapping[self.kernel]

        # Fit the model
        options = "-s 5 -t {} -d {} -g {} -r {} -c {} -m {} -e {} -h {} -q".format(
            str(kernel_type),
            str(self.degree),
            str(gamma_value),
            str(self.coef0),
            str(self.C),
            str(self.cache_size),
            str(self.tol),
            str(1 if self.shrinking else 0),
        )
        self.model_ = svm.fit(y.tolist(), X.tolist(), options)

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test patterns array, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        """
        # Check is fit had been called
        check_is_fitted(self, ["model_"])

        # Input validation
        X = check_array(X)

        y_pred = np.array(svm.predict(X.tolist(), self.model_))

        return y_pred
