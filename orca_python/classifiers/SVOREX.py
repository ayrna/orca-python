"""Support Vector for Ordinal Regression (Explicit constraints) (SVOREX)."""

from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from orca_python.classifiers.svorex import svorex


class SVOREX(BaseEstimator, ClassifierMixin):
    """Support Vector for Ordinal Regression (Explicit constraints).

    This class derives from the Algorithm Class and implements the SVOREX method.
    This class uses SVOREX implementation by W. Chu et al
    (http://www.gatsby.ucl.ac.uk/~chuwei/svor.htm).

    Parameters
    ----------
    C : float, default=1
        Set the parameter C.

    kernel : str, default="gaussian"
        Set type of kernel function.
        - gaussian: use gaussian kernel
        - linear: use imbalanced Linear kernel
        - poly: use Polynomial kernel with order p

    degree : int, default=2
        Set degree in kernel function.

    tol : float, default=0.001
        Set tolerance of termination criterion.

    kappa : float, default=1
        Set kappa value.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array that contains all different class labels found in the original dataset.

    model_ : object
        Fitted estimator.

    References
    ----------
    .. [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data Engineering,
           Vol. 28. Issue 1, 2016, https://doi.org/10.1109/TKDE.2015.2457911

    .. [2] W. Chu and S. S. Keerthi, "Support Vector Ordinal Regression", Neural
           Computation, vol. 19, no. 3, pp. 792-815, 2007,
           http://10.1162/neco.2007.19.3.792

    """

    _parameter_constraints: dict = {
        "C": [Interval(Real, 0.0, None, closed="neither")],
        "kernel": [
            StrOptions(
                {
                    "gaussian",
                    "linear",
                    "poly",
                }
            )
        ],
        "degree": [Interval(Integral, 0, None, closed="left")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "kappa": [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(self, C=1.0, kernel="gaussian", degree=2, tol=0.001, kappa=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.tol = tol
        self.kappa = kappa

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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        arg = ""
        # Prepare the kernel type arguments
        if self.kernel == "linear":
            arg = "-L"
        elif self.kernel == "poly":
            arg = "-P {}".format(self.degree)

        # Fit the model
        options = "svorex {} -T {} -K {} -C {}".format(
            arg, str(self.tol), str(self.kappa), str(self.C)
        )
        self.model_ = svorex.fit(y.tolist(), X.tolist(), options)

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
            If the input is invalid.

        """
        # Check is fit had been called
        check_is_fitted(self, ["model_"])

        # Input validation
        X = check_array(X)

        y_pred = np.array(svorex.predict(X.tolist(), self.model_))

        return y_pred
