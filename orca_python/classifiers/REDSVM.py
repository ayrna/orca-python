"""Reduction from ordinal regression to binary SVM (REDSVM)."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# from .libsvmRank.python import svm

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
    
    kernel : int, default=2
        Set type of kernel function.
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
        4 -- stump: -|u-v|_1 + coef0
        5 -- perceptron: -|u-v|_2 + coef0
        6 -- laplacian: exp(-gamma*|u-v|_1)
        7 -- exponential: exp(-gamma*|u-v|_2)
        8 -- precomputed kernel (kernel values in training_instance_matrix)

    degree : int, default=3
        Set degree in kernel function.

    gamma : float, default=1/n_features
        Set gamma in kernel function.

    coef0 : float, default=0
        Set coef0 in kernel function.

    shrinking : int, default=1
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

    def __init__(
        self,
        C=1,
        kernel=2,
        degree=3,
        gamma=None,
        coef0=0,
        shrinking=1,
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

        # Set the default g value if necessary
        if self.gamma is None:
            self.gamma = 1 / np.size(X, 1)

        # Fit the model
        options = "-s 5 -t {} -d {} -g {} -r {} -c {} -m {} -e {} -h {} -q".format(
            str(self.kernel),
            str(self.degree),
            str(self.gamma),
            str(self.coef0),
            str(self.C),
            str(self.cache_size),
            str(self.tol),
            str(self.shrinking),
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

        y_pred = svm.predict(X.tolist(), self.model_)

        return y_pred
