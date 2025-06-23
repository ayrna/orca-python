"""Support Vector for Ordinal Regression (Explicit constraints) (SVOREX)."""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# from .svorex import svorex
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

    kernel : int, default=0
        Set type of kernel function.
        0 -- gaussian: use gaussian kernel
        1 -- linear: use imbalanced Linear kernel
        2 -- polynomial: use Polynomial kernel with order p

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

    def __init__(self, C=1.0, kernel=0, degree=2, tol=0.001, kappa=1):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.tol = tol
        self.kappa = kappa

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
        if self.kernel == 1:
            arg = "-L"
        elif self.kernel == 2:
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

        y_pred = svorex.predict(X.tolist(), self.model_)

        return y_pred
