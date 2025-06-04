# encoding: utf-8
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# from .svorex import svorex
from orca_python.classifiers.svorex import svorex


class SVOREX(BaseEstimator, ClassifierMixin):
    """
        SVOREX Support Vector for Ordinal Regression (Explicit constraints)
    This class derives from the Algorithm Class and implements the
    SVOREX method. This class uses SVOREX implementation by
    W. Chu et al (http://www.gatsby.ucl.ac.uk/~chuwei/svor.htm)

                SVOREX methods:
                        fit                        - Fits a model from training data
                        predict                    - Performs label prediction

        References:
         [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
             F. Fernández-Navarro and C. Hervás-Martínez
             Ordinal regression methods: survey and experimental study
             IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue 1
             2016
             http://dx.doi.org/10.1109/TKDE.2015.2457911
         [2] W. Chu and S. S. Keerthi, Support Vector Ordinal Regression,
             Neural Computation, vol. 19, no. 3, pp. 792–815, 2007.
             http://10.1162/neco.2007.19.3.792

        Model Parameters:
                kernel:
                        0 -- gaussian: use gaussian kernel (default)
                        1 -- linear:   use imbalanced Linear kernel
                        2 -- polynomial: (Use parameter p to change the order) use Polynomial kernel with order p
                tol: set Tolerance (default 0.001)
                kappa: set kappa value (default 1)
                C: set C value (default  1)
    """

    # Set parameters values
    def __init__(self, C=1.0, kernel=0, degree=2, tol=0.001, kappa=1):

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.tol = tol
        self.kappa = kappa

    def fit(self, X, y):
        """
        Fit the model with the training data

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape (n_samples, n_features)
                Training patterns array, where n_samples is the number of samples
                and n_features is the number of features

        y: array-like, shape (n_samples)
                Target vector relative to X

        Returns
        -------

        self: object
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
        """
        Performs classification on samples in X

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------

        y_pred : array, shape (n_samples,)
                Class labels for samples in X.
        """

        # Check is fit had been called
        check_is_fitted(self, ["model_"])

        # Input validation
        X = check_array(X)

        y_pred = svorex.predict(X.tolist(), self.model_)

        return y_pred
