"""Neural Network with Ordered Partitions (NNOP)."""

import math as math
from numbers import Integral, Real

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class NNOP(BaseEstimator, ClassifierMixin):
    """Neural Network with Ordered Partitions (NNOP).

    This model considers the OrderedPartitions coding scheme for the labels and a rule
    for decisions based on the first node whose output is higher than a predefined
    threshold (T=0.5, in our experiments). The model has one hidden layer with hiddenN
    neurons and one output layer with as many neurons as the number of classes minus
    one.

    The learning is based on iRProp+ algorithm and the implementation provided by
    Roberto Calandra in his toolbox Rprop Toolbox for MATLAB:
    http://www.ias.informatik.tu-darmstadt.de/Research/RpropToolbox

    The model is adjusted by minimizing mean squared error. A regularization parameter
    "lambda" is included based on L2, and the number of iterations is specified by the
    "iterations" parameter.

    Parameters
    ----------
    epsilon_init : float, default=0.5
        Range for initializing the weights.

    n_hidden : int, default=50
        Number of hidden neurons of the model.

    max_iter : int, default=500
        Number of iterations for fmin_l_bfgs_b algorithm.

    lambda_value : float, default=0.01
        Regularization parameter.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array that contains all different class labels found in the original dataset.

    theta1_ : ndarray of shape (n_hidden, n_features + 1)
        Hidden layer weights (with bias).

    theta2_ : ndarray of shape (n_classes - 1, n_hidden + 1)
        Output layer weights.

    Notes
    -----
    This file is part of ORCA: https://github.com/ayrna/orca

    References
    ----------
    .. [1] J. Cheng, Z. Wang, and G. Pollastri, "A neural network approach to ordinal
           regression," in Proc. IEEE Int. Joint Conf. Neural Netw. (IEEE World Congr.
           Comput. Intell.), 2008, pp. 1279-1284.

    .. [2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data
           Engineering, Vol. 28. Issue 1, 2016,
           http://dx.doi.org/10.1109/TKDE.2015.2457911

    Copyright
    ---------
    This software is released under the The GNU General Public License v3.0 licence
    available at http://www.gnu.org/licenses/gpl-3.0.html

    Authors
    -------
    Pedro Antonio Gutiérrez, María Pérez Ortiz, Javier Sánchez Monedero

    Citation
    --------
    If you use this code, please cite the associated paper
    http://www.uco.es/grupos/ayrna/orreview

    """

    _parameter_constraints: dict = {
        "epsilon_init": [Interval(Real, 0.0, None, closed="neither")],
        "n_hidden": [Interval(Integral, 1, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "lambda_value": [Interval(Real, 0.0, None, closed="neither")],
    }

    def __init__(self, epsilon_init=0.5, n_hidden=50, max_iter=500, lambda_value=0.01):
        self.epsilon_init = epsilon_init
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.lambda_value = lambda_value

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model with the training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples
            and n_features is the number of features.

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
        if (
            self.epsilon_init < 0
            or self.n_hidden < 1
            or self.max_iter < 1
            or self.lambda_value < 0
        ):
            return None

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Aux variables
        y = y[:, np.newaxis]
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # Recode y to Y using ordinalPartitions coding
        Y = 1 * (
            np.tile(y, (1, n_classes))
            <= np.tile(np.arange(1, n_classes + 1)[np.newaxis, :], (n_samples, 1))
        )

        # Hidden layer weights (with bias)
        initial_theta1 = self._rand_initialize_weights(n_features + 1, self.n_hidden)
        # Output layer weights
        initial_theta2 = self._rand_initialize_weights(self.n_hidden + 1, n_classes - 1)

        # Pack parameters
        initial_nn_params = np.concatenate(
            (initial_theta1.flatten(order="F"), initial_theta2.flatten(order="F")),
            axis=0,
        )[:, np.newaxis]

        results_optimization = scipy.optimize.fmin_l_bfgs_b(
            func=self._nnop_cost_function,
            x0=initial_nn_params.ravel(),
            args=(n_features, self.n_hidden, n_classes, X, Y, self.lambda_value),
            fprime=None,
            factr=1e3,
            maxiter=self.max_iter,
            iprint=-1,
        )

        self.nn_params = results_optimization[0]
        # Unpack the parameters
        theta1, theta2 = self._unpack_parameters(
            self.nn_params, n_features, self.n_hidden, n_classes
        )
        self.theta1_ = theta1
        self.theta2_ = theta2

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test patterns array, where n_samples is the number of samples and n_features
            is the number of features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.append(np.ones((n_samples, 1)), np.matmul(a1, self.theta1_.T), axis=1)

        a2 = 1.0 / (1.0 + np.exp(-z2))
        projected = np.matmul(a2, self.theta2_.T)
        projected = 1.0 / (1.0 + np.exp(-projected))

        a3 = np.multiply(
            np.where(np.append(projected, np.ones((n_samples, 1)), axis=1) > 0.5, 1, 0),
            np.tile(np.arange(1, n_classes + 1), (n_samples, 1)),
        )
        a3[np.where(a3 == 0)] = n_classes + 1
        y_pred = a3.min(axis=1)

        return y_pred

    def _unpack_parameters(self, nn_params, n_features, n_hidden, n_classes):
        """Get theta1 and theta2 back from nn_params.

        Parameters
        ----------
        nn_params : ndarray of shape ((n_features+1)*n_hidden + n_hidden +
                                      (n_classes-1))
            Array that is a column vector. It stores the values of theta1, theta2 and
            thresholds_param, all of them together in an array in this order.

        n_features : int
            Number of nodes in the input layer of the neural network model.

        n_hidden : int
            Number of nodes in the hidden layer of the neural network model.

        n_classes : int
            Number of classes.

        Returns
        -------
        theta1 : ndarray of shape (n_hidden, n_features + 1)
            The weights between the input layer and the hidden layer (with biases
            included).

        theta2 : ndarray of shape (n_classes - 1, n_hidden + 1)
            The weights between the hidden layer and the output layer.

        """
        n_theta1 = n_hidden * (n_features + 1)
        theta1 = np.reshape(
            nn_params[0:n_theta1], (n_hidden, (n_features + 1)), order="F"
        )

        theta2 = np.reshape(
            nn_params[n_theta1:], (n_classes - 1, n_hidden + 1), order="F"
        )

        return theta1, theta2

    def _rand_initialize_weights(self, L_in, L_out):
        """Initialize layer weights randomly.

        Randomly initialize the weights of a layer with L_in incoming connections and
        L_out outgoing connections.

        Parameters
        ----------
        L_in : int
            Number of inputs of the layer.

        L_out : int
            Number of outputs of the layer.

        Returns
        -------
        W : ndarray of shape (L_out, L_in)
            Array with the weights of each synaptic relationship between nodes.

        """
        W = np.random.rand(L_out, L_in) * 2 * self.epsilon_init - self.epsilon_init

        return W

    def _nnop_cost_function(
        self, nn_params, n_features, n_hidden, n_classes, X, Y, lambda_value
    ):
        """Implement the cost function and obtain the corresponding derivatives.

        Parameters
        ----------
        nn_params : ndarray of shape ((n_features+1)*n_hidden + n_hidden)
            Array that is a column vector. It stores the values of Theta1 and Theta2,
            all of them together in an array in this order.

        n_features : int
            Number of nodes in the input layer of the neural network model.

        n_hidden : int
            Number of nodes in the hidden layer of the neural network model.

        n_classes : int
            Number of classes.

        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training patterns array, where n_samples is the number of samples and
            n_features is the number of features

        Y : array-like of shape (n_samples,)
            Target vector relative to X

        lambda_value : float
            Regularization parameter.

        Returns
        -------
        J : float
            Matrix with cost function (updated weight matrix).

        grad : ndarray
            Array with the error gradient of each weight of each layer.

        """
        # Unroll all the parameters
        theta1, theta2 = self._unpack_parameters(
            nn_params, n_features, n_hidden, n_classes
        )

        # Setup some useful variables
        n_samples = np.size(X, 0)

        # Neural Network model
        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, theta1.T)
        a2 = np.append(np.ones((n_samples, 1)), 1.0 / (1.0 + np.exp(-z2)), axis=1)
        z3 = np.matmul(a2, theta2.T)
        h = np.append(1.0 / (1.0 + np.exp(-z3)), np.ones((n_samples, 1)), axis=1)

        # Final output
        out = h

        # Calculate penalty (regularización L2)
        p = np.sum((theta1[:, 1:] ** 2).sum() + (theta2[:, 1:] ** 2).sum())

        # MSE
        J = np.sum((out - Y) ** 2).sum() / (2 * n_samples) + lambda_value * p / (
            2 * n_samples
        )

        # MSE
        error_der = out - Y

        # Calculate sigmas
        sigma3 = np.multiply(np.multiply(error_der, h), (1 - h))
        sigma3 = sigma3[:, :-1]

        sigma2 = np.multiply(np.multiply(np.matmul(sigma3, theta2), a2), (1 - a2))
        sigma2 = sigma2[:, 1:]

        # Accumulate gradients
        delta_1 = np.matmul(sigma2.T, a1)
        delta_2 = np.matmul(sigma3.T, a2)

        # Calculate regularized gradient
        p1 = (lambda_value / n_samples) * np.concatenate(
            (np.zeros((np.size(theta1, axis=0), 1)), theta1[:, 1:]), axis=1
        )
        p2 = (lambda_value / n_samples) * np.concatenate(
            (np.zeros((np.size(theta2, axis=0), 1)), theta2[:, 1:]), axis=1
        )
        theta1_grad = delta_1 / n_samples + p1
        theta2_grad = delta_2 / n_samples + p2

        # Unroll gradients
        grad = np.concatenate(
            (theta1_grad.flatten(order="F"), theta2_grad.flatten(order="F")), axis=0
        )

        return J, grad
