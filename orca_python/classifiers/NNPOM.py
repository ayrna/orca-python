# encoding: utf-8
from re import T
import numpy as np
import math as math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import scipy


class NNPOM(BaseEstimator, ClassifierMixin):
    """

    NNPOM Neural Network based on Proportional Odd Model (NNPOM). This
            class implements a neural network model for ordinal regression. The
            model has one hidden layer with hidden_n neurons and one outputlayer
            with only one neuron but as many threshold as the number of classes
            minus one. The standard POM model is applied in this neuron to have
            probabilistic outputs. The learning is based on iRProp+ algorithm and
            the implementation provided by Roberto Calandra in his toolbox Rprop
            Toolbox for {MATLAB}:
            http://www.ias.informatik.tu-darmstadt.de/Research/RpropToolbox
            The model is adjusted by minimizing cross entropy. A regularization
            parameter "lambda" is included based on L2, and the number of
            iterations is specified by the "iterations" parameter.

            NNPOM public methods:
                    fit						- Fits a model from training data
                    predict					- Performs label prediction

            References:
                    [1] P. McCullagh, Regression models for ordinal data,  Journal of
                            the Royal Statistical Society. Series B (Methodological), vol. 42,
                            no. 2, pp. 109–142, 1980.
                    [2] M. J. Mathieson, Ordinal models for neural networks, in Proc.
                            3rd Int. Conf. Neural Netw. Capital Markets, 1996, pp.
                            523-536.
                    [3] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
                            F. Fernández-Navarro and C. Hervás-Martínez
                            Ordinal regression methods: survey and experimental study
                            IEEE Transactions on Knowledge and Data Engineering, Vol. 28.
                            Issue 1, 2016
                            http://dx.doi.org/10.1109/TKDE.2015.2457911

            This file is part of ORCA: https://github.com/ayrna/orca
            Original authors: Pedro Antonio Gutiérrez, María Pérez Ortiz, Javier Sánchez Monedero
            Citation: If you use this code, please cite the associated paper http://www.uco.es/grupos/ayrna/orreview
            Copyright:
                    This software is released under the The GNU General Public License v3.0 licence
                    available at http://www.gnu.org/licenses/gpl-3.0.html


            NNPOM properties:
                    epsilon_init				- Range for initializing the weights.
                    n_hidden					- Number of hidden neurons of the
                                                                            model.
                    max_iter					- Number of iterations for fmin_l_bfgs_b
                                                                            algorithm.
                    lambda_value				- Regularization parameter.
                    theta1_						- Hidden layer weigths (with bias)
                    theta2_						- Output layer weigths (without bias, the biases will be the thresholds)
                    thresholds_					- Class thresholds parameters
                    n_classes_					- Number of labels in the problem
                    n_samples_					- Number of samples of X (train patterns array).

    """

    # Constructor of class NNPOM (set parameters values).
    def __init__(self, epsilon_init=0.5, n_hidden=50, max_iter=500, lambda_value=0.01):

        self.epsilon_init = epsilon_init
        self.n_hidden = n_hidden
        self.max_iter = max_iter
        self.lambda_value = lambda_value

    # --------Main functions (Public Access)--------

    def fit(self, X, y):
        """

        Trains the model for the model NNPOM method with TRAIN data.
        Returns the projection of patterns (only valid for threshold models) and the predicted labels.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape (n_samples, n_features)
                Training patterns array, where n_samples is the number of samples
                and n_features is the number of features

        y: array-like, shape (n_samples)
                Target vector relative to X

        Returns
        -------

        self: The object NNPOM.

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
        n_classes = np.size(np.unique(y))
        n_samples = X.shape[0]

        # Recode y to Y using nominal coding
        Y = 1 * (
            np.tile(y, (1, n_classes))
            == np.tile(np.arange(1, n_classes + 1)[np.newaxis, :], (n_samples, 1))
        )

        # Hidden layer weigths (with bias)
        initial_theta1 = self._rand_initialize_weights(
            n_features + 1, self.get_n_hidden()
        )
        # Output layer weigths (without bias, the biases will be the thresholds)
        initial_theta2 = self._rand_initialize_weights(self.get_n_hidden(), 1)
        # Class thresholds parameters
        initial_thresholds = self._rand_initialize_weights((n_classes - 1), 1)

        # Pack parameters
        initial_nn_params = np.concatenate(
            (
                initial_theta1.flatten(order="F"),
                initial_theta2.flatten(order="F"),
                initial_thresholds.flatten(order="F"),
            ),
            axis=0,
        )[:, np.newaxis]

        results_optimization = scipy.optimize.fmin_l_bfgs_b(
            func=self._nnpom_cost_function,
            x0=initial_nn_params.ravel(),
            args=(n_features, self.n_hidden, n_classes, X, Y, self.lambda_value),
            fprime=None,
            factr=1e3,
            maxiter=self.max_iter,
            iprint=-1,
        )

        self.nn_params = results_optimization[0]

        # Unpack the parameters
        theta1, theta2, thresholds_param = self._unpack_parameters(
            self.nn_params, n_features, self.n_hidden, n_classes
        )

        self.theta1_ = theta1
        self.theta2_ = theta2
        self.thresholds_ = self._convert_thresholds(thresholds_param, n_classes)
        self.n_classes_ = n_classes
        self.n_samples_ = n_samples

        return self

    def predict(self, X):
        """

        Predicts labels of TEST patterns labels. The object needs to be fitted to the data first.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape (n_samples, n_features)
                test patterns array, where n_samples is the number of samples
                and n_features is the number of features

        Returns
        -------

        predicted: {array-like, sparse matrix}, shape (n_samples,)
                Vector array with predicted values for each pattern of test patterns.

        """

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        n_samples = X.shape[0]

        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, self.theta1_.T)
        a2 = 1.0 / (1.0 + np.exp(-z2))
        projected = np.matmul(a2, self.theta2_.T)

        z3 = np.tile(self.thresholds_, (n_samples, 1)) - np.tile(
            projected, (1, self.n_classes_ - 1)
        )
        a3T = 1.0 / (1.0 + np.exp(-z3))
        a3 = np.append(a3T, np.ones((n_samples, 1)), axis=1)
        a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]
        y_pred = a3.argmax(1) + 1

        return y_pred

    # --------Getters & Setters (Public Access)--------

    # Getter & Setter of "epsilon_init"
    def get_epsilon_init(self):
        """

        This method returns the value of the variable self.epsilon_init.
        self.epsilon_init contains the value of epsilon, which is the initialization range of the weights.

        """

        return self.epsilon_init

    def set_epsilon_init(self, epsilon_init):
        """

        This method modify the value of the variable self.epsilon_init.
        This is replaced by the value contained in the epsilon_init variable passed as an argument.

        """

        self.epsilon_init = epsilon_init

    # Getter & Setter of "n_hidden"
    def get_n_hidden(self):
        """

        This method returns the value of the variable self.n_hidden.
        self.n_hidden contains the number of nodes/neurons in the hidden layer.

        """

        return self.n_hidden

    def set_n_hidden(self, n_hidden):
        """

        This method modify the value of the variable self.n_hidden.
        This is replaced by the value contained in the n_hidden variable passed as an argument.

        """

        self.n_hidden = n_hidden

    # Getter & Setter of "max_iter"
    def get_max_iter(self):
        """

        This method returns the value of the variable self.max_iter.
        self.max_iter contains the number of iterations.

        """

        return self.max_iter

    def set_max_iter(self, max_iter):
        """

        This method modify the value of the variable self.max_iter.
        This is replaced by the value contained in the max_iter variable passed as an argument.

        """

        self.max_iter = max_iter

    # Getter & Setter of "lambda_value"
    def get_lambda_value(self):
        """

        This method returns the value of the variable self.lambda_value.
        self.lambda_value contains the Lambda parameter used in regularization.

        """

        return self.lambda_value

    def set_lambda_value(self, lambda_value):
        """

        This method modify the value of the variable self.lambda_value.
        This is replaced by the value contained in the lambda_value variable passed as an argument.

        """

        self.lambda_value = lambda_value

    # Getter & Setter of "theta1"
    def get_theta1(self):
        """

        This method returns the value of the variable self.theta1_.
        self.theta1_ contains an array with the weights of the hidden layer (with biases included).

        """

        return self.theta1_

    def set_theta1(self, theta1):
        """

        This method modify the value of the variable self.theta1_.
        This is replaced by the value contained in the theta1 variable passed as an argument.

        """

        self.theta1_ = theta1

    # Getter & Setter of "theta2"
    def get_theta2(self):
        """

        This method returns the value of the variable self.theta2_.
        self.theta2_ contains an array with output layer weigths (without bias, the biases will be the thresholds)

        """

        return self.theta2_

    def set_theta2(self, theta2):
        """

        This method modify the value of the variable self.theta2_.
        This is replaced by the value contained in the theta2 variable passed as an argument.

        """

        self.theta2_ = theta2

    # Getter & Setter of "thresholds"
    def get_thresholds(self):
        """

        This method returns the value of the variable self.thresholds_.
        self.thresholds_ contains an array with the class thresholds parameters.

        """

        return self.thresholds_

    def set_thresholds(self, thresholds):
        """

        This method modify the value of the variable self.thresholds_.
        This is replaced by the value contained in the thresholds variable passed as an argument.

        """

        self.thresholds_ = thresholds

    # Getter & Setter of "n_classes_"
    def get_n_classes(self):
        """

        This method returns the value of the variable self.n_classes_.
        self.n_classes_ contains the number of labels in the problem.

        """

        return self.n_classes_

    def set_n_classes(self, n_classes):
        """

        This method modify the value of the variable self.n_classes_.
        This is replaced by the value contained in the n_classes variable passed as an argument.

        """

        self.n_classes_ = n_classes

    # Getter & Setter of "n_samples_"
    def get_n_samples(self):
        """

        This method returns the value of the variable self.n_samples_.
        self.n_samples_ contains the number of samples of X (train patterns array).

        """

        return self.n_samples_

    def set_n_samples(self, n_samples):
        """

        This method modify the value of the variable self.n_samples_.
        This is replaced by the value contained in the n_samples variable passed as an argument.

        """

        self.n_samples_ = n_samples

    # --------------Private Access functions------------------

    # Download and save the values ​​of Theta1, Theta2 and thresholds_param
    # from the nn_params array to their corresponding array
    def _unpack_parameters(self, nn_params, n_features, n_hidden, n_classes):
        """

        This method gets theta1, theta2 and thresholds_param back from the whole array nn_params.

        Parameters
        ----------

        nn_params: column array, shape ((n_features+1)*n_hidden
        + n_hidden + (n_classes-1))
                Array that is a column vector. It stores the values of theta1,
                theta2 and thresholds_param, all of them together in an array in this order.

        n_features: integer
                Number of nodes in the input layer of the neural network model.

        n_hidden: integer
                Number of nodes in the hidden layer of the neural network model.

        n_classes: integer
                Number of classes.


        Returns
        -------

        theta1: The weights between the input layer and the hidden layer (with biases included).

        theta2: The weights between the hidden layer and the output layer
                (biases are not included as they are the thresholds).

        thresholds_param: classification thresholds.

        """

        n_theta1 = n_hidden * (n_features + 1)
        theta1 = np.reshape(
            nn_params[0:n_theta1], (n_hidden, (n_features + 1)), order="F"
        )

        n_theta2 = n_hidden
        theta2 = np.reshape(
            nn_params[n_theta1 : (n_theta1 + n_theta2)], (1, n_hidden), order="F"
        )

        thresholds_param = np.reshape(
            nn_params[(n_theta1 + n_theta2) :], ((n_classes - 1), 1), order="F"
        )

        return theta1, theta2, thresholds_param

    # Randomly initialize the weights of the neural network layer
    # by entering the number of input and output nodes of that layer
    def _rand_initialize_weights(self, L_in, L_out):
        """

        This method randomly initializes the weights of a layer
         with L_in incoming connections and L_out outgoing connections

         Parameters
        ----------

        L_in: integer
                Number of inputs of the layer.

        L_out: integer
                Number of outputs of the layer.

        Returns
        -------

        W: Array with the weights of each synaptic relationship between nodes.

        """

        W = (
            np.random.rand(L_out, L_in) * 2 * self.get_epsilon_init()
            - self.get_epsilon_init()
        )

        return W

    # Calculate the thresholds
    def _convert_thresholds(self, thresholds_param, n_classes):
        """

        This method transforms thresholds to perform unconstrained optimization.

        thresholds(1) = thresholds_param(1)
        thresholds(2) = thresholds_param(1) + thresholds_param(2)**2
        thresholds(3) = thresholds_param(1) + thresholds_param(2)**2
                                        + thresholds_param(3)**2

        Parameters
        ----------

        thresholds_param: {array-like, column vector}, shape (n_classes-1, 1)
                Contains the original value of the thresholds between classes

        n_classes: integer
                Number of classes.

        Returns
        -------

        thresholds: thresholds of the line

        """

        # Threshold ^2 element by element
        thresholds_pquad = thresholds_param**2

        # Gets row-array containing the thresholds
        thresholds = np.reshape(
            np.multiply(
                np.tile(
                    np.concatenate(
                        (thresholds_param[0:1], thresholds_pquad[1:]), axis=0
                    ),
                    (1, n_classes - 1),
                ).T,
                np.tril(np.ones((n_classes - 1, n_classes - 1))),
            ).sum(axis=1),
            (n_classes - 1, 1),
        ).T

        return thresholds

    # Implements the cost function and obtains the corresponding derivatives.
    def _nnpom_cost_function(
        self, nn_params, n_features, n_hidden, n_classes, X, Y, lambda_value
    ):
        """
        This method implements the cost function and obtains
        the corresponding derivatives.

        Parameters
        ----------

        nn_params: column array, shape ((n_features+1)*n_hidden
        + n_hidden + (n_classes-1))

        Array that is a column vector. It stores the values of theta1,
        theta2 and thresholds_param, all of them together in an array in this order.

        n_features: integer
                Number of nodes in the input layer of the neural network model.

        n_hidden: integer
                Number of nodes in the hidden layer of the neural network model.

        n_classes: integer
                Number of classes.

        X: {array-like, sparse matrix}, shape (n_samples, n_features)
                Training patterns array, where n_samples is the number of samples
                and n_features is the number of features

        Y: array-like, shape (n_samples)
                Target vector relative to X

        lambdaValue:
                Regularization parameter.

        Returns
        -------

        J: Matrix with cost function (updated weight matrix).
        grad: Array with the error gradient of each weight of each layer.

        """

        # Unroll all the parameters
        nn_params = nn_params.reshape((nn_params.shape[0], 1))

        theta1, theta2, thresholds_param = self._unpack_parameters(
            nn_params, n_features, n_hidden, n_classes
        )

        # Convert thresholds
        thresholds = self._convert_thresholds(thresholds_param, n_classes)

        # Setup some useful variables
        n_samples = np.size(X, 0)

        # Neural Network model
        a1 = np.append(np.ones((n_samples, 1)), X, axis=1)
        z2 = np.matmul(a1, theta1.T)
        a2 = 1.0 / (1.0 + np.exp(-z2))

        z3 = np.tile(thresholds, (n_samples, 1)) - np.tile(
            np.matmul(a2, theta2.T), (1, n_classes - 1)
        )
        a3T = 1.0 / (1.0 + np.exp(-z3))
        a3 = np.append(a3T, np.ones((n_samples, 1)), axis=1)
        h = np.concatenate(
            (a3[:, 0].reshape((a3.shape[0], 1)), a3[:, 1:] - a3[:, 0:-1]), axis=1
        )

        # Final output
        out = h

        # Calculate penalty (regularización L2)
        p = np.sum((theta1[:, 1:] ** 2).sum() + (theta2[:, 0:] ** 2).sum())

        # Cross entropy
        J = np.sum(
            -np.log(out[np.where(Y == 1)]), axis=0
        ) / n_samples + lambda_value * p / (2 * n_samples)

        # Cross entropy
        error_der = np.zeros(Y.shape)
        error_der[np.where(Y != 0)] = np.divide(
            -Y[np.where(Y != 0)], out[np.where(Y != 0)]
        )

        # Calculate sigmas
        f_gradients = np.multiply(a3T, (1 - a3T))
        g_gradients = np.multiply(
            error_der,
            np.concatenate(
                (
                    f_gradients[:, 0].reshape(-1, 1),
                    (f_gradients[:, 1:] - f_gradients[:, :-1]),
                    -f_gradients[:, -1].reshape(-1, 1),
                ),
                axis=1,
            ),
        )
        sigma3 = -np.sum(g_gradients, axis=1)[:, np.newaxis]
        sigma2 = np.multiply(np.multiply(np.matmul(sigma3, theta2), a2), (1 - a2))

        # Accumulate gradients
        delta_1 = np.matmul(sigma2.T, a1)
        delta_2 = np.matmul(sigma3.T, a2)

        # Calculate regularized gradient
        p1 = (lambda_value / n_samples) * np.concatenate(
            (np.zeros((np.size(theta1, axis=0), 1)), theta1[:, 1:]), axis=1
        )
        p2 = (lambda_value / n_samples) * theta2[:, 0:]
        theta1_grad = delta_1 / n_samples + p1
        theta2_grad = delta_2 / n_samples + p2

        # Treshold gradients
        thresh_grad_matrix = np.multiply(
            np.concatenate(
                (
                    np.triu(np.ones((n_classes - 1, n_classes - 1))),
                    np.ones((n_classes - 1, 1)),
                ),
                axis=1,
            ),
            np.tile(g_gradients.sum(axis=0), (n_classes - 1, 1)),
        )

        original_shape = thresh_grad_matrix.shape
        thresh_grad_matrix = thresh_grad_matrix.flatten(order="F")

        thresh_grad_matrix[(n_classes)::n_classes] = thresh_grad_matrix.flatten(
            order="F"
        )[(n_classes)::n_classes] + np.multiply(
            error_der[:, 1 : (n_classes - 1)], f_gradients[:, 0 : (n_classes - 2)]
        ).sum(
            axis=0
        )

        thresh_grad_matrix = np.reshape(
            thresh_grad_matrix[:, np.newaxis], original_shape, order="F"
        )

        threshold_grad = thresh_grad_matrix.sum(axis=1)[:, np.newaxis] / n_samples
        threshold_grad[1:] = 2 * np.multiply(threshold_grad[1:], thresholds_param[1:])

        # Unroll gradients
        grad = np.concatenate(
            (
                theta1_grad.flatten(order="F"),
                theta2_grad.flatten(order="F"),
                threshold_grad.flatten(order="F"),
            ),
            axis=0,
        )

        return J, grad
