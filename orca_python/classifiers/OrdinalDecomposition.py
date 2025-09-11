"""OrdinalDecomposition ensemble."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_is_fitted

# scikit-learn >= 1.6
try:
    from sklearn.utils.validation import validate_data as _sk_validate_data

    def _validate_data_compat(estimator, X, y=None, *, reset=True, **kwargs):
        y_arg = "no_validation" if y is None else y
        return _sk_validate_data(estimator, X, y_arg, reset=reset, **kwargs)


# scikit-learn < 1.6
except Exception:

    def _validate_data_compat(estimator, X, y=None, *, reset=True, **kwargs):
        if y is None:
            return estimator._validate_data(X, reset=reset, **kwargs)
        return estimator._validate_data(X, y, reset=reset, **kwargs)


from orca_python.model_selection import load_classifier


class OrdinalDecomposition(ClassifierMixin, BaseEstimator):
    """Ordinal decomposition ensemble classifier.

    This class implements an ensemble model where an ordinal problem is decomposed into
    several binary subproblems, each one of which will generate a different (binary)
    model, though all will share the same base classifier and parameters.

    There are 4 different ways to decompose the original problem based on how the
    coding matrix is built.

    Parameters
    ----------
    dtype : {'ordered_partitions', 'one_vs_next', 'one_vs_followers', 'one_vs_previous'}, \
            default='ordered_partitions'
        Type of decomposition used to build the coding matrix. Each row of the
        coding matrix corresponds to a class and each column to a binary subproblem.
        Entries are in {-1, 0, +1}: -1 for negative class, +1 for positive class,
        and 0 if the class is ignored in that subproblem.

    decision_method : {'exponential_loss', 'hinge_loss', 'logarithmic_loss', 'frank_hall'}, \
            default='frank_hall'
        Method to aggregate the predictions of the binary estimators into class
        probabilities or labels.

    base_classifier : str, default='LogisticRegression'
        Name of the base classifier to be instantiated via
        :func:`orca_python.model_selection.load_classifier`. It can refer to
        a classifier available in the orca-python framework or to a scikit-learn
        compatible classifier.

    parameters : dict or None, default=None
        Hyperparameters to initialize the base classifier. If ``None``,
        defaults of the base classifier are used. Each key must map to a single value.

    Attributes
    ----------
    estimators_ : list of estimators
        Estimators used for predictions.

    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    n_features_in_ : int
        Number of features seen during fit.

    coding_matrix_ : array-like, shape (n_targets, n_targets-1)
        Matrix that defines which classes will be used to build the model of each
        subproblem, and in which binary class they belong inside those new models.
        Further explained previously.

    Notes
    -----
    For ``n_classes=5``, the four decomposition types generate the following
    coding matrices (rows = classes, columns = binary subproblems). Entries are
    ``+1`` for positive class membership and ``-1`` for negative class membership.

    ::

        ordered_partitions     one_vs_next         one_vs_followers     one_vs_previous

        [-1 -1 -1 -1]          [-1  0  0  0]       [-1  0  0  0]        [ 1  1  1  1]
        [ 1 -1 -1 -1]          [ 1 -1  0  0]       [ 1 -1  0  0]        [ 1  1  1 -1]
        [ 1  1 -1 -1]          [ 0  1 -1  0]       [ 1  1 -1  0]        [ 1  1 -1  0]
        [ 1  1  1 -1]          [ 0  0  1 -1]       [ 1  1  1 -1]        [ 1 -1  0  0]
        [ 1  1  1  1]          [ 0  0  0  1]       [ 1  1  1  1]        [-1  0  0  0]

    References
    ----------
    .. [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero, F. Fernández-Navarro
           and C. Hervás-Martínez, "Ordinal regression methods: survey and
           experimental study", IEEE Transactions on Knowledge and Data
           Engineering, Vol. 28. Issue 1, 2016,
           http://dx.doi.org/10.1109/TKDE.2015.2457911

    """

    _parameter_constraints: dict = {
        "dtype": [
            StrOptions(
                {
                    "ordered_partitions",
                    "one_vs_next",
                    "one_vs_followers",
                    "one_vs_previous",
                }
            )
        ],
        "decision_method": [
            StrOptions(
                {"exponential_loss", "hinge_loss", "logarithmic_loss", "frank_hall"}
            )
        ],
        "base_classifier": [str],
        "parameters": [dict, None],
    }

    def __init__(
        self,
        dtype="ordered_partitions",
        decision_method="frank_hall",
        base_classifier="LogisticRegression",
        parameters=None,
    ):
        self.dtype = dtype
        self.decision_method = decision_method
        self.base_classifier = base_classifier
        self.parameters = parameters

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit underlying estimators to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If parameters are invalid or data has wrong format.

        """
        X, y = _validate_data_compat(
            self, X, y, accept_sparse=False, ensure_2d=True, dtype=None
        )

        # Get list of different labels of the dataset
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("OrdinalDecomposition requires at least 2 classes.")

        dtype = str(self.dtype).lower()
        decision = str(self.decision_method).lower()
        if decision == "frank_hall" and dtype != "ordered_partitions":
            raise ValueError(
                "When using Frank and Hall decision method,\
				ordered_partitions must be used"
            )

        # Give each train input its corresponding output label
        # for each binary classifier
        self.coding_matrix_ = self._coding_matrix(dtype, len(self.classes_))
        class_labels = self.coding_matrix_[(np.digitize(y, self.classes_) - 1), :]

        self.estimators_ = []
        parameters = {} if self.parameters is None else self.parameters

        # Fitting n_targets - 1 classifiers
        for n in range(len(class_labels[0, :])):
            estimator = load_classifier(self.base_classifier, param_grid=parameters)
            if not hasattr(estimator, "predict_proba"):
                raise TypeError(
                    f'Base estimator "{self.base_classifier}" must implement predict_proba.'
                )

            mask = class_labels[:, n] != 0
            estimator.fit(X[mask], class_labels[mask, n].ravel())

            self.estimators_.append(estimator)

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted classes.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        AttributeError
            If the specified loss method is not implemented.

        """
        check_is_fitted(self, ["estimators_", "classes_", "coding_matrix_"])
        X = _validate_data_compat(self, X, reset=False, ensure_2d=True, dtype=None)

        # Getting predicted labels for dataset from each classifier
        predictions = self._get_predictions(X)

        decision_method = self.decision_method.lower()
        if decision_method == "exponential_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._exponential_loss(predictions)
            y_pred = self.classes_[np.argmin(losses, axis=1)]

        elif decision_method == "hinge_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._hinge_loss(predictions)
            y_pred = self.classes_[np.argmin(losses, axis=1)]

        elif decision_method == "logarithmic_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._logarithmic_loss(predictions)
            y_pred = self.classes_[np.argmin(losses, axis=1)]

        elif decision_method == "frank_hall":
            # Transforming from binary problems to the original problem
            y_proba = self._frank_hall_method(predictions)
            y_pred = self.classes_[np.argmax(y_proba, axis=1)]

        else:
            raise AttributeError(
                'The specified loss method "%s" is not implemented' % decision_method
            )

        return y_pred

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_proba : ndarray of shape (n_samples,)
            The probability of the sample for each class in the model, where classes are
            ordered as they are in self.classes_.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        ValueError
            If input is invalid.

        AttributeError
            If the specified loss method is not implemented.

        """
        check_is_fitted(self, ["estimators_", "classes_", "coding_matrix_"])
        X = _validate_data_compat(self, X, reset=False, ensure_2d=True, dtype=None)

        # Getting predicted labels for dataset from each classifier
        predictions = self._get_predictions(X)

        decision_method = self.decision_method.lower()
        if decision_method == "exponential_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._exponential_loss(predictions).astype(float)
            eps = np.finfo(float).tiny
            scores = 1.0 / (losses + eps)
            scores -= scores.max(axis=1, keepdims=True)
            y_proba = np.exp(scores)
            y_proba /= y_proba.sum(axis=1, keepdims=True)

        elif decision_method == "hinge_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._hinge_loss(predictions).astype(float)
            eps = np.finfo(float).tiny
            scores = 1.0 / (losses + eps)
            scores -= scores.max(axis=1, keepdims=True)
            y_proba = np.exp(scores)
            y_proba /= y_proba.sum(axis=1, keepdims=True)

        elif decision_method == "logarithmic_loss":
            # Scaling predictions from [0,1] range to [-1,1]
            predictions = predictions * 2 - 1

            # Transforming from binary problems to the original problem
            losses = self._logarithmic_loss(predictions).astype(float)
            eps = np.finfo(float).tiny
            scores = 1.0 / (losses + eps)
            scores -= scores.max(axis=1, keepdims=True)
            y_proba = np.exp(scores)
            y_proba /= y_proba.sum(axis=1, keepdims=True)

        elif decision_method == "frank_hall":
            # Transforming from binary problems to the original problem
            y_proba = self._frank_hall_method(predictions)

        else:
            raise AttributeError(
                'The specified loss method "%s" is not implemented' % decision_method
            )

        return y_proba

    def _coding_matrix(self, dtype, n_classes):
        """Return the coding matrix for a given dataset.

        Parameters
        ----------
        dtype : str
            Type of decomposition to be performed by classifier.

        n_classes : int
            Number of different classes in actual dataset.

        Returns
        -------
        coding_matrix: array-like, shape (n_targets, n_targets-1)
            Each value must be in range {-1, 1, 0}, whether that class will belong to
            negative class, positive class or will not be used for that particular
            binary classifier.

        Raises
        ------
        ValueError
            If the decomposition type does not exist.

        """
        if dtype == "ordered_partitions":
            coding_matrix = np.triu((-2 * np.ones(n_classes - 1))) + 1
            coding_matrix = np.vstack([coding_matrix, np.ones((1, n_classes - 1))])

        elif dtype == "one_vs_next":
            plus_ones = np.diagflat(np.ones((1, n_classes - 1), dtype=int), -1)
            minus_ones = -(np.eye(n_classes, n_classes - 1, dtype=int))
            coding_matrix = minus_ones + plus_ones[:, :-1]

        elif dtype == "one_vs_followers":
            minus_ones = np.diagflat(-np.ones((1, n_classes), dtype=int))
            plus_ones = np.tril(np.ones(n_classes), -1)
            coding_matrix = (plus_ones + minus_ones)[:, :-1]

        elif dtype == "one_vs_previous":
            plusones = np.triu(np.ones(n_classes))
            minusones = -np.diagflat(np.ones((1, n_classes - 1)), -1)
            coding_matrix = np.flip((plusones + minusones)[:, :-1], axis=1)

        else:
            raise ValueError("Decomposition type %s does not exist" % dtype)

        return coding_matrix.astype(int)

    def _get_predictions(self, X):
        """Return the probability of positive class membership.

        For each pattern inside the dataset X, this method returns the probability for
        that pattern to belong to the positive class. There will be as many predictions
        (columns) as different binary classifiers have been fitted previously.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predictions : array, shape (n_samples, n_targets-1)
            Probability estimates or binary classification outcomes.

        """
        predictions = np.array(
            [est.predict_proba(X)[:, 1] for est in self.estimators_]
        ).T

        return predictions

    def _exponential_loss(self, predictions):
        """Compute the exponential losses for each label.

        Computation of the exponential losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_targets-1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        e_losses : ndarray of shape (n_samples, n_classes)
            Exponential losses for each sample of dataset X. One different value for
            each class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        e_losses = np.exp(-M * C).sum(axis=2)
        return e_losses

    def _hinge_loss(self, predictions):
        """Compute the Hinge losses for each label.

        Computation of the Hinge losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_targets-1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        h_losses : ndarray of shape (n_samples, n_classes)
            Hinge losses for each sample of dataset X. One different value for each
            class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        h_losses = np.maximum(0.0, 1.0 - C * M).sum(axis=2)
        return h_losses

    def _logarithmic_loss(self, predictions):
        """Compute the logarithmic losses for each label.

        Computation of the logarithmic losses for each label of the original ordinal
        multinomial problem. Transforms from n-1 binary subproblems to the original
        ordinal problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_targets-1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        l_losses : ndarray of shape (n_samples, n_classes)
            Logarithmic losses for each sample of dataset X. One different value for
            each class label.

        """
        C = self.coding_matrix_[None, :, :]
        M = predictions[:, None, :]
        l_losses = np.log1p(np.exp(-2.0 * C * M)).sum(axis=2)
        return l_losses

    def _frank_hall_method(self, predictions):
        """Calculate probability of each pattern belonging to each target.

        Returns the probability for each pattern of dataset to belong to each one of
        the original targets. Transforms from n-1 subproblems to the original ordinal
        problem with n targets.

        Parameters
        ----------
        predictions : array, shape (n_samples, n_targets-1)
            Probability estimates or binary classification outcomes.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Class membership probabilities for each sample.

        """
        y_proba = np.empty([(predictions.shape[0]), (predictions.shape[1] + 1)])

        # Probabilities of each set to belong to the first ordinal class
        y_proba[:, 0] = 1 - predictions[:, 0]

        # Probabilities for the central classes
        y_proba[:, 1:-1] = predictions[:, :-1] - predictions[:, 1:]

        # Probabilities of each set to belong to the last class
        y_proba[:, -1] = predictions[:, -1]

        return y_proba
