# encoding: utf-8
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.svm import SVC


class CSSVC(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        kernel="rbf",
        degree=3,
        gamma=1,
        coef0=0,
        C=1,
        cache_size=200,
        tol=1e-3,
        shrinking=True,
        probability_estimates=False,
        weight=None,
        random_state=None,
    ):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.cache_size = cache_size
        self.tol = tol
        self.shrinking = shrinking
        self.probability_estimates = probability_estimates
        self.weight = weight
        self.random_state = random_state

        self.models_ = []

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

        p: Label of the pattern which is choose for 1vsALL
        Returns
        -------
        self: object
        """
        X, y = check_X_y(X, y)

        for c in np.unique(y):
            patterns_class = np.where(y == c, 1, 0)

            self.classifier_ = SVC(
                C=self.C,
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0,
                shrinking=self.shrinking,
                probability=self.probability_estimates,
                tol=self.tol,
                cache_size=self.cache_size,
                class_weight=self.weight,
                random_state=self.random_state,
            )

            w = self.ordinalWeights(c, y)
            self.models_.append(
                self.classifier_.fit(X, patterns_class, sample_weight=w)
            )
        return self

    def predict(self, X):
        """
        Performs classification on samples in X
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Returns
        -------
        predicted_y : array, shape (n_samples, n_samples)
        Class labels for samples in X.
        """
        check_is_fitted(self, "models_")

        decfuncs = np.zeros((len(X), len(self.models_)))

        X = check_array(X)

        for idx, model in enumerate(self.models_):
            decfuncs[:, idx] = model.decision_function(X)

        preds = np.argmax(decfuncs, axis=1) + 1

        return preds

    def ordinalWeights(self, p, targets):
        w = np.ones(len(targets))
        w[targets != p] = (
            (abs(p - targets[targets != p]) + 1)
            * len(targets[targets != p])
            / sum(abs(p - targets[targets != p]) + 1)
        )
        return w
