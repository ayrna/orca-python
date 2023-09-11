import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RegressorWrapper(BaseEstimator, ClassifierMixin):
    """
     Regression algorithms wrapper

     The mainly purpose of this class is create a generic wrapper which could
     obtains ordinal models by regression algorithms, the targets for the independent
     variable could be provided by the users and it works all the regression algorithms
     avaliable in sklearn.

    Parameters
    ------------

    classifier: sklearn regressor
        Base regressor used to build de model. this need to be a sklearn regressor.

     labels: String[]
        Array which include the labels choosed by the user to transform the continous
        data into nominal data, if users does not specify the labels by himself the
        method will use a predefined values

     params: String
        path of the Json file from where the method load the configuration for sklearn
        regressor in case of the user do not incluide it the regressor will use the
        default value by sklearn.


    """

    def __init__(self, base_regressor=None, **params):
        self.params = params
        self.classifier_ = None

        self.base_regressor = base_regressor
        if self.base_regressor is None:
            self.base_regressor = "sklearn.svm.SVR"

        classifier = __import__(self.base_regressor.rsplit(".", 1)[0], fromlist="None")
        classifier = getattr(classifier, self.base_regressor.rsplit(".", 1)[1])
        self.classifier_ = classifier(**self.params)

        self.classes_ = None

    def fit(self, X, y, **params):
        """
        Fit the model with the training data and set the kwargs for the regressor.

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
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        self.classifier_.fit(X, y, **params)
        return self

    def predict(self, X):
        """
        Performs classification on samples in X

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------

        predicted_y : array, shape (n_samples,)
                Class labels for samples in X.
        """
        check_is_fitted(self, ["classifier_"])
        X = check_array(X)

        predicted_y = self.classifier_.predict(X)
        predicted_y = np.clip(
            np.round(predicted_y, 0), self.classes_[0], self.classes_[-1]
        )
        return np.asarray(predicted_y, dtype=int)

    def set_params(self, **kwargs):
        if not kwargs["base_regressor"]:
            self.base_regressor = "sklearn.svm.SVR"
        else:
            self.base_regressor = kwargs["base_regressor"]
            kwargs.pop("base_regressor")

        classifier = __import__(self.base_regressor.rsplit(".", 1)[0], fromlist="None")
        classifier = getattr(classifier, self.base_regressor.rsplit(".", 1)[1])
        self.classifier_ = classifier(**kwargs)

        return self

    def get_params(self, deep=True):
        # This function is overrided to get the params of the internal regressor.
        out = dict()
        keys = self._get_param_names() + list(self.classifier_.get_params().keys())
        for num, key in enumerate(keys):
            if num < len(self._get_param_names()):
                value = getattr(self, key)
            else:
                value = getattr(self.classifier_, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
