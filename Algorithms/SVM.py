
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn import svm

class SVM(BaseEstimator, ClassifierMixin):

	def __init__(self, C=0.1, gamma=0.1):
		
		self.C = C
		self.gamma = gamma


	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y

		# Fit model
		svm_model = svm.SVC(kernel='rbf',C=self.C,gamma=self.gamma)
		self.svm_model_ = svm_model.fit(self.X_, self.y_)

		# Return the classifier
		return self

	def predict(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		# Outputs predicted to given data by fitted model
		predicted_y = self.svm_model_.predict(X)

		return predicted_y


	def score(self, X, y):

		check_is_fitted(self, ['X_','y_'])
		X = check_array(X)

		return self.svm_model_.score(X, y)




