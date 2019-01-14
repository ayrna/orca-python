
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from sklearn import linear_model

class LR(BaseEstimator, ClassifierMixin):

	def __init__(self, C=0.1, l2="True"):
		
		self.C = C
		self.l2 = l2


	def fit(self, X, y):

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y

		# Fit model
		penalty = 'l1'
		if self.l2:
			penalty = 'l2'

		lr_model = linear_model.LogisticRegression(solver='liblinear', C=self.C, penalty=penalty, multi_class='ovr')
		self.lr_model_ = lr_model.fit(self.X_, self.y_)

		# Return the classifier
		return self

	def predict(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		# Outputs predicted to given data by fitted model
		predicted_y = self.lr_model_.predict(X)

		return predicted_y


	def predict_proba(self, X):

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		# Outputs predicted to given data by fitted model
		predicted_y_proba = self.lr_model_.predict_proba(X)

		return predicted_y_proba


	def score(self, X, y):

		check_is_fitted(self, ['X_','y_'])
		X = check_array(X)

		return self.lr_model_.score(X, y)



	def setParameters(self, parameters):

		if 'C' in parameters:

			self.C = parameters['C']

		if 'l2' in parameters:

			self.l2 = parameters['l2']
















