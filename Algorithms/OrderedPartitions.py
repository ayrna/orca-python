
import sys
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class OrderedPartitions(BaseEstimator, ClassifierMixin):
	"""
	"""

	def __init__(self, algorithm="", parameters={}):

		self.algorithm = algorithm
		self.parameters = parameters


	def fit(self, train_inputs, train_outputs):

		sys.path.insert(0, '../')

		module = __import__(self.algorithm)
		algorithm = getattr(module, self.algorithm)

		classes = self._extractClasses(train_inputs, train_outputs)

		self.classifiers_ = []
		for n in range(len(classes) - 1):


			# Divides samples in positive class and negative class
			negative_class_inputs = np.concatenate(classes[:n+1])
			negative_class_outputs = np.ones((negative_class_inputs.shape[0],), dtype=int)

			positive_class_inputs = np.concatenate(classes[n+1:])
			positive_class_outputs = np.zeros((positive_class_inputs.shape[0],), dtype=int)

			X = np.append(negative_class_inputs, positive_class_inputs, axis=0)
			y = np.append(negative_class_outputs, positive_class_outputs, axis=0)

			optimal_estimator = self._getOptimalEstimator(X, y, algorithm, self.parameters)
			self.classifiers_.append(optimal_estimator)


	def predict(self, X):

		X = check_array(X)

		# Outputs predicted to given data by fitted model
		predicted_proba_y = np.empty( [X.shape[0], len(self.classifiers_) + 1] )

		for i, c in enumerate(self.classifiers_):

			if i == 0:
				predicted_proba_y[:,i] = 1 - c.predict_proba(X)[:,0]
			else:
				predicted_proba_y[:,i] = previous_proba_y - c.predict_proba(X)[:,0]

			previous_proba_y = c.predict_proba(X)[:,0]

		predicted_proba_y[:,-1] = self.classifiers_[-1].predict_proba(X)[:,0]
		predicted_y = np.argmax(predicted_proba_y, axis=1)

		return predicted_y


	def _extractClasses(self, train_inputs, train_outputs):

		"""


		Returns all samples segregated by their classes in decreasing order of the class label
		"""
		
		class_labels = np.unique(train_outputs)
		class_labels.sort()

		# Dividing classes for labels
		classes = []
		for c in class_labels:

			classes.append(train_inputs[np.where(train_outputs == c)])

		return classes



	def _getOptimalEstimator(self, train_inputs, train_outputs, algorithm, parameters):


		module = __import__("Metrics")
		metric = getattr(module, "ccr")

		# TODO: Cuidado con el greater is better (MAE es un indicador que funciona a la inversa - menor es mejor)
		scoring_function = make_scorer(metric, greater_is_better=True)

		# TODO: Logistic Regression no necesita de la opcion probability en el __init__, pero si que tiene funcion predict_proba
		optimal = GridSearchCV(estimator=algorithm(probability=True), param_grid=parameters, scoring=scoring_function, cv=5)
		optimal.fit(train_inputs, train_outputs)
		return optimal








