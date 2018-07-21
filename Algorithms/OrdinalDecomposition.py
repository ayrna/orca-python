
import sys
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class OrdinalDecomposition(BaseEstimator, ClassifierMixin):
	"""
	"""

	def __init__(self, algorithm="", dtype="OrderedPartitions", parameters={}):

		self.algorithm = algorithm
		self.parameters = parameters
		self.dtype = dtype

	def fit(self, X, y):
		"""

		"""

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)

		self.X_ = X
		self.y_ = y

		#TODO: Otra forma de cargar la carpeta donde esta el fichero de utilities
		sys.path.insert(0, '../')

		module = __import__(self.algorithm)
		algorithm = getattr(module, self.algorithm)

		# Gives each train input its corresponding output label for each binary classifier
		self.coding_matrix_ = self._codingMatrix( len(np.unique(y)), self.dtype )
		class_labels = self.coding_matrix_[y.astype(int),:]


		#TODO:	Hay tres clases para OneVsNext/Previous/Followers, o los algoritmos manejan los 0s como si fueran patrones
		#		no validos ?

		self.classifiers_ = []
		for n in range(len(class_labels[0,:])):

			optimal_estimator = self._getOptimalEstimator(X, class_labels[:,n], algorithm, self.parameters)
			self.classifiers_.append(optimal_estimator)

		return self

	def predict(self, X):
		"""
		"""

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		predictions = np.array([np.interp( c.predict_proba(X)[:,0], (0, 1), (-1, +1) ) for c in self.classifiers_]).T
		eLosses = np.zeros( (X.shape[0], self.coding_matrix_.shape[0]) )

		for i in range(self.coding_matrix_.shape[0]):

			eLosses[:,i] = np.sum(np.exp( predictions * np.matlib.repmat(self.coding_matrix_[i,:], predictions.shape[0], 1) ), axis=1)

		predicted_y = np.argmin(eLosses, axis=1)
		return predicted_y


	"""
	def predict(self, X):

		X = check_array(X)

		# Outputs predicted to given data by fitted model
		predicted_proba_y = np.empty( [X.shape[0], len(self.classifiers_) + 1] )

		for i, c in enumerate(self.classifiers_):

			if i == 0:
				predicted_proba_y[:,i] = 1 - c.predict_proba(X)[:,0]
			else:
				predicted_proba_y[:,i] = previous_proba_y - c.predict_proba(X)[:,0]

			# Storing actual prediction for next iteration
			previous_proba_y = c.predict_proba(X)[:,1]

		predicted_proba_y[:,-1] = self.classifiers_[-1].predict_proba(X)[:,0]
		predicted_y = np.argmax(predicted_proba_y, axis=1)

		return predicted_y
	"""

	def _codingMatrix(self, nClasses, dType):

		"""
			Method that returns the coding matrix for a given dataset.

			Receives:
 
				- nClasses: Number of classes in actual dataset
				- dType: Type of decomposition to be applied

			Returns:

				- Coding Matrix: M x N matrix, where M is the number of classes in dataset and N is the number 
								of binary classifiers (M-1) created by the ordinal decomposition. Each value must be in
								range {-1, 1, 0}, whether that class will belong to negative class, positive class or
								will not be used for that particular binary classifier.
		"""

		dType = dType.lower()

		if dType == "orderedpartitions":

			coding_matrix = np.triu( (-2 * np.ones(nClasses - 1)) ) + 1
			coding_matrix = np.vstack([coding_matrix, np.ones((1, nClasses-1))])

		elif dType == "onevsnext":

			plus_ones = np.diagflat(np.ones((1, nClasses - 1), dtype=int), -1)
			minus_ones = -( np.eye(nClasses, nClasses - 1, dtype=int) )
			coding_matrix = minus_ones + plus_ones[:,:-1]

		elif dType == "onevsfollowers":

			minus_ones = np.diagflat(-np.ones((1, nClasses), dtype=int))
			plus_ones = np.tril(np.ones(nClasses), -1)
			coding_matrix = (plus_ones + minus_ones)[:,:-1]

		elif dType == "onevsprevious":

			plusones = np.triu(np.ones(nClasses))
			minusones = -np.diagflat(np.ones((1, nClasses - 1)), -1)
			coding_matrix = np.flip( (plusones + minusones)[:,:-1], axis=1 )

		else:

			print "Decomposition type", dType, "does not exist."
			#sys.exit()

		return coding_matrix.astype(int)

	def _getOptimalEstimator(self, train_inputs, train_outputs, algorithm, parameters):


		module = __import__("Metrics")
		metric = getattr(module, "ccr")

		gib = module.greater_is_better("ccr".lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)

		optimal = GridSearchCV(estimator=algorithm(probability=True), param_grid=parameters, scoring=scoring_function, cv=5)
		optimal.fit(train_inputs, train_outputs)
		return optimal








