
from sys import exit
import numpy as np

from sklearn.metrics.scorer import make_scorer

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class OrdinalDecomposition(BaseEstimator, ClassifierMixin):
	"""
	"""

	def __init__(self, algorithm="", dtype="", parameters={}):

		self.dtype = dtype
		self.algorithm = algorithm
		self.parameters = parameters

	def fit(self, X, y):
		"""

		"""

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)

		self.X_ = X
		self.y_ = y

		# Gives each train input its corresponding output label for each binary classifier
		self.unique_y_ = np.unique(y)

		self.coding_matrix_ = self._codingMatrix( len(self.unique_y_), self.dtype )
		class_labels = self.coding_matrix_[ (np.digitize(y, self.unique_y_) - 1), :]


		self.classifiers_ = []
		for n in range(len(class_labels[0,:])):

			estimator = self._loadAlgorithm().fit( X[ np.where(class_labels[:,n] != 0) ], \
										np.ravel(class_labels[np.where(class_labels[:,n] != 0), n].T) )
			self.classifiers_.append(estimator)

		return self


	def predict(self, X):
		"""
		"""

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)


		#TODO: Cual es la clase positiva ??
		positive_class = -1
		predictions = np.array([np.interp( np.ravel(c.predict_proba(X)[:, np.where(c.classes_ == positive_class) ]), (0, 1), (-1, +1) )\
						for c in self.classifiers_]).T
		#predictions = np.array([np.interp( c.predict_proba(X)[:,0], (0, 1), (-1, +1) ) for c in self.classifiers_]).T

		eLosses = np.zeros( (X.shape[0], self.coding_matrix_.shape[0]) )

		for i in range(self.coding_matrix_.shape[0]):

			eLosses[:,i] = np.sum(np.exp( predictions * np.tile(self.coding_matrix_[i,:], (predictions.shape[0], 1)) ), axis=1)

		predicted_y = self.unique_y_[np.argmin(eLosses, axis=1)]
		return predicted_y


	def _loadAlgorithm(self):

		# Loading modules to execute algorithm given in configuration file
		modules = [x for x in self.algorithm.split('.')]

		if (len(modules) == 1):
			algorithm = __import__(modules[0])
			algorithm = getattr(algorithm, modules[0])

		elif (len(modules) == 3):
			algorithm = __import__(modules[0] + '.' + modules[1], fromlist=[str(modules[2])])
			algorithm = getattr(algorithm, modules[2])

		else:
			pass

		#TODO: Manejar este parametro internamente o hacer que se deba poner en el .JSON y se gestione en Utilities
		#self.parameters['probability'] = True
		algorithm = algorithm(**self.parameters)
		return algorithm


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
			#exit()

		return coding_matrix.astype(int)


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






