
from sys import path
path.insert(0, '..')

import numpy as np
from sklearn.metrics.scorer import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from utilities import load_classifier


class OrdinalDecomposition(BaseEstimator, ClassifierMixin):

	"""
	OrdinalDecomposition ensemble classifier

	This class implements an ensemble model where an ordinal problem
	is decomposed into several binary subproblems, each one of which
	will generate a different (binary) model, though all will share same base
	classifier and parameters for it.

	There are 4 different ways to decompose the original problem based
	on how the coding matrix is built.


	Parameters
	----------

	dtype: string
		Type of decomposition to be performed by class. May be one of
		4 different types: 'OrderedPartitions', 'OneVsNext',
		'OneVsFollowers' or 'OneVsPrevious'

		The coding matrix generated by each method, for a problem with
		5 classes will be as follows:

		OrderedPartitions	OneVsNext	OneVsFollowers	OneVsPrevious
		
		-, -, -, -;		-,  ,  ,  ;		-,  ,  ,  ;		+, +, +, +;
		+, -, -, -;		+, -,  ,  ;		+, -,  ,  ;		+, +, +, -;
		+, +, -, -;		 , +, -,  ;		+, +, -,  ;		+, +, -,  ;
		+, +, +, -;		 ,  , +, -;		+, +, +, -;		+, -,  ,  ;
		+, +, +, +;		 ,  ,  , +;		+, +, +, +;		-,  ,  ,  ;

		where rows represent classes and columns represent base classifiers. plus
		signs indicate that for that classifier, that class will be
		part of the positive class, on the other hand, a minus sign
		places that class into the negative one for that binary
		problem. If there is no sign, then those samples will not be
		used when building the model.

	decision_method: string
		Decision method that transforms the predictions of the n different
		base classifiers to produce the final label (one among the real 
		ordinal classes).

	base_classifier: string
		Base classifier used to build a model for each binary subproblem.
		The base classifier need to be a classifier of orca-python framework 
		or any classifier available in scikit-learn. Other classifiers that 
		implements the scikit-learn API can be used here. 

	parameters: dict
		This dictionary will store the parameters used to build the base 
		classifier. Only one value per parameter is allowed.

	Attributes
	----------

	classes_: list
		List that contains all different class labels found in the original
		dataset.

	coding_matrix_: array-like, shape (n_targets, n_targets-1)
		Matrix that defines which classes will be used to build the model of each
		subproblem, and in which binary class they belong inside
		those new models. Further explained previously.

	classifiers_: list of classifiers
		Initialy empty, will include all fitted models for each
		subproblem once the fit function for this class is called
		successfully.


	References
	----------
	P.A. Gutierrez, M. Perez-Ortiz, J. Sanchez-Monedero, F. Fernandez-Navarro and C. Hervas-Martinez (2016),
	"Ordinal regression methods: survey and experimental study",
	IEEE Transactions on Knowledge and Data Engineering. Vol. 28. Issue 1
	http://dx.doi.org/10.1109/TKDE.2015.2457911

	"""

	#TODO: Especificar valores por defecto
	def __init__(self, dtype="ordered_partitions", decision_method="frank_hall", base_classifier="",  parameters={}):

		self.dtype = dtype
		self.decision_method = decision_method
		self.base_classifier = base_classifier
		self.parameters = parameters


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

		Returns
		-------

		self: object
		"""


		X, y = check_X_y(X, y)

		self.X_ = X
		self.y_ = y

		# Get list of different labels of the dataset
		self.classes_ = np.unique(y)

		# Gives each train input its corresponding output label for each binary classifier
		self.coding_matrix_ = self._coding_matrix( len(self.classes_) )
		class_labels = self.coding_matrix_[ (np.digitize(y, self.classes_) - 1), :]


		self.classifiers_ = []
		# Fitting n_targets - 1 classifiers, each one with a different
		# combination of train inputs given by the coding_matrix
		for n in range(len(class_labels[0,:])):

			estimator = load_classifier(self.base_classifier, self.parameters)
			estimator.fit(X[ np.where(class_labels[:,n] != 0) ], \
						  np.ravel(class_labels[np.where(class_labels[:,n] != 0), n].T) )


			self.classifiers_.append(estimator)

		return self



	def predict(self, X):

		"""
		Performs classification on samples in X

		Parameters
		----------

		X: {array-like, sparse matrix}, shape (n_samples, n_features)

		Returns
		-------

		predicted_y: array, shape (n_samples,)
			Class labels for samples in X.
		"""

		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		# Getting predicted labels for dataset from each classifier
		predictions = self._get_predictions(X)

		decision_method = self.decision_method.lower()
		if decision_method == "exponential_loss":

			# Scaling predictions from [0,1] range to [-1,1]
			predictions = (predictions*2 - 1)

			# Transforming from binary problems to the original problem
			losses = self._exponential_loss(predictions)
			predicted_y = self.classes_[np.argmin(losses, axis=1)]


		elif decision_method == "hinge_loss":
			
			# Scaling predictions from [0,1] range to [-1,1]
			predictions = (predictions*2 - 1)

			# Transforming from binary problems to the original problem
			losses = self._hinge_loss(predictions)
			predicted_y = self.classes_[np.argmin(losses, axis=1)]


		elif decision_method == "logaritmic_loss":

			# Scaling predictions from [0,1] range to [-1,1]
			predictions = (predictions*2 - 1)

			# Transforming from binary problems to the original problem
			losses = self._logaritmic_loss(predictions)
			predicted_y = self.classes_[np.argmin(losses, axis=1)]


		elif decision_method == "frank_hall":

			# Transforming from binary problems to the original problem
			predicted_proba_y = self._frank_hall_method(predictions)
			predicted_y = self.classes_[np.argmax(predicted_proba_y, axis=1)]


		else:
			raise AttributeError('The specified loss method "%s" is not implemented' % decision_method)


		return predicted_y




	def _coding_matrix(self, n_classes):

		"""
		Method that returns the coding matrix for a given dataset.

		Parameters
		----------

		n_classes: int
			Number of different classes in actual dataset

		Returns
		-------

		coding_matrix: array-like, shape (n_targets, n_targets-1)
			Each value must be in range {-1, 1, 0}, whether that class
		 	will belong to negative class, positive class or will not
			be used for that particular binary classifier.
		"""

		dtype = self.dtype.lower()
		if dtype == "ordered_partitions":

			coding_matrix = np.triu( (-2 * np.ones(n_classes - 1)) ) + 1
			coding_matrix = np.vstack([coding_matrix, np.ones((1, n_classes-1))])

		elif dtype == "one_vs_next":

			plus_ones = np.diagflat(np.ones((1, n_classes - 1), dtype=int), -1)
			minus_ones = -( np.eye(n_classes, n_classes - 1, dtype=int) )
			coding_matrix = minus_ones + plus_ones[:,:-1]

		elif dtype == "one_vs_followers":

			minus_ones = np.diagflat(-np.ones((1, n_classes), dtype=int))
			plus_ones = np.tril(np.ones(n_classes), -1)
			coding_matrix = (plus_ones + minus_ones)[:,:-1]

		elif dtype == "one_vs_previous":

			plusones = np.triu(np.ones(n_classes))
			minusones = -np.diagflat(np.ones((1, n_classes - 1)), -1)
			coding_matrix = np.flip( (plusones + minusones)[:,:-1], axis=1 )

		else:

			raise ValueError("Decomposition type %s does not exist" % dtype)

		return coding_matrix.astype(int)



	def _get_predictions(self, X):

		"""
		For each pattern inside the dataset X, this method returns
		the probability for that pattern to belong to the positive
		or negative class. There will be as many predictions as
		different binary classifiers has been fitted previously.

		Parameters
		----------

		X: {array-like, sparse matrix}, shape (n_samples, n_features)

		Returns
		-------

		predictions: array, shape (n_targets-1, n_samples, 2)
		"""

		predictions = np.array(list(map(lambda c: c.predict_proba(X), self.classifiers_)))

		return predictions



	def _exponential_loss(self, predictions):

		"""
		Computation of the exponential losses for each label of the
		original ordinal multinomial problem. Transforms n 
		binary subproblems into the original ordinal problem.

		Parameters
		----------

		predictions: array, shape (n_targets-1, n_samples, 2)

		Returns
		-------

		e_losses: array, shape (n_samples, unique_labels)
			Exponential losses for each sample of dataset X. One
			different value for each class label.
		"""


		# Computing exponential losses
		e_losses = np.zeros( (predictions.shape[1], self.coding_matrix_.shape[0]) )
		for i in range(self.coding_matrix_.shape[0]):

			e_losses[:,i] = np.sum(np.exp( -(predictions[:,:,1].T) * np.tile(self.coding_matrix_[i,:],\
											(predictions.shape[1], 1)) ), axis=1)

		return e_losses



	def _hinge_loss(self, predictions):

		"""
		Computation of the Hinge losses for each label of the
		original ordinal multinomial problem. Transforms from n 
		binary subproblems to the original ordinal problem.

		Parameters
		----------

		predictions: array, shape (n_targets-1, n_samples, 2)

		Returns
		-------

		hLosses: array, shape (n_samples, unique_labels)
			Hinge losses for each sample of dataset X. One
			different value for each class label.

		"""

		# Computing Hinge losses
		h_losses = np.zeros( (predictions.shape[1], self.coding_matrix_.shape[0]) )
		for i in range(self.coding_matrix_.shape[0]):

			h_losses[:,i] = np.sum( np.maximum(0, (1 - np.tile(self.coding_matrix_[i,:], (predictions.shape[1], 1)) * predictions[:,:,1].T) ), axis=1 )

		return h_losses



	def _logaritmic_loss(self, predictions):

		"""
		Computation of the logaritmic losses for each label of the
		original ordinal multinomial problem. Transforms from n 
		binary subproblems to the original ordinal problem again.

		Parameters
		----------

		predictions: array, shape (n_targets-1, n_samples, 2)

		Returns
		-------

		eLosses: array, shape (n_samples, unique_labels)
			Logaritmic losses for each sample of dataset X. One
			different value for each class label.

		"""


		# Computing logaritmic losses
		l_losses = np.zeros( (predictions.shape[1], self.coding_matrix_.shape[0]) )
		for i in range(self.coding_matrix_.shape[0]):

			l_losses[:,i] = np.sum( np.log(1 + np.exp(-2 * np.tile(self.coding_matrix_[i,:], (predictions.shape[1], 1)) * predictions[:,:,1].T)), axis=1 )

		return l_losses



	def _frank_hall_method(self, predictions):

		"""
		Decision method used to transform from n predictions of binary
		problems to predictions of a multinomial ordinal classification

		Parameters
		----------

		predictions: array, shape (n_targets-1, n_samples, 2)

		Returns
		-------

		predicted_y: array, shape (n_samples,)
			Class labels predicted for samples in dataset X.
		"""


		if self.dtype.lower() != "ordered_partitions":
			raise AttributeError("When using Frank and Hall decision method, ordered_partitions must be used")


		predicted_proba_y = np.empty([predictions.shape[1], (predictions.shape[0] + 1)])

		# Probabilities of each set to belong to the first ordinal class
		predicted_proba_y[:,0] = 1 - np.ravel(predictions[0][:, 1])

		for i in range(1, predictions.shape[0]):

			# Probability of sets to belong to class i
			predicted_proba_y[:,i] = np.ravel(predictions[i-1][:, 1]) -\
									np.ravel(predictions[i][:, 1])

		# Probabilities of each set to belong to the last class
		predicted_proba_y[:,-1] = np.ravel(predictions[-1][:, 1])

		return predicted_proba_y






