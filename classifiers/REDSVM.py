# encoding: utf-8
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from libsvmRank.python import svm


class REDSVM(BaseEstimator, ClassifierMixin):

	"""
	REDSVM Reduction from ordinal regression to binary SVM classifiers [1].
    The configuration used is the identity coding matrix, the absolute
    cost matrix and the standard binary soft-margin SVM. This class uses
    libsvm-rank-2.81 implementation
    (http://www.work.caltech.edu/~htlin/program/libsvm/)
	
	   REDSVM methods:
		  fit                        - Fits a model from training data
		  predict                    - Performs label prediction
	
	   References:
		 [1] H.-T. Lin and L. Li, "Reduction from cost-sensitive ordinal
			 ranking to weighted binary classification" Neural Computation,
			 vol. 24, no. 5, pp. 1329-1367, 2012.
			 http://10.1162/NECO_a_00265
		 [2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
			 F. Fernández-Navarro and C. Hervás-Martínez
			 Ordinal regression methods: survey and experimental study
			 IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue 1
			 2016
			 http://dx.doi.org/10.1109/TKDE.2015.2457911

	Model Parameters:
	t kernel_type : set type of kernel function (default 2)
		0 -- linear: u'*v
		1 -- polynomial: (gamma*u'*v + coef0)^degree
		2 -- radial basis function: exp(-gamma*|u-v|^2)
		3 -- sigmoid: tanh(gamma*u'*v + coef0)
		4 -- stump: -|u-v|_1 + coef0\n"
		5 -- perceptron: -|u-v|_2 + coef0\n"
		6 -- laplacian: exp(-gamma*|u-v|_1)\n"
		7 -- exponential: exp(-gamma*|u-v|_2)\n"
		8 -- precomputed kernel (kernel values in training_instance_matrix)
	d degree : set degree in kernel function (default 3)
	g gamma : set gamma in kernel function (default 1/num_features)
	r coef0 : set coef0 in kernel function (default 0)
	c cost : set the parameter C (default 1)
	m cachesize : set cache memory size in MB (default 100)
	e epsilon : set tolerance of termination criterion (default 0.001)
	h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
	q : quiet mode (no outputs)
	"""

	#Set parameters values
	def __init__(self, t=2, d=3, g=None, r=0, c=1, m=100, e=0.001, h=1):

		self.t = t
		self.d = d
		self.g = g
		self.r = r
		self.c = c
		self.m = m
		self.e = e
		self.h = h
		

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

		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)
		
		#Set the default g value if necessary
		if self.g == None:
			self.g = 1 / np.size(X, 1)
			
		# Fit the model
		options = "-s 5 -t {} -d {} -g {} -r {} -c {} -m {} -e {} -h {} -q".format(str(self.t),
																				str(self.d),
																				str(self.g),
																				str(self.r),
																				str(self.c),
																				str(self.m),
																				str(self.e),
																				str(self.h))
		self.classifier_ = svm.fit(y.tolist(), X.tolist(), options)
		
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
		
		# Check is fit had been called
		check_is_fitted(self, ['classifier_'])

		# Input validation
		X = check_array(X)

		predicted_y = svm.predict(X.tolist(), self.classifier_)
		
		return predicted_y
