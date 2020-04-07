# encoding: utf-8
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from svorex import svorextrain, svorexpredict


class SVOREX(BaseEstimator, ClassifierMixin):

	"""
	SVOREX Support Vector for Ordinal Regression (Explicit constraints)
    This class derives from the Algorithm Class and implements the
    SVOREX method. This class uses SVOREX implementation by
    W. Chu et al (http://www.gatsby.ucl.ac.uk/~chuwei/svor.htm)
    
		SVOREX methods:
			fit                        - Fits a model from training data
			predict                    - Performs label prediction
    
       	References:
         [1] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
             F. Fernández-Navarro and C. Hervás-Martínez
             Ordinal regression methods: survey and experimental study
             IEEE Transactions on Knowledge and Data Engineering, Vol. 28. Issue 1
             2016
             http://dx.doi.org/10.1109/TKDE.2015.2457911
         [2] W. Chu and S. S. Keerthi, Support Vector Ordinal Regression,
             Neural Computation, vol. 19, no. 3, pp. 792–815, 2007.
             http://10.1162/neco.2007.19.3.792
		
	Model Parameters:
		kernel_type:
			0 -- gaussian: use gaussian kernel (default)
			1 -- linear:   use imbalanced Linear kernel
			2 -- polynomial: (Use parameter p to change the order) use Polynomial kernel with order p
		T t: set Tolerance at t (default 0.001)
		K o: set kappa value at o (default 1)	
		C o: set C value at o (default  1)	
	"""

	#Set parameters values
	def __init__(self, kernel_type=0, p=2, t=0.001, c=1, k=1):

		self.kernel_type_ = kernel_type
		self.p_ = p
		self.t_ = t
		self.c_ = c
		self.k_ = k
		

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

		arg = ""
		#Prepare the kernel type arguments
		if (self.kernel_type_ == 1):
			arg = "-L"
		elif (self.kernel_type_ == 2):
			arg = "-P {}".format(self.p_)
			
		# Fit the model
		options = "svorex {} -T {} -K {} -C {}".format(arg, str(self.t_), str(self.k_), str(self.c_))
		self.classifier_ = svorextrain.run(self.y_.tolist(), self.X_.tolist(), options)
		
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
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		predicted_y = svorexpredict.run(X.tolist(), self.classifier_)
		
		return predicted_y