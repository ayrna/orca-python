# encoding: utf-8
import numpy as np
import math as math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from lbfgs import fmin_lbfgs

class NNOP(BaseEstimator, ClassifierMixin):
	
	"""
	NNOP Neural Network with Ordered Partitions (NNOP). This model
		considers the OrderedPartitions coding scheme for the labels and a
		rule for decisions based on the first node whose output is higher
		than a predefined threshold (T=0.5, in our experiments). The
		model has one hidden layer with hiddenN neurons and one outputlayer
		with as many neurons as the number of classes minus one. The learning
		is based on iRProp+ algorithm and the implementation provided by
		Roberto Calandra in his toolbox Rprop Toolbox for {MATLAB}:
		http://www.ias.informatik.tu-darmstadt.de/Research/RpropToolbox
		The model is adjusted by minimizing mean squared error. A regularization
		parameter "lambda" is included based on L2, and the number of
		iterations is specified by the "iter" parameter.
	
		NNPOM public methods:
			fit						- Fits a model from training data
			predict						- Performs label prediction
	
		References:
			[1] J. Cheng, Z. Wang, and G. Pollastri, "A neural network
				approach to ordinal regression," in Proc. IEEE Int. Joint
				Conf. Neural Netw. (IEEE World Congr. Comput. Intell.), 2008,
				pp. 1279-1284.
			[2] P.A. Gutiérrez, M. Pérez-Ortiz, J. Sánchez-Monedero,
				F. Fernández-Navarro and C. Hervás-Martínez
				Ordinal regression methods: survey and experimental study
				IEEE Transactions on Knowledge and Data Engineering, Vol. 28.
				Issue 1, 2016
				http://dx.doi.org/10.1109/TKDE.2015.2457911

		This file is part of ORCA: https://github.com/ayrna/orca
		Original authors: Pedro Antonio Gutiérrez, María Pérez Ortiz, Javier Sánchez Monedero
		Citation: If you use this code, please cite the associated paper http://www.uco.es/grupos/ayrna/orreview
		Copyright:
			This software is released under the The GNU General Public License v3.0 licence
			available at http://www.gnu.org/licenses/gpl-3.0.html    

		NNOP properties:
			epsilonInit					- Range for initializing the weights.
			hiddenN						- Number of hidden neurons of the
										model.
			iter						- Number of iterations for iRProp+
										algorithm.
			lambda						- Regularization parameter.
			theta1						- Hidden layer weigths (with bias)
			theta2						- Output layer weigths (without bias, the biases will be the thresholds)
			thresholds					- Class thresholds parameters
			num_labels					- Number of labels in the problem
			m							- Number of samples of X (train patterns array).
		
	"""

	#Set parameters values
	def __init__(self, epsilonInit=0.5, hiddenN=50, iterations=500, lambdaValue=0.01):
		
		self.__epsilonInit = epsilonInit
		self.__hiddenN = hiddenN
		self.__iter = iterations
		self.__lambdaValue = lambdaValue
	
	
