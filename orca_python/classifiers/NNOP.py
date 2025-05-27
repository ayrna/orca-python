# encoding: utf-8
import numpy as np
import math as math
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import scipy

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
		iterations is specified by the "iterations" parameter.

		NNOP public methods:
			fit						- Fits a model from training data
			predict					- Performs label prediction

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
			epsilon_init				- Range for initializing the weights.
			hidden_n					- Number of hidden neurons of the
										model.
			iterations					- Number of iterations for fmin_l_bfgs_b
										algorithm.
			lambda_value				- Regularization parameter.
			theta1						- Hidden layer weigths (with bias).
			theta2						- Output layer weigths.
			num_labels					- Number of labels in the problem.
			m							- Number of samples of X (train patterns array).

	"""

	# Constructor of class NNOP (set parameters values).
	def __init__(self, epsilon_init=0.5, hidden_n=50, iterations=500, lambda_value=0.01):
		
		self.epsilon_init = epsilon_init
		self.hidden_n = hidden_n
		self.iterations = iterations
		self.lambda_value = lambda_value


	#--------Main functions (Public Access)--------


	def fit(self,X,y):

		"""

		Trains the model for the model NNOP method with TRAIN data.
		Returns the projection of patterns (only valid for threshold models) and the predicted labels.
		
		Parameters
		----------

		X: {array-like, sparse matrix}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of samples
			and n_features is the number of features

		y: array-like, shape (n_samples)
			Target vector relative to X

		Returns
		-------

		self: The object NNOP.

		"""
		if self.epsilon_init < 0 or self.hidden_n < 1 or self.iterations < 1 or self.lambda_value < 0:
			return None
		
		
		# Check that X and y have correct shape
		X, y = check_X_y(X, y)
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		# Aux variables
		y = y[:,np.newaxis]
		input_layer_size = X.shape[1]
		num_labels = np.size(np.unique(y))
		m = X.shape[0]
		
		# Recode y to Y using ordinalPartitions coding
		Y = 1 * (np.tile(y, (1,num_labels)) <= np.tile(np.arange(1,num_labels+1)[np.newaxis,:], (m,1)))

		# Hidden layer weigths (with bias)
		initial_theta1 = self.__rand_initialize_weights(input_layer_size+1, self.hidden_n)
		# Output layer weigths
		initial_theta2 = self.__rand_initialize_weights(self.hidden_n+1, num_labels-1)
		
		# Pack parameters
		initial_nn_params = np.concatenate((initial_theta1.flatten(order='F'),
		 initial_theta2.flatten(order='F')), axis=0)[:,np.newaxis]
		
		results_optimization = scipy.optimize.fmin_l_bfgs_b(func=self.__nnop_cost_function, x0=initial_nn_params.ravel(),args=(input_layer_size, self.hidden_n,
			num_labels, X, Y, self.lambda_value), fprime=None, factr=1e3, maxiter=self.iterations,iprint=-1)
		
		self.nn_params = results_optimization[0]
		# Unpack the parameters
		theta1, theta2 = self.__unpack_parameters(self.nn_params, input_layer_size, self.get_hidden_n(), num_labels)
		self.theta1 = theta1
		self.theta2 = theta2
		self.num_labels = num_labels
		self.m = m

		return self
	
	def predict (self, test):
		
		"""

		Predicts labels of TEST patterns labels. The object needs to be fitted to the data first.

		Parameters
		----------

		test: {array-like, sparse matrix}, shape (n_samples, n_features)
			test patterns array, where n_samples is the number of samples
			and n_features is the number of features

		Returns
		-------

		predicted: {array-like, sparse matrix}, shape (n_samples,)
			Vector array with predicted values for each pattern of test patterns.

		"""
		# Check is fit had been called
		check_is_fitted(self)
		
		# Input validation
		test = check_array(test)
		m = test.shape[0]

		a1 = np.append(np.ones((m, 1)), test, axis=1)
		z2 = np.append(np.ones((m,1)), np.matmul(a1, self.theta1.T), axis=1)

		a2 =  1.0 / (1.0 + np.exp(-z2))
		projected = np.matmul(a2,self.theta2.T)
		projected = 1.0 / (1.0 + np.exp(-projected))

		a3 = np.multiply(np.where(np.append(projected, np.ones((m,1)), axis=1)>0.5, 1, 0),
		 np.tile(np.arange(1,self.num_labels+1), (m,1)))
		a3[np.where(a3==0)] = self.num_labels + 1
		predicted = a3.min(axis=1) 

		return predicted
	
	#--------Getters & Setters (Public Access)--------
	

	# Getter & Setter of "epsilonInit"
	def get_epsilon_init (self):
	
		"""

		This method returns the value of the variable self.epsilon_init.
		self.epsilon_init contains the value of epsilon, which is the initialization range of the weights.

		"""

		return self.epsilon_init

	def set_epsilon_init (self, epsilon_init):

		"""

		This method modify the value of the variable self.epsilon_init.
		This is replaced by the value contained in the epsilon_init variable passed as an argument.

		"""

		self.epsilon_init = epsilon_init
	

	# Getter & Setter of "hiddenN"
	def get_hidden_n (self):

		"""

		This method returns the value of the variable self.hidden_n.
		self.hidden_n contains the number of nodes/neurons in the hidden layer.

		"""

		return self.hidden_n

	def set_hidden_n (self, hidden_n):
		
		"""

		This method modify the value of the variable self.hidden_n.
		This is replaced by the value contained in the hidden_n variable passed as an argument.

		"""

		self.hidden_n = hidden_n
	

	# Getter & Setter of "iterations"
	def get_iterations (self):
		
		"""

		This method returns the value of the variable self.iterations.
		self.iterations contains the number of iterations.

		"""

		return self.iterations
	
	def set_iterations (self, iterations):

		"""

		This method modify the value of the variable self.iterations.
		This is replaced by the value contained in the iterations variable passed as an argument.

		"""

		self.iterations = iterations
	

	# Getter & Setter of "lambdaValue"
	def get_lambda_value (self):

		"""

		This method returns the value of the variable self.lambda_value.
		self.lambda_value contains the Lambda parameter used in regularization.

		"""

		return self.lambda_value
	
	def set_lambda_value (self, lambda_value):

		"""

		This method modify the value of the variable self.lambda_value.
		This is replaced by the value contained in the lambda_value variable passed as an argument.

		"""

		self.lambda_value = lambda_value


	# Getter & Setter of "theta1"
	def get_theta1 (self):
		
		"""

		This method returns the value of the variable self.theta1.
		self.theta1 contains an array with the weights of the hidden layer (with biases included).

		"""

		return self.theta1

	def set_theta1 (self, theta1):
		
		"""

		This method modify the value of the variable self.theta1.
		This is replaced by the value contained in the theta1 variable passed as an argument.

		"""

		self.theta1 = theta1
	

	# Getter & Setter of "theta2"
	def get_theta2 (self):
		
		"""

		This method returns the value of the variable self.theta2.
		self.theta2 contains an array with output layer weigths.

		"""

		return self.theta2
	
	def set_theta2 (self, theta2):
		
		"""

		This method modify the value of the variable self.theta2.
		This is replaced by the value contained in the theta2 variable passed as an argument.
		
		"""

		self.theta2 = theta2

	# Getter & Setter of "num_labels"
	def get_num_labels (self):
		
		"""

		This method returns the value of the variable self.num_labels.
		self.num_labels contains the number of labels in the problem.
		
		"""

		return self.num_labels
	
	def set_num_labels (self, num_labels):
		
		"""

		This method modify the value of the variable self.num_labels.
		This is replaced by the value contained in the num_labels variable passed as an argument.
		
		"""

		self.num_labels = num_labels


	# Getter & Setter of "m"
	def get_m (self):
		
		"""

		This method returns the value of the variable self.m.
		self.m contains the number of samples of X (train patterns array).
		
		"""

		return self.m
	
	def set_m (self, m):
		
		"""

		This method modify the value of the variable self.m.
		This is replaced by the value contained in the m variable passed as an argument.
		
		"""

		self.m = m

	#--------------Private Access functions------------------


	# Download and save the values ​​of theta1, theta2 and thresholds_param
	# from the nn_params array to their corresponding array
	def __unpack_parameters(self, nn_params, input_layer_size, hidden_layer_size, num_labels):
		
		"""

		This method gets theta1 and theta2 back from the whole array nn_params.

		Parameters
		----------

		nn_params: column array, shape ((imput_layer_size+1)*hidden_layer_size
		+ hidden_layer_size + (num_labels-1))
			Array that is a column vector. It stores the values ​​of theta1,
			theta2 and thresholds_param, all of them together in an array in this order.

		input_layer_size: integer
			Number of nodes in the input layer of the neural network model.
		
		hidden_layer_size: integer
			Number of nodes in the hidden layer of the neural network model.
			
		num_labels: integer
			Number of classes.


		Returns
		-------

		theta1: The weights between the input layer and the hidden layer (with biases included).

		theta2: The weights between the hidden layer and the output layer.

		"""

		n_theta1 = hidden_layer_size * (input_layer_size + 1)
		theta1 = np.reshape(nn_params[0:n_theta1],(hidden_layer_size,
		 (input_layer_size + 1)),order='F')
		
		theta2 = np.reshape(nn_params[n_theta1:], (num_labels-1,
		 hidden_layer_size+1),order='F')
		
		return theta1, theta2
	

	# Randomly initialize the weights of the neural network layer
	# by entering the number of input and output nodes of that layer
	def __rand_initialize_weights(self, L_in, L_out):

		"""

		This method randomly initializes the weights of a layer
		 with L_in incoming connections and L_out outgoing connections

		 Parameters
		----------

		L_in: integer
			Number of inputs of the layer.

		L_out: integer
			Number of outputs of the layer.
		
		Returns
		-------

		W: Array with the weights of each synaptic relationship between nodes.

		"""

		W = np.random.rand(L_out,L_in)*2*self.get_epsilon_init() - self.get_epsilon_init()

		return W


	# Implements the cost function and obtains the corresponding derivatives.
	def __nnop_cost_function(self, nn_params, input_layer_size, hidden_layer_size,
	num_labels, X, Y, lambda_value):
		
		"""
		This method implements the cost function and obtains
		the corresponding derivatives.
			
		Parameters
		----------

		nn_params: column array, shape ((imput_layer_size+1)*hidden_layer_size
		+ hidden_layer_size)
		
		Array that is a column vector. It stores the values ​​of Theta1 and
		Theta2, all of them together in an array in this order.
			
		input_layer_size: integer
			Number of nodes in the input layer of the neural network model.
		
		hidden_layer_size: integer
			Number of nodes in the hidden layer of the neural network model.
			
		num_labels: integer
			Number of classes.

		X: {array-like, sparse matrix}, shape (n_samples, n_features)
			Training patterns array, where n_samples is the number of samples
			and n_features is the number of features

		Y: array-like, shape (n_samples)
			Target vector relative to X

		lambda_value:
			Regularization parameter.

		Returns
		-------

		J: Matrix with cost function (updated weight matrix).
		grad: Array with the error gradient of each weight of each layer.

		"""

		# Unroll all the parameters
		theta1,theta2 = self.__unpack_parameters(nn_params,input_layer_size, hidden_layer_size, num_labels)

		# Setup some useful variables
		m = np.size(X, 0)

		# Neural Network model
		a1 = np.append(np.ones((m, 1)), X, axis=1)
		z2 = np.matmul(a1,theta1.T)
		a2 = np.append(np.ones((m, 1)), 1.0 / (1.0 + np.exp(-z2)), axis=1)
		z3 = np.matmul(a2,theta2.T)
		h = np.append(1.0 / (1.0 + np.exp(-z3)), np.ones((m, 1)), axis=1)

		# Final output
		out = h

		# Calculate penalty (regularización L2)
		p = np.sum((theta1[:,1:]**2).sum() + (theta2[:,1:]**2).sum())

		# MSE
		J = np.sum((out-Y)**2).sum()/(2*m) + lambda_value*p/(2*m)

		# MSE
		error_der = (out-Y)

		# Calculate sigmas
		sigma3 = np.multiply(np.multiply(error_der,h), (1-h))
		sigma3 = sigma3[:,:-1]

		sigma2 = np.multiply(np.multiply(np.matmul(sigma3, theta2), a2), (1-a2))
		sigma2 = sigma2[:,1:]

		# Accumulate gradients
		delta_1 = np.matmul(sigma2.T, a1)
		delta_2 = np.matmul(sigma3.T, a2)

		# Calculate regularized gradient
		p1 = (lambda_value/m) * np.concatenate((np.zeros((np.size(theta1, axis=0), 1)), theta1[:,1:]), axis=1)
		p2 = (lambda_value/m) * np.concatenate((np.zeros((np.size(theta2, axis=0), 1)), theta2[:,1:]), axis=1)
		theta1_grad = delta_1 / m + p1
		theta2_grad = delta_2 / m + p2

		# Unroll gradients
		grad = np.concatenate((theta1_grad.flatten(order='F'),
		 theta2_grad.flatten(order='F')),axis=0)

		return J,grad
	
