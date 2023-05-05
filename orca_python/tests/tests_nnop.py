from sys import path as syspath
from os import path as ospath

import unittest

import numpy as np
import numpy.testing as npt

# syspath.append(ospath.join('..', 'classifiers'))

# from NNOP import NNOP
from orca_python.classifiers.NNOP import NNOP


class TestNNOP(unittest.TestCase):
	"""
	Class testing NNOP's functionality.

	This classifier is built in classifiers/NNOP.py.
	"""

	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_nnpom_nnop_load_dataset", "toy")

	train_file = np.loadtxt(ospath.join(dataset_path,"train_toy.0"))
	test_file = np.loadtxt(ospath.join(dataset_path,"test_toy.0"))

	#	-----	NOT APPLIED	-----
	# It doesn't apply to the because can't set seed to randomize model weights.
	# def test_nnop_fit_correct(self):
	# 	#Check if this algorithm can correctly classify a toy problem.
		
	# 	#Test preparation
	# 	X_train = self.train_file[:,0:(-1)]
	# 	y_train = self.train_file[:,(-1)]

	# 	X_test = self.test_file[:,0:(-1)]
		
		# expected_predictions = [ospath.join(self.dataset_path,"expectedPredictions.0")]
								# ospath.join(self.dataset_path,"expectedPredictions.1"),
								# ospath.join(self.dataset_path,"expectedPredictions.2"),
								# ospath.join(self.dataset_path,"expectedPredictions.3")]

	# 	classifiers = [NNOP(epsilonInit = 0.5, hiddenN = 10, iterations = 500, lambdaValue = 0.01)]
		#			   NNOP(epsilonInit = 0.5, hiddenN = 20, iterations = 500, lambdaValue = 0.01),
		#			   NNOP(epsilonInit = 0.5, hiddenN = 10, iterations = 250, lambdaValue = 0.01),
		#			   NNOP(epsilonInit = 0.5, hiddenN = 20, iterations = 500, lambdaValue = 0.01)]


	# 	#Test execution and verification
	# 	for expected_prediction, classifier in zip(expected_predictions, classifiers):
	# 		classifier.fit(X_train, y_train)
	# 		predictions = classifier.predict(X_test)
	# 		expected_prediction = np.loadtxt(expected_prediction)
	# 		npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")

	def test_nnop_fit_not_valid_parameter(self):

		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifiers = [NNOP(epsilonInit=0.5, hiddenN=-1, iterations=1000, lambdaValue=0.01),
						NNOP(epsilonInit=0.5, hiddenN=10, iterations=-1, lambdaValue=0.01)]

		#Test execution and verification
		for classifier in classifiers:
				model = classifier.fit(X_train, y_train)
				self.assertIsNone(model, "The NNOP fit method doesnt return Null on error")

	def test_nnop_fit_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[0:(-1),0:(-2)]
		y_train_broken = self.train_file[0:(-1),(-1)]

		#Test execution and verification
		classifier = NNOP(epsilonInit=0.5, hiddenN=10, iterations=1000, lambdaValue=0.01)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The NNOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The NNOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The NNOP fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The NNOP fit method doesnt return Null on error")


	#	-----	NOT APPLIED	-----
	# It doesn't apply to the because it has no internal model
	# like in other classifiers like REDSVM or SVOREX. 
	# def test_nnop_model_is_not_a_dict(self):
	# 	#Test preparation
	# 	X_train = self.train_file[:,0:(-1)]
	# 	y_train = self.train_file[:,(-1)]

	# 	X_test = self.test_file[:,0:(-1)]

	# 	classifier = NNOP(epsilonInit = 0.5, hiddenN = 10, iterations = 500, lambdaValue = 0.01)
	# 	classifier.fit(X_train, y_train)

	# 	#Test execution and verification
	# 	with self.assertRaisesRegex(TypeError, "Model should be a dictionary!"):
	# 			classifier.classifier_ = 1
	# 			classifier.predict(X_test)


	def test_nnop_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifier = NNOP(epsilonInit = 0.5, hiddenN = 10, iterations = 500, lambdaValue = 0.01)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])

if __name__ == '__main__':
	unittest.main()
