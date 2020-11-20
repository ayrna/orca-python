from sys import path as syspath
from os import path as ospath

import unittest

import numpy as np
import numpy.testing as npt

syspath.append(ospath.join('..', 'classifiers'))

from REDSVM import REDSVM


class TestRedsvm(unittest.TestCase):
	"""
	Class testing REDSVM's functionality.

	This classifier is built in classifiers/REDSVM.py.
	"""

	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_redsvm_dataset")
	
	train_file = np.loadtxt(ospath.join(dataset_path,"train.0"))
	test_file = np.loadtxt(ospath.join(dataset_path,"test.0"))

	def test_redsvm_fit_correct(self):
		#Check if this algorithm can correctly classify a toy problem.
		
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

		expected_predictions = [ospath.join(self.dataset_path,"expectedPredictions.0"), 
								ospath.join(self.dataset_path,"expectedPredictions.1"),
								ospath.join(self.dataset_path,"expectedPredictions.2"),
								ospath.join(self.dataset_path,"expectedPredictions.3"),
								ospath.join(self.dataset_path,"expectedPredictions.4"),
								ospath.join(self.dataset_path,"expectedPredictions.5"),
								ospath.join(self.dataset_path,"expectedPredictions.6"),
								ospath.join(self.dataset_path,"expectedPredictions.7")]

		classifiers = [REDSVM(t=0, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=0),
					REDSVM(t=1, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=0),
					REDSVM(t=2, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=0),
					REDSVM(t=3, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=0),
					REDSVM(t=4, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=0),
					REDSVM(t=5, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=1),
					REDSVM(t=6, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=1),
					REDSVM(t=7, d=2, g=0.1, r=0.5, c=0.1, m=150, e=0.005, h=1)]

		#Test execution and verification
		for expected_prediction, classifier in zip(expected_predictions, classifiers):
			classifier.fit(X_train, y_train)
			predictions = classifier.predict(X_test)
			expected_prediction = np.loadtxt(expected_prediction)
			npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")

	def test_redsvm_fit_not_valid_parameter(self):

		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifiers = [REDSVM(g=0.1, c=1, t=-1),
					REDSVM(g=0.1, c=1, m=-1),
					REDSVM(g=0.1, c=1, e=-1),
					REDSVM(g=0.1, c=1, h=2)]
		
		error_msgs = ["unknown kernel type",
					"cache_size <= 0",
					"eps <= 0",
					"shrinking != 0 and shrinking != 1"]

		#Test execution and verification
		for classifier, error_msg in zip(classifiers, error_msgs):
			with self.assertRaisesRegex(ValueError, error_msg):
				model = classifier.fit(X_train, y_train)
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

	def test_redsvm_fit_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]
		X_train_broken = self.train_file[:(-1),0:(-1)]
		y_train_broken = self.train_file[0:(-1),(-1)]

		#Test execution and verification
		classifier = REDSVM(g=0.1, c=1, t=8)
		with self.assertRaisesRegex(ValueError, "Wrong input format: sample_serial_number out of range"):
				model = classifier.fit(X_train, y_train)
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

		classifier = REDSVM(g=0.1, c=1)
		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, y_train_broken)
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit([], y_train)
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train, [])
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

		with self.assertRaises(ValueError):
				model = classifier.fit(X_train_broken, y_train)
				self.assertIsNone(model, "The REDSVM fit method doesnt return Null on error")

	def test_redsvm_model_is_not_a_dict(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		X_test = self.test_file[:,0:(-1)]

		classifier = REDSVM(g=0.1, c=1)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaisesRegex(TypeError, "Model should be a dictionary!"):
				classifier.classifier_ = 1
				classifier.predict(X_test)

	def test_redsvm_predict_not_valid_data(self):
		#Test preparation
		X_train = self.train_file[:,0:(-1)]
		y_train = self.train_file[:,(-1)]

		classifier = REDSVM(g=0.1, c=1)
		classifier.fit(X_train, y_train)

		#Test execution and verification
		with self.assertRaises(ValueError):
			classifier.predict([])


if __name__ == '__main__':
	unittest.main()
