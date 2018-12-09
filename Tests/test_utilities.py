
import os, sys
import unittest

import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np

sys.path.append('../')
sys.path.append('../Algorithms')
from Utilities import Utilities


class TestAuxiliarMethods(unittest.TestCase):

	general_conf = {}
	configurations = {}

	util = Utilities(general_conf, configurations)

	def test_get_dataset_path(self):

		path = '/path/without/final/backslash'
		dataset = 'dataset'
		dataset_path = self.util._getDatasetPath(path, dataset)

		self.assertEqual(dataset_path, '/path/without/final/backslash/dataset/')

		path = '/path/with/final/backslash/'
		dataset = 'dataset'
		dataset_path = self.util._getDatasetPath(path, dataset)

		self.assertEqual(dataset_path, '/path/with/final/backslash/dataset/')


	def test_load_complete_dataset(self):

		# Loading dataset composed of 5 partitions, each one of them composed of a train and test file

		dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/test_datasets/test_load_dataset/complete/"
		partition_list = self.util._loadDataset(dataset_path)

		# Check all partitions have been loaded
		self.assertEqual(len(partition_list), ( len([name for name in os.listdir(dataset_path)]) / 2 ))
		# Check if every partition has train and test inputs and outputs (4 diferent dictionaries)
		self.assertTrue(all([ len(partition) == 4 for partition in partition_list ]))


	def test_load_partitionless_dataset(self):

		# This dataset is composed of only two csv files (train and test files)

		dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/test_datasets/test_load_dataset/partitionless/"
		partition_list = self.util._loadDataset(dataset_path)

		self.assertEqual(len(partition_list), 1)
		self.assertTrue(all([ len(partition) == 4 for partition in partition_list ]))


	def test_load_nontestfile_dataset(self):

		# Dataset composed of just five train files

		dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/test_datasets/test_load_dataset/nontestfile/"
		partition_list = self.util._loadDataset(dataset_path)

		self.assertEqual(len(partition_list), len([name for name in os.listdir(dataset_path)]))
		self.assertTrue(all([ len(partition) == 2 for partition in partition_list ]))


	def test_load_nontrainfile_dataset(self):

		# This dataset has 2 partitions, but one of them lacks it's train file

		# Trying to load a dataset where at least one of it's partitions
		# doesn't have a train file, should raise an exception

		dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/test_datasets/test_load_dataset/nontrainfile/"
		self.assertRaises(RuntimeError, self.util._loadDataset, dataset_path)


	def test_load_algorithm(self):

		# Loading a method from within this framework
		from OrdinalDecomposition import OrdinalDecomposition
		imported_class = self.util._loadAlgorithm("OrdinalDecomposition")
		self.assertEqual(imported_class, OrdinalDecomposition)

		# Loading a scikit-learn classifier
		from sklearn.svm import SVC
		imported_class = self.util._loadAlgorithm("sklearn.svm.SVC")
		self.assertEqual(imported_class, SVC)


		# Raising an exception if the algorithm path has not proper length
		# Understanding for length the strings splitted by dots
		# Only 1 or 3 length paths are allowed

		self.assertRaises(AttributeError, self.util._loadAlgorithm, "sklearn.svm")
		self.assertRaises(AttributeError, self.util._loadAlgorithm, "sklearn.svm.SVC.submethod")

		# Being unable to load an algorithm should return an exception
		self.assertRaises(ImportError, self.util._loadAlgorithm, "sklearn.svm.SVCC")


	def test_extract_params(self):

		# Normal use of configuration file with a non nested method
		params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 100], 'probability': "True"}
		formatted_params = self.util._extractParams(params)

		expected_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 100], 'probability': ["True"]}
		self.assertEqual(formatted_params, expected_params) 

		# Configuration file using a meta-clasifier method
		params = {	'dtype': 'OrderedPartitions', 
					'algorithm': 'sklearn.svm.SVC', 
					'parameters': {'C': [1, 10], 'gamma': [1, 10], 'probability': ['True']}}

		formatted_params = self.util._extractParams(params)

		expected_params = {	'dtype': ['OrderedPartitions'],
							'algorithm': ['sklearn.svm.SVC'],
							'parameters': [	{'C': 1.0, 'gamma': 1.0, 'probability': True},
											{'C': 1.0, 'gamma': 10.0, 'probability': True}, 
											{'C': 10.0, 'gamma': 1.0, 'probability': True}, 
											{'C': 10.0, 'gamma': 10.0, 'probability': True}]
						  }

		self.assertEqual(formatted_params, expected_params)



class TestMainMethod(unittest.TestCase):


	#TODO: Como testear el funcionamiento del main ???
	pass



# Running all tests
if __name__ == "__main__":
	unittest.main()













