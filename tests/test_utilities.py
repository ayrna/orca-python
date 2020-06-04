import os
from sys import path as syspath
from os import path as ospath
from shutil import rmtree
import unittest

import numpy.testing as npt
import pandas.testing as pdt
import pandas as pd
import numpy as np

syspath.append('..')
syspath.append(ospath.join('..', 'classifiers'))

from utilities import Utilities
from utilities import load_classifier


class TestAuxiliarMethods(unittest.TestCase):
	"""
	This class will test whether all different auxiliar functions
	built in Utilities class work as expected or not.

	This class can be found in utilities.py, file located at the
	root folder of this framework.
	"""

	general_conf = {}
	configurations = {}

	util = Utilities(general_conf, configurations)


	def test_load_complete_dataset(self):
		"""
		Loading dataset composed of 5 partitions, 
		each one of them composed of a train and test file
		"""

		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "complete")

		partition_list = self.util._load_dataset(dataset_path)

		# Check all partitions have been loaded
		npt.assert_equal(len(partition_list), (len([name for name in os.listdir(dataset_path)]) / 2))
		# Check if every partition has train and test inputs and outputs (4 diferent dictionaries)
		npt.assert_equal(all([len(partition[1]) == 4 for partition in partition_list]), True)


	def test_load_partitionless_dataset(self):
		"""
		Loading dataset composed of only two csv
		files (train and test files)
		"""

		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "partitionless")

		partition_list = self.util._load_dataset(dataset_path)

		npt.assert_equal(len(partition_list), 1)
		npt.assert_equal(all([len(partition[1]) == 4 for partition in partition_list]), True)


	def test_load_nontestfile_dataset(self):
		"""
		Loading dataset composed of five train files
		"""

		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "nontestfile")

		partition_list = self.util._load_dataset(dataset_path)

		npt.assert_equal(len(partition_list), len([name for name in os.listdir(dataset_path)]))
		npt.assert_equal(all([len(partition[1]) == 2 for partition in partition_list]), True)


	def test_load_nontrainfile_dataset(self):
		"""
		Loading dataset with 2 partitions, one of them lacking
		it's train file. This should raise an exception.
		"""

		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "nontrainfile")

		npt.assert_raises(RuntimeError, self.util._load_dataset, dataset_path)


	def test_normalize_data(self):
		#Test preparation
		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "partitionless")

		train_file = np.loadtxt(ospath.join(dataset_path,"train_partitionless.csv"))
		X_train = train_file[:,0:(-1)]

		test_file = np.loadtxt(ospath.join(dataset_path,"test_partitionless.csv"))
		X_test = test_file[:,0:(-1)]

		#Test execution
		norm_X_train, _= self.util._normalize_data(X_train, X_test)

		#Test verification
		result = (norm_X_train >= 0).all() and (norm_X_train <= 1).all()
		npt.assert_equal(result, True)


	def test_standardize_data(self):
		#Test preparation
		dataset_path = os.path.dirname(os.path.abspath(__file__))
		dataset_path = ospath.join(dataset_path, "test_datasets", "test_load_dataset", "partitionless")

		train_file = np.loadtxt(ospath.join(dataset_path,"train_partitionless.csv"))
		X_train = train_file[:,0:(-1)]

		test_file = np.loadtxt(ospath.join(dataset_path,"test_partitionless.csv"))
		X_test = test_file[:,0:(-1)]

		#Test execution
		std_X_train, _= self.util._standardize_data(X_train, X_test)

		#Test verification
		npt.assert_almost_equal(np.mean(std_X_train), 0)
		npt.assert_almost_equal(np.std(std_X_train), 1)


	def test_load_algorithm(self):

		# Loading a method from within this framework
		from OrdinalDecomposition import OrdinalDecomposition
		imported_class = load_classifier("OrdinalDecomposition")
		npt.assert_equal(imported_class, OrdinalDecomposition)

		# Loading a scikit-learn classifier
		from sklearn.svm import SVC
		imported_class = load_classifier("sklearn.svm.SVC")
		npt.assert_equal(imported_class, SVC)

		# Raising exceptions when the classifier cannot be loaded
		npt.assert_raises(ImportError, load_classifier, "sklearn.svm.SVC.submethod")
		npt.assert_raises(AttributeError, load_classifier, "sklearn.svm.SVCC")


	def test_check_params(self):
		"""
		Testing functionality of check_params method.

		It will test the 3 different scenarios contemplated
		within the framework for passing the configuration.
		"""


		# Normal use of configuration file with a non nested method
		self.util.configurations = {'conf1': {'classifier': 'sklearn.svm.SVC',
												'parameters': {'C': [0.1, 1, 10],
																'gamma': [0.1, 1, 100], 
																'probability': "True"}}}

		# Getting formatted_params and expected_params
		self.util._check_params(); formatted_params = self.util.configurations['conf1']['parameters']

		random_state = self.util.configurations['conf1']['parameters']['random_state']
		expected_params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 100], 'probability': ["True"], 'random_state': random_state}

		npt.assert_equal(formatted_params, expected_params) 


		# Configuration file using an ensemble method
		self.util.configurations = {'conf2': {'classifier': 'OrdinalDecomposition',
												'parameters': {'dtype': 'OrderedPartitions', 
															'base_classifier': 'sklearn.svm.SVC', 
															'parameters': {'C': [1, 10], 
																		'gamma': [1, 10], 
																		'probability': ['True']}}}}



		# Getting formatted_params and expected_params
		self.util._check_params(); formatted_params = self.util.configurations['conf2']['parameters']

		random_state = self.util.configurations['conf2']['parameters']['parameters'][0]['random_state']
		expected_params = {	'dtype': ['OrderedPartitions'],
							'base_classifier': ['sklearn.svm.SVC'],
							'parameters': 	[{'C': 1, 'gamma': 1, 'probability': True, 'random_state': random_state},
											{'C': 1, 'gamma': 10, 'probability': True, 'random_state': random_state}, 
											{'C': 10, 'gamma': 1, 'probability': True, 'random_state': random_state}, 
											{'C': 10, 'gamma': 10, 'probability': True, 'random_state': random_state}]
						  }


		# Ordering list of parameters from formatted_params to prevent inconsistencies
		formatted_params['parameters'] = sorted(formatted_params['parameters'], key=lambda k: k['C'])

		npt.assert_equal(expected_params, formatted_params)



		# Configuration file where it's not necessary to perform cross-validation
		self.util.configurations = {'conf3': {'classifier': 'OrdinalDecomposition',
												'parameters': {'dtype': 'OrderedPartitions', 
															'base_classifier': 'sklearn.svm.SVC', 
															'parameters': {'C': [1], 'gamma': [1]}}}}

		# Getting formatted_params and expected_params
		self.util._check_params(); formatted_params = self.util.configurations['conf3']['parameters']

		random_state = self.util.configurations['conf3']['parameters']['parameters']['random_state']
		expected_params = {	'dtype': 'OrderedPartitions',
							'base_classifier': 'sklearn.svm.SVC',
							'parameters': {'C': 1, 'gamma': 1, 'random_state': random_state}}

		npt.assert_equal(formatted_params, expected_params)


		# Resetting configurations to not interfere with other experiments
		self.util.configurations = {}



class TestMainMethod(unittest.TestCase):
	"""
	This class will test the proper behavior of the main
	method of this framework, "run_experiment".

	For this, a fixed configuration will be used.
	"""


	# Getting path to datasets folder
	main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	dataset_folder = os.path.join(main_folder, "datasets")

	# Declaring a simple configuration
	general_conf = {"basedir": dataset_folder,
					"datasets": ["tae", "contact-lenses"],
					"input_preprocessing": "std",
					"hyperparam_cv_nfolds": 3,
					"jobs": 10,
					"output_folder": "my_runs/",
					"metrics": ["ccr", "mae", "amae", "mze"],
					"cv_metric": "mae"}

	configurations = {
		"SVM": {
			"classifier": "sklearn.svm.SVC",
			"parameters": {
				"C": [0.001, 0.1, 1, 10, 100],
				"gamma": [0.1, 1, 10]
			}
		},
		"SVMOP": {
		
			"classifier": "OrdinalDecomposition",
			"parameters": {
				"dtype": "ordered_partitions",
				"decision_method": "frank_hall",
				"base_classifier": "sklearn.svm.SVC",
				"parameters": {
					"C": [0.01, 0.1, 1, 10],
					"gamma": [0.01, 0.1, 1, 10],
					"probability": ["True"]
	}}}}


	@npt.dec.slow
	def test_run_experiment(self):
		"""
		To test the main method, a configuration will be run
		until the end. Next we will check that every expected
		result file has been created, having all of them the
		proper dimensions and types.
		"""

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf, self.configurations, verbose=False)
		util.run_experiment()
		# Saving results information
		util.write_report()

		# Checking if all outputs have been generated and are correct
		outputs_folder = os.path.join(self.main_folder, "tests", "my_runs")
		npt.assert_equal(os.path.exists(outputs_folder), True)

		experiment_folder = sorted(os.listdir(outputs_folder))
		experiment_folder = os.path.join(outputs_folder, experiment_folder[-1])


		for dataset in util.general_conf['datasets']:
			for conf_name, _ in util.configurations.items():

				# Check if the folder for that dataset-configurations exists
				conf_folder = os.path.join(experiment_folder, (dataset + "-" + conf_name))
				npt.assert_equal(os.path.exists(conf_folder), True)

				# Checking CSV containning all metrics for that configuration
				metrics_csv = pd.read_csv(os.path.join(conf_folder, (dataset + "-" + conf_name + ".csv")))
				metrics_csv = metrics_csv.iloc[:,-12:]

				npt.assert_equal(metrics_csv.shape, (30, 12))
				npt.assert_equal(all(str(c) == "float64" for c in metrics_csv.dtypes), True)

				# Checking that all models have been saved
				models_folder = os.path.join(conf_folder, "models")
				npt.assert_equal(os.path.exists(models_folder), True)
				npt.assert_equal(len(os.listdir(models_folder)), 30)

				# Checking that all predictions have been saved
				predictions_folder = os.path.join(conf_folder, "predictions")
				npt.assert_equal(os.path.exists(predictions_folder), True)
				npt.assert_equal(len(os.listdir(predictions_folder)), 60)


		# Checking if summaries are correct
		train_summary = pd.read_csv(os.path.join(experiment_folder, "train_summary.csv"))
		npt.assert_equal(train_summary.shape, (4, 13))
		npt.assert_equal(all(str(c) == "float64" for c in train_summary.dtypes.iloc[1:]), True)

		test_summary = pd.read_csv(os.path.join(experiment_folder, "test_summary.csv"))
		npt.assert_equal(test_summary.shape, (4, 13))
		npt.assert_equal(all(str(c) == "float64" for c in test_summary.dtypes.iloc[1:]), True)

		rmtree(outputs_folder)

# Running all tests
if __name__ == "__main__":
	unittest.main()
