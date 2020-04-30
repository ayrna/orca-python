from sys import path as syspath
from os import path as ospath
import ntpath
from shutil import rmtree
import gc

import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing

syspath.append('..')
syspath.append(ospath.join('..', 'classifiers'))

from utilities import Utilities


class TestRedsvmLoad(unittest.TestCase):
	"""
	Class testing REDSVM's functionality.

	This classifier is built in classifiers/REDSVM.py.
	"""

	# Getting path to datasets folder
	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_redsvm_svorex_load_dataset")

	# Parameters values for experiments
	values = np.logspace(-3, 3, 7).tolist()
	
	# Declaring a simple configuration
	general_conf = {"basedir": dataset_path,
					"datasets": ["automobile", "balance-scale", "bondrate", "car", "contact-lenses", "ERA", "ESL", "eucalyptus", "LEV", "newthyroid",
								 "pasture", "squash-stored", "squash-unstored", "SWD", "tae", "toy", "winequality-red"],
					"input_preprocessing": "std",
					"hyperparam_cv_nfolds": 3,
					"jobs": 10,
					"output_folder": "my_runs/",
					"metrics": ["ccr", "mae", "amae", "mze"],
					"cv_metric": "mae"}

	configurations = {
		"redsvm_linear": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 0,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_polynomial": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 1,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_radial": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 2,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_sigmoid": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 3,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_stump": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 4,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_perceptron": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 5,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_laplacian": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 6,
				"d": 2,
				"c": values,
				"g": values
			}

		},
		"redsvm_exponential": {

			"classifier": "REDSVM",
			"parameters": {
				"t": 7,
				"d": 2,
				"c": values,
				"g": values
			}

		}
	}
		
	def test_redsvm_load(self):
		gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)

		print("\n")
		print("###############################")
		print("REDSVM load test")
		print("###############################")

		# Declaring Utilities object and running the experiment
		util = Utilities(self.general_conf, self.configurations, verbose=True)
		util.run_experiment()
		# Saving results information
		util.write_report()

		#Delete all the test results after load test
		rmtree("my_runs")


if __name__ == '__main__':
	unittest.main()
