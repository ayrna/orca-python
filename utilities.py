from __future__ import print_function

import os
from time import time
from collections import OrderedDict
from itertools import product
from sys import path

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

from results import Results

class Utilities:

	"""

	Utilities

	Class in charge of running an experiment over N datasets, where we
	apply M different configurations over each one of them.

	Configurations are composed of a classifier and different parameters,
	where it may be multiple values for every parameter.

	This function will find, for all dataset-configuration pairs, the best
	value for each parameter of classifier applying cross-validation, after 
	what it will train that model and test it's accurancy over the dataset.


	Parameters
	----------

	general_conf: dict
		Dictionary that contains values needed to run the experiment
		itself. It gives this class info as to where are located the
		different datasets, which one are going to be tested, the metrics
		to use, etc.

	configurations: dict
		Dictionary in which are stated the different classifiers
		to build methods upon the selected datasets, as well as
		parameters with different possible values of which found the
		best combination through cross-validation.

	For more usage information, read User Guide of this framework.


	Attributes
	----------

	results_: Results object
		Class used to manage and store all information obtained when
		testing the different built models during the run of an experiment.
	"""


	def __init__(self, general_conf, configurations):


		self.general_conf_ = general_conf
		self.configurations_ = configurations


	def runExperiment(self):

		"""
		Runs an experiment. Main method of this framework.

		Builds one model for each possible combination of dataset and 
		configuration entry stated in it's corresponding configuration file.

		Loads every dataset, which can be fragmented in different partitions.
		Builds a model for every partition, using cross-validation for finding
		the best possible values for the different parameters of that actual
		configuration entry.
		Get train and test metrics for each dataset-config pair.


		"""


		self.results_ = Results()
		# Adding classifier folder to sys path. Needed to import modules from different folders
		path.insert(0, 'Classifiers/')


		self._checkDatasetList()
		self._checkParams()


		print("\n###############################")
		print("\tRunning Experiment")
		print("###############################")


		# Iterating over Datasets
		for x in self.general_conf_['datasets']:


			# Getting dataset name and path, stripped out of whitespaces
			dataset_name = x.strip()
			dataset_path = self._getDatasetPath(self.general_conf_['basedir'], dataset_name)

			# Loading dataset into a list of partitions. Each partition represented as a dictionary
			# containing train and test inputs/outputs. It also stores its partition number

			dataset = self._loadDataset(dataset_path)
			print("\nRunning", dataset_name, "dataset")
			print("--------------------------")


			# Iterating over all different Configurations
			for conf_name, configuration in self.configurations_.items():
				print("Running", conf_name, "...")


				# Loading Classifier stated in configuration
				classifier = loadClassifier(configuration["classifier"])


				# Iterating over all partitions in each dataset
				for idx, partition in enumerate(dataset):
					print("  Running Partition", idx)


					# Finding optimal parameters
					optimal_estimator = self._getOptimalEstimator(partition["train_inputs"], partition["train_outputs"],\
																  classifier, configuration["parameters"])

					# Getting predictions for train and test datasets
					train_predicted_y = optimal_estimator.predict(partition["train_inputs"])
					if "test_outputs" in partition:

						start = time()
						test_predicted_y = optimal_estimator.predict(partition["test_inputs"])
						elapsed = time() - start

					else:

						test_predicted_y = None
						elapsed = 0


					# Creating tuples with each specified tuple and passing it to specified dataframe
					train_metrics = OrderedDict()
					test_metrics = OrderedDict()

					# Iterating over Metrics
					for metric_name in self.general_conf_['metrics']:

						try:
							# Loading metric from metrics file
							module = __import__("Metrics")
							metric = getattr(module, metric_name.strip().lower())

						except AttributeError:
							raise AttributeError("No metric named '%s'" % metric_name.strip().lower())


						# Get train scores
						train_score = metric(partition["train_outputs"], train_predicted_y)
						train_metrics[metric_name.strip() + '_train'] = train_score


						# Get test scores
						if "test_outputs" in partition:

							test_score = metric(partition["test_outputs"], test_predicted_y)
							test_metrics[metric_name.strip() + '_test'] = test_score

						else:

							test_metrics[metric_name.strip() + '_test'] = np.nan



					# Saving time taken to cross-validate hyperparameters and re-fitting time of best model
					train_metrics['crossval_time'] = optimal_estimator.cv_results_['mean_fit_time'].mean()
					test_metrics['validation_time'] = optimal_estimator.cv_results_['mean_score_time'].mean()
					train_metrics['train_time'] = optimal_estimator.refit_time_
					test_metrics['test_time'] = elapsed

					# Save this partition's results
					self.results_.addRecord(idx, optimal_estimator.best_params_, optimal_estimator.best_estimator_,\
											{'dataset': dataset_name, 'config': conf_name},\
											{'train': train_metrics, 'test': test_metrics},\
											{'train': train_predicted_y, 'test': test_predicted_y})



	def _getDatasetPath(self, base_path, dataset_name):

		"""
		Gets path to actual dataset.

		Parameters
		----------

		base_path: string
			Base path in which dataset folder can be found.
			It can be absolute or relative.

		dataset_name: string
			Name given to dataset folder


		Returns
		-------

		dataset_path: string
			Path to folder containing dataset files
		"""


		# Check if basedir has a final backslash or not
		if base_path[-1] == '/':
			dataset_path = base_path + dataset_name + '/'
		else:
			dataset_path = base_path + "/" + dataset_name + '/'

		return dataset_path



	def _loadDataset(self, dataset_path):

		"""
		Loads all datasets files, divided into train and test.

		Parameters
		----------

		dataset_path: string
			Path to dataset folder


		Returns
		-------

		partition_list: list of dicts
			List of partitions found inside a dataset folder.
			Each partition is stored into a dictionary, disjoining
			train and test inputs, and outputs
		"""


		try:

			# TODO: Usar mejor el nombre completo del fichero (sin la parte de train) ??


			# Creating dicts for all partitions (saving partition order as keys)
			partition_list = { filename[ filename.find('.') + 1 : ]: {} for filename in os.listdir(dataset_path)\
								if filename.startswith("train_") }


			# Saving info for every dataset file
			for filename in os.listdir(dataset_path):

				if filename.startswith("train_"):

					train_inputs, train_outputs = self._readFile(dataset_path + filename)
					partition_list[filename[ filename.find('.') + 1 : ]]["train_inputs"] = train_inputs
					partition_list[filename[ filename.find('.') + 1 : ]]["train_outputs"] = train_outputs


				elif filename.startswith("test_"):

					test_inputs, test_outputs = self._readFile(dataset_path + filename)
					partition_list[filename[ filename.find('.') + 1 : ]]["test_inputs"] = test_inputs
					partition_list[filename[ filename.find('.') + 1 : ]]["test_outputs"] = test_outputs

		except OSError:
			raise ValueError("No such file or directory: '%s'" % dataset_path)

		except KeyError:
			raise RuntimeError("Found partition without train files: partition %s" % filename[ filename.find('.') + 1 : ])


		# Saving partitions as a sorted list of dicts (according to it's partition order)
		partition_list = list( OrderedDict(sorted(partition_list.items(), key=(lambda t: getKey(t[0])))).values() )

		return partition_list



	def _readFile(self, filename):

		"""
		Reads a CSV containing a partition or the entirety of a 
		dataset (train and test files must be previously divided for 
		the experiment to run though).

		Parameters
		----------

		filename: string
			Full path to train or test file.


		Returns
		-------

		inputs: {array-like, sparse-matrix}, shape (n_samples, n_features)
			vector of sample's features.

		outputs: array-like, shape (n_samples)
			Target vector relative to inputs.

		"""

		# Separator is automatically found
		f = pd.read_csv(filename, header=None, engine='python', sep=None)

		inputs = f.values[:,0:(-1)]
		outputs = f.values[:,(-1)]

		return inputs, outputs



	def _checkDatasetList(self):

		"""
		Checks if there is some inconsistency in the dataset list.
		It also simplifies running all datasets inside one folder.

		Parameters
		----------
		dataset_list: list of strings
			list containing all the dataset names to run in given experiment.
			If 'all' is specified without any other string, then all datasets
			in basedir folder will be run.

		"""


		base_path = self.general_conf_['basedir']
		dataset_list = self.general_conf_['datasets']

		# Check if home path is shortened
		if base_path.startswith("~"):
			base_path = base_path.replace('~', os.path.expanduser('~'), 1)



		# Compatibility between python 2 and 3
		try:
			basestring = unicode
		except NameError:
			basestring = str


		# Check if 'all' it's the only value, and if it is, expand it
		if len(dataset_list) == 1 and dataset_list[0] == 'all':

			dataset_list = [ item for item in os.listdir(base_path) \
						if os.path.isdir(os.path.join(base_path, item)) ]


		elif not all(isinstance(item, basestring) for item in dataset_list):
			raise ValueError("Dataset list can only contain strings")


		self.general_conf_['basedir'] = base_path
		self.general_conf_['datasets'] = dataset_list




	def _checkParams(self):

		"""
		Checks if all given configurations are correct.

		Performs two different transformations over parameter dictionaries
		when needed. Those consist of:

		- If one parameter's values are not inside a list, GridSearchCV will not be
		  able to handle them, so they must be enclosed into one.

		- When an ensemble method, as OrderedPartitions, is chosen as classifier,
		  transforms the dict of lists in which the parameters for the internal
		  classifier are stated into a list of dicts (all possible combiantions of
		  those different parameters).


		"""

		random_seed = np.random.get_state()[1][0]
		for conf_name, conf in self.configurations_.items():

			parameters = conf['parameters']

			# If parameter is a dict named 'parameters', then an ensemble method it's being used
			# we need to transform a dict of lists, into a list of dicts.
			if 'parameters' in parameters and type(parameters['parameters'] == dict):

				# Using given seed as random_state value
				parameters['parameters']['random_state'] = [random_seed]

				try:

					#TODO: Si hay algun parametro que contenga ';', falla (se esta usando para representar los valores)

					# Creating a list for each parameter. Elements represented as 'parameterName-parameterValue'.
					p_list = [ [p_name + ';' + str(v) for v in p] for p_name, p in parameters['parameters'].items() ]
					# Permutations of all lists. Generates all possible combination of elements between lists.
					p_list = [ list(item) for item in list(product(*p_list)) ]
					# Creates a list of dictionaries, containing all combinations of given parameters
					p_list = [ dict( [item.split(';') for item in p] ) for p in p_list ]

				except TypeError:
					raise TypeError('All parameters for the inner classifier must be an iterable object')


				# TODO: Debe haber una forma mas eficiente de hacer esto
				# Returns non-string values back to it's normal self
				for d in p_list:
					for (k, v) in d.items():

						if isInt(v):		#TODO: Solamente se usa para random_state (no admite floats)
							d[k] = int(v)
						elif isFloat(v):
							d[k] = float(v)
						elif isBoolean(v):
							d[k] = bool(v)

				parameters['parameters'] = p_list


			else:

				# Using given seed as random_state value
				parameters['random_state'] = [random_seed]


			for param_name, param in parameters.items():

				# If parameter is not a list, convert it into one
				if (type(param) != list) and (type(param) != dict):
					parameters[param_name] = [param]





	def _getOptimalEstimator(self, train_inputs, train_outputs, classifier, parameters):

		"""
		Perform cross-validation technique for finding best
		hyper-parameters out of all the options given in configuration file.

		Parameters
		----------

		train_inputs: {array-like, sparse-matrix}, shape (n_samples, n_features)
			vector of features for each sample for this dataset.

		train_outputs: array-like, shape (n_samples)
			Target vector relative to train_inputs.

		classifier: Classifier Object
			Class implementing an mathematical model able to be trained and to
			perform predictions over given datasets.

		parameters: dictionary
			Dictionary containing parameters to cross-validate as keys
			and the list of values that want to be compared as values.

		Returns
		-------

		optimal: GridSearchCV object
			An already fitted model of the given classifier,
			with the best found parameters after performing cross-validation
			over train samples from given partition (or dataset)
		"""

		try:
			module = __import__("Metrics")
			metric = getattr(module, self.general_conf_['cv_metric'].lower().strip())

		except AttributeError:

			if type(self.general_conf_['cv_metric']) == list:
				raise AttributeError("Cross-Validation Metric must be a string")

			raise AttributeError("No metric named '%s'" % self.general_conf_['cv_metric'].strip().lower())

		gib = module.greater_is_better(self.general_conf_['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)


		optimal = GridSearchCV(estimator=classifier(), param_grid=parameters, scoring=scoring_function,\
					n_jobs=self.general_conf_['jobs'], cv=self.general_conf_['folds'], iid=False)

		optimal.fit(train_inputs, train_outputs)

		return optimal




	def writeReport(self):

		"""
		Saves information about experiment through Results class

		"""


		print("\nSaving Results...")


		# Names of each metric used
		metrics_names = [x.strip().lower() for x in self.general_conf_['metrics']]

		self.results_.saveResults(self.general_conf_['runs_folder'], metrics_names)



##########################
# END OF UTILITIES CLASS #
##########################



def loadClassifier(classifier_path, params=None):

	"""
	Loads and returns a classifier.

	Parameters
	----------

	classifier_path: string
		Package path where the classifier class is located in.
		That module can be local if the classifier is built inside the
		framework, or relative to scikit-learn package.

	params: dictionary
		Parameters to initialize the classifier with. Used when loading a
		classifiers inside of an ensemble algorithm.


	Returns
	-------

	classifier: object
		Returns a loaded classifier, either from an scikit-learn module, or from
		a module of this framework.
		Depending is params are specified, the object will instantiated or not.

	"""

	if (len(classifier_path.split('.')) == 1):

		classifier = __import__(classifier_path)
		classifier = getattr(classifier, classifier_path)

	else:

		classifier = __import__(classifier_path.rsplit('.', 1)[0], fromlist="None")
		classifier = getattr(classifier, classifier_path.rsplit('.', 1)[1])

	if params is not None:
		classifier = classifier(**params)

	return classifier



def isInt(value):

	"""
	Check if an string can be converted to int

	Parameters
	----------
	value: string

	Returns
	-------
	Int
	"""

	try:
		int(value)
		return True
	except ValueError:
		return False

def isFloat(value):

	"""
	Check if an string can be converted to float

	Parameters
	----------
	value: string

	Returns
	-------
	Float
	"""

	try:
		float(value)
		return True
	except ValueError:
		return False



def isBoolean(value):

	"""
	Check if an string can be converted to Boolean

	Parameters
	----------
	value: string

	Returns
	-------
	Boolean
	"""


	if value == "True" or value == "False":
		return True
	else:
		return False


def getKey(key):

	"""
	Checks if the key of a dict can be converted to int, if not, returns the key as is

	Parameters
	----------
	value: string

	Returns
	-------
	int or boolean
	"""

	try:
		return int(key)
	except ValueError:
		return key










