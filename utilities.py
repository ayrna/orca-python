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
	apply M different configurations over each dataset.

	Configurations are composed of a classifier method and different
	parameters, where it may be multiple values for every one of them.

	Running the main function of this class will perform
	cross-validation for each partition per dataset-configuration pairs,
	obtaining the most optimal model, after what will be used to infere
	the labels for the test sets.


	Parameters
	----------

	general_conf: dict
		Dictionary containing values needed to run the experiment.
		It gives this class information about where are located the
		different datasets, which one are going to be tested, the
		metrics to use, etc.

	configurations: dict
		Dictionary in which are stated the different classifiers
		to build methods upon the selected datasets, as well as
		the different values for the hyper-parameters used to
		optimize the model during cross-validation phase.

	For more usage information, read User Guide of this framework.


	Attributes
	----------

	_results: Results object
		Class used to manage and store all information obtained
		during the run of an experiment.
	"""


	def __init__(self, general_conf, configurations):


		self._general_conf = general_conf
		self._configurations = configurations


	def run_experiment(self):

		"""
		Runs an experiment. Main method of this framework.

		Loads all datasets, which can be fragmented in partitions.
		Builds a model per partition, using cross-validation to find
		the optimal values among the hyper-parameters to compare from.

		Uses the built model to get train and test metrics, storing all
		the information into a Results object.
		"""


		self._results = Results()
		# Adding classifier folder to sys path.
		path.insert(0, 'classifiers')

		self._check_dataset_list()
		self._check_params()

		print("\n###############################")
		print("\tRunning Experiment")
		print("###############################")

		# Iterating over Datasets
		for x in self._general_conf['datasets']:

			# Getting dataset name and path
			dataset_name = x.strip()
			dataset_path = os.path.join(self._general_conf['basedir'], dataset_name)

			# Loading dataset into a list of partitions.
			dataset = self._load_dataset(dataset_path)
			print("\nRunning", dataset_name, "dataset")
			print("--------------------------")


			# Iterating over Configurations
			for conf_name, configuration in self._configurations.items():
				print("Running", conf_name, "...")

				classifier = load_classifier(configuration["classifier"])

				# Iterating over partitions
				for part_idx, partition in enumerate(dataset):
					print("  Running Partition", part_idx)

					# Finding optimal classifier
					optimal_estimator = self._get_optimal_estimator(partition["train_inputs"], partition["train_outputs"],\
																	classifier, configuration["parameters"])

					# Getting train and test predictions
					train_predicted_y = optimal_estimator.predict(partition["train_inputs"])

					test_predicted_y = None; elapsed = np.nan
					if "test_outputs" in partition:
						start = time()
						test_predicted_y = optimal_estimator.predict(partition["test_inputs"])
						elapsed = time() - start



					# Obtaining train and test metric's values.
					train_metrics = OrderedDict(); test_metrics = OrderedDict()
					for metric_name in self._general_conf['metrics']:

						try:
							# Loading metric from file
							module = __import__("metrics")
							metric = getattr(module, metric_name.strip().lower())

						except AttributeError:
							raise AttributeError("No metric named '%s'" % metric_name.strip().lower())

						# Get train scores
						train_score = metric(partition["train_outputs"], train_predicted_y)
						train_metrics[metric_name.strip() + '_train'] = train_score

						# Get test scores
						test_metrics[metric_name.strip() + '_test'] = np.nan
						if "test_outputs" in partition:
							test_score = metric(partition["test_outputs"], test_predicted_y)
							test_metrics[metric_name.strip() + '_test'] = test_score


					# Cross-validation was performed to tune hyper-parameters
					if isinstance(optimal_estimator, GridSearchCV):
						train_metrics['cv_time_train'] = optimal_estimator.cv_results_['mean_fit_time'].mean()
						test_metrics['cv_time_test'] = optimal_estimator.cv_results_['mean_score_time'].mean()
						train_metrics['time_train'] = optimal_estimator.refit_time_
						test_metrics['time_test'] = elapsed


					else:
						optimal_estimator.best_params_ = configuration['parameters']
						optimal_estimator.best_estimator_ = optimal_estimator

						train_metrics['cv_time_train'] = np.nan
						test_metrics['cv_time_test'] = np.nan
						train_metrics['time_train'] = optimal_estimator.refit_time_
						test_metrics['time_test'] = elapsed


					# Saving this partition's results
					self._results.add_record(part_idx, optimal_estimator.best_params_, optimal_estimator.best_estimator_,\
											{'dataset': dataset_name, 'config': conf_name},\
											{'train': train_metrics, 'test': test_metrics},\
											{'train': train_predicted_y, 'test': test_predicted_y})



	def _load_dataset(self, dataset_path):

		"""
		Loads all dataset's files, divided into train and test.

		Parameters
		----------

		dataset_path: string
			Path to dataset folder.


		Returns
		-------

		partition_list: list of dicts
			List of partitions found inside a dataset folder.
			Each partition is stored into a dictionary, disjoining
			train and test inputs, and outputs.
		"""


		try:

			# Creating dicts for all partitions (saving partition order as keys)
			partition_list = {filename[filename.find('.') + 1:]: {} for filename in os.listdir(dataset_path)\
								if filename.startswith("train_")}

			# Loading each dataset
			for filename in os.listdir(dataset_path):

				if filename.startswith("train_"):
					train_inputs, train_outputs = self._read_file(os.path.join(dataset_path, filename))
					partition_list[filename[filename.find('.') + 1:]]["train_inputs"] = train_inputs
					partition_list[filename[filename.find('.') + 1:]]["train_outputs"] = train_outputs

				elif filename.startswith("test_"):
					test_inputs, test_outputs = self._read_file(os.path.join(dataset_path, filename))
					partition_list[filename[filename.find('.') + 1:]]["test_inputs"] = test_inputs
					partition_list[filename[filename.find('.') + 1:]]["test_outputs"] = test_outputs

		except OSError:
			raise ValueError("No such file or directory: '%s'" % dataset_path)

		except KeyError:
			raise RuntimeError("Found partition without train files: partition %s" % filename[filename.find('.') + 1:])


		# Saving partitions as a sorted list of dicts (according to it's partition order)
		partition_list = list(OrderedDict(sorted(partition_list.items(), key=(lambda t: get_key(t[0])))).values())

		return partition_list



	def _read_file(self, filename):

		"""
		Reads a CSV containing partitions, or full datasets.
		Train and test files must be previously divided for
		the experiment to run.

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



	def _check_dataset_list(self):

		"""
		Checks if there is some inconsistency in the dataset list.
		It also simplifies running all datasets inside one folder.

		Parameters
		----------
		dataset_list: list of strings
			list containing all the dataset names to run in a given
			experiment.
			If 'all' is specified without any other string, then all
			datasets in basedir folder will be run.

		"""


		base_path = self._general_conf['basedir']
		dataset_list = self._general_conf['datasets']

		# Check if home path is shortened
		if base_path.startswith("~"):
			base_path = base_path.replace('~', os.path.expanduser('~'), 1)


		# Compatibility between python 2 and 3
		try:
			basestring = (unicode, str)
		except NameError:
			basestring = str

		# Check if 'all' is the only value, and if it is, expand it
		if len(dataset_list) == 1 and dataset_list[0] == 'all':

			dataset_list = [ item for item in os.listdir(base_path) \
						if os.path.isdir(os.path.join(base_path, item)) ]


		elif not all(isinstance(item, basestring) for item in dataset_list):
			raise ValueError("Dataset list can only contain strings")


		self._general_conf['basedir'] = base_path
		self._general_conf['datasets'] = dataset_list



	def _check_params(self):

		"""
		Checks if all given configurations are sintactly correct.

		Performs two different transformations over parameter
		dictionaries when needed. Those consist of:

		- If one parameter's values are not inside a list, GridSearchCV
		will not be able to handle them, so they must be enclosed into one.

		- When an ensemble method, as OrderedPartitions, is chosen as
		classifier, transforms the dict of lists in which the
		parameters for the internal classifier are stated into a list
		of dicts (all possible combiantions of those different parameters).
		"""

		random_seed = np.random.get_state()[1][0]
		for _, conf in self._configurations.items():

			parameters = conf['parameters']

			# If parameter is a dict named 'parameters', then an ensemble method it's being used
			# we need to transform a dict of lists, into a list of dicts.
			if 'parameters' in parameters and type(parameters['parameters'] == dict):

				# Using given seed as random_state value
				parameters['parameters']['random_state'] = [random_seed]

				try:

					# Creating a list for each parameter. Elements represented as 'parameterName-parameterValue'.
					p_list = [ [p_name + ';' + str(v) for v in p] for p_name, p in parameters['parameters'].items() ]
					# Permutations of all lists. Generates all possible combination of elements between lists.
					p_list = [ list(item) for item in list(product(*p_list)) ]
					# Creates a list of dictionaries, containing all combinations of given parameters
					p_list = [ dict( [item.split(';') for item in p] ) for p in p_list ]

				except TypeError:
					raise TypeError('All parameters for the inner classifier must be an iterable object')


				# Returns non-string values back to it's normal self
				for d in p_list:
					for (k, v) in d.items():

						if is_int(v):
							d[k] = int(v)
						elif is_float(v):
							d[k] = float(v)
						elif is_boolean(v):
							d[k] = bool(v)

				parameters['parameters'] = p_list


			# No ensemble classifier was specified
			else:
				# Using given seed as random_state value
				parameters['random_state'] = [random_seed]


			# If there is just one value per parameter, it won't be necessary to use GridSearchCV
			if all(not isinstance(p, list) or len(p) == 1 for _, p in parameters.items()):
				# Pop out of list the lonely values
				for p_name, p in parameters.items():
					if isinstance(p, list):
						parameters[p_name] = p[0]

			# There are enough parameters to perform cross-validation
			else:
				# Convert non-list values to lists
				for p_name, p in parameters.items():
					if not isinstance(p, list) and not isinstance(p, dict):
						parameters[p_name] = [p]



	def _get_optimal_estimator(self, train_inputs, train_outputs, classifier, parameters):

		"""
		Perform cross-validation over one dataset and configuration.

		Each configuration consists of one classifier and none, one or
		multiple hyper-parameters, that, in turn, can contain one or
		multiple values used to optimize the resulting model.

		At the end of cross-validation phase, the model with the
		especific combination of values from the hyper-parameters
		that achieved the best metrics from all the combinations
		will remain.

		Parameters
		----------

		train_inputs: {array-like, sparse-matrix}, shape (n_samples, n_features)
			vector of features for each sample for this dataset.

		train_outputs: array-like, shape (n_samples)
			Target vector relative to train_inputs.

		classifier: object
			Class implementing a mathematical model able to be trained
			and to perform predictions over given datasets.

		parameters: dictionary
			Dictionary containing parameters to optimize as keys,
			and the list of values that we want to compare as values.

		Returns
		-------

		optimal: GridSearchCV object
			An already fitted model of the given classifier,
			with the best found parameters after cross-validation
		"""


		# No need to cross-validate when there is just one value per parameter
		if all(not isinstance(p, list) for k, p in parameters.items()):

			optimal = classifier(**parameters)

			start = time()
			optimal.fit(train_inputs, train_outputs)
			elapsed = time() - start

			optimal.refit_time_ = elapsed
			return optimal


		# More than one value per parameter. Cross-validation needed.
		try:
			module = __import__("metrics")
			metric = getattr(module, self._general_conf['cv_metric'].lower().strip())

		except AttributeError:

			if type(self._general_conf['cv_metric']) == list:
				raise AttributeError("Cross-Validation Metric must be a string")

			raise AttributeError("No metric named '%s'" % self._general_conf['cv_metric'].strip().lower())


		gib = module.greater_is_better(self._general_conf['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)

		optimal = GridSearchCV(estimator=classifier(), param_grid=parameters, scoring=scoring_function,\
					n_jobs=self._general_conf['jobs'], cv=self._general_conf['hyperparam_cv_nfolds'], iid=False)

		optimal.fit(train_inputs, train_outputs)

		return optimal




	def write_report(self):

		"""
		Saves information about experiment through Results class
		"""

		print("\nSaving Results...")

		# Names of each metric used (plus computational times)
		metrics_names = [x.strip().lower() for x in self._general_conf['metrics']] + ["cv_time", "time"]

		# Saving results through Results class
		self._results.save_results(self._general_conf['output_folder'], metrics_names)



##########################
# END OF UTILITIES CLASS #
##########################



def load_classifier(classifier_path, params=None):

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
		Returns a loaded classifier, either from an scikit-learn module,
		or from a module of this framework.
		Depending if hyper-parameters are specified, the object will be
		instantiated or not.

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



def is_int(value):

	"""
	Check if an string can be converted to int.

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



def is_float(value):

	"""
	Check if an string can be converted to float.

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



def is_boolean(value):

	"""
	Check if an string can be converted to boolean.

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



def get_key(key):

	"""
	Checks if the key of a dict can be converted to int,
	if not, returns the key as is.

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










