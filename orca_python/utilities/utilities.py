from __future__ import print_function

import os
from time import time
from collections import OrderedDict
from itertools import product
from sys import path as syspath
from copy import deepcopy
import pandas as pd

from ast import literal_eval
from pkg_resources import parse_version, get_distribution

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn import preprocessing

from orca_python.results import Results


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

	verbose: boolean
		Variable used for testing purposes. Silences all prints.

	For more usage information, read User Guide of this framework.


	Attributes
	----------

	_results: Results object
		Class used to manage and store all information obtained
		during the run of an experiment.
	"""


	def __init__(self, general_conf, configurations, verbose=True):


		self.general_conf = deepcopy(general_conf)
		self.configurations = deepcopy(configurations)
		self.verbose = verbose

		syspath.append('classifiers')


	def run_experiment(self):

		"""
		Runs an experiment. Main method of this framework.

		Loads all datasets, which can be fragmented in partitions.
		Builds a model per partition, using cross-validation to find
		the optimal values among the hyper-parameters to compare from.

		Uses the built model to get train and test metrics, storing all
		the information into a Results object.
		"""

		self._results = Results(self.general_conf['output_folder'])

		self._check_dataset_list()
		self._check_params()


		if self.verbose:
			print("\n###############################")
			print("\tRunning Experiment")
			print("###############################")

		# Iterating over Datasets
		for x in self.general_conf['datasets']:

			dataset_name = x.strip()
			dataset_path = os.path.join(self.general_conf['basedir'], dataset_name)


			dataset = self._load_dataset(dataset_path)

			if self.verbose:
				print("\nRunning", dataset_name, "dataset")
				print("--------------------------")


			# Iterating over Configurations
			for conf_name, configuration in self.configurations.items():

				if self.verbose:
					print("Running", conf_name, "...")


				classifier = load_classifier(configuration["classifier"])

				# Iterating over partitions
				for part_idx, partition in dataset:

					if self.verbose:
						print("  Running Partition", part_idx)


					#Normalization or Standardization of the partition if requested
					if self.general_conf['input_preprocessing'].strip().lower() == 'norm':
						partition["train_inputs"], partition["test_inputs"] = self._normalize_data(partition["train_inputs"], partition["test_inputs"])
					elif self.general_conf['input_preprocessing'].strip().lower() == 'std':
						partition["train_inputs"], partition["test_inputs"] = self._standardize_data(partition["train_inputs"], partition["test_inputs"])

					elif self.general_conf['input_preprocessing'].strip().lower() != '':
						raise AttributeError("Input preprocessing named '%s' unknown" % self.general_conf['input_preprocessing'].strip().lower())

					optimal_estimator = self._get_optimal_estimator(partition["train_inputs"],
																	partition["train_outputs"],
																	classifier,
																	configuration["parameters"])

					# Getting train and test predictions
					train_predicted_y = optimal_estimator.predict(partition["train_inputs"])

					test_predicted_y = None; elapsed = np.nan
					if "test_outputs" in partition:
						start = time()
						test_predicted_y = optimal_estimator.predict(partition["test_inputs"])
						elapsed = time() - start


					# Obtaining train and test metrics values.
					train_metrics = OrderedDict(); test_metrics = OrderedDict()
					for metric_name in self.general_conf['metrics']:

						try:
							# Loading metric from file
							module = __import__("orca_python").metrics
							metric = getattr(module, self.general_conf['cv_metric'].lower().strip())

						except AttributeError:
							raise AttributeError("No metric named '%s'" 
												% metric_name.strip().lower())

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


					# Saving the results for this partition
					self._results.add_record(part_idx, optimal_estimator.best_params_,
											optimal_estimator.best_estimator_,
											{'dataset': dataset_name, 'config': conf_name},
											{'train': train_metrics, 'test': test_metrics},
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

		partition_list: list of tuples
			List of partitions found inside a dataset folder.
			Each partition is stored into a dictionary, disjoining
			train and test inputs and outputs.
		"""


		try:

			# Creating dicts for all partitions (saving partition order as keys)
			partition_list = {filename[filename.find('.') + 1:]: {} for filename
																	in os.listdir(dataset_path)
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
			raise RuntimeError("Found partition without train files: partition %s"
								% filename[filename.find('.') + 1:])


		# Saving partitions as a sorted list of (index, partition) tuples
		partition_list = sorted(partition_list.items(), key=(lambda t: get_key(t[0])))

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
			Vector of sample's features.

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


		base_path = self.general_conf['basedir']
		dataset_list = self.general_conf['datasets']

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

			dataset_list = [item for item in os.listdir(base_path) \
								if os.path.isdir(os.path.join(base_path, item))]

		elif not all(isinstance(item, basestring) for item in dataset_list):
			raise ValueError("Dataset list can only contain strings")


		self.general_conf['basedir'] = base_path
		self.general_conf['datasets'] = dataset_list



	def _normalize_data(self, train_data, test_data):

		"""
		Normalize the data. Test data normalization will be based on train data

		Parameters
		----------
		train_data: 2d array
			contain the train data features
		test_data: 2d array
			contain the test data features
		"""


		mm_scaler = preprocessing.MinMaxScaler().fit(train_data)

		return mm_scaler.transform(train_data), mm_scaler.transform(test_data)



	def _standardize_data(self, train_data, test_data):

		"""
		Standardize the data. Test data standardization will be based on train data

		Parameters
		----------
		train_data: 2d array
			contain the train data features
		test_data: 2d array
			contain the test data features
		"""


		std_scaler = preprocessing.StandardScaler().fit(train_data)

		return std_scaler.transform(train_data), std_scaler.transform(test_data)



	def _check_params(self):

		"""
		Checks if all given configurations are sintactly correct.

		Performs two different transformations over parameter
		dictionaries when needed:

		- If one parameter's values are not inside a list, GridSearchCV
		will not be able to handle them, so they must be enclosed into one.

		- When an ensemble method, as OrderedPartitions, is chosen as
		classifier, transforms the dict of lists in which the
		parameters for the internal classifier are stated into a list
		of dicts (all possible combiantions of those different parameters).
		"""

		random_seed = np.random.get_state()[1][0]
		for _, conf in self.configurations.items():


			parameters = conf['parameters'] # Aliasing

			# Adding given seed as random_state value
			if check_for_random_state(conf['classifier']):
				parameters['random_state'] = [random_seed]


			# An ensemble method is going to be used
			if 'parameters' in parameters and type(parameters['parameters'] == dict):

				# Adding given seed as random_state value
				if check_for_random_state(parameters['base_classifier']):
					parameters['parameters']['random_state'] = [random_seed]


				try:

					# Creating a list for each parameter.
					# Elements represented as 'parameterName;parameterValue'.
					p_list = [[p_name + ';' + str(v) for v in p] for p_name, p in
								parameters['parameters'].items()]
					# Permutations of all lists. Generates all possible
					# combination of elements between lists.
					p_list = [list(item) for item in list(product(*p_list))]
					# Creates a list of dictionaries, containing all
					# combinations of given parameters
					p_list = [dict([item.split(';') for item in p]) for p in p_list]

				except TypeError:
					raise TypeError('All parameters for base_classifier must be list')


				# Returns non-string values back to it's normal self
				for d in p_list:
					for (k, v) in d.items():

						try:
							d[k] = literal_eval(v)
						except ValueError:
							pass

				parameters['parameters'] = p_list


			# No need to cross-validate when there is just one value per parameter
			if all(not isinstance(p, list) or len(p) == 1 for _, p in parameters.items()):
				# Pop lonely values out of list
				for p_name, p in parameters.items():
					if isinstance(p, list):
						parameters[p_name] = p[0]

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

		optimal: GridSearchCV object or classifier object
			An already fitted model of the given classifier,
			with the best found parameters after cross-validation.
			If cross-validation is not needed, it will return the
			classifier model already trained.
		"""


		# No need to cross-validate when there is just one value per parameter
		if all(not isinstance(p, list) for k, p in parameters.items()):

			optimal = classifier(**parameters)

			start = time()
			optimal.fit(train_inputs, train_outputs)
			elapsed = time() - start

			optimal.refit_time_ = elapsed
			return optimal


		try:
			module = __import__("orca_python").metrics
			metric = getattr(module, self.general_conf['cv_metric'].lower().strip())

		except AttributeError:

			if not isinstance(self.general_conf['cv_metric'], str):
				raise AttributeError("cv_metric must be string")

			raise AttributeError("No metric named '%s' implemented"
								% self.general_conf['cv_metric'].strip().lower())


		# Making custom metrics compatible with sklearn
		gib = module.greater_is_better(self.general_conf['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)

		# Creating object to split train data for cross-validation
		# This will make GridSearch have a pseudo-random beheaviour
		skf = StratifiedKFold(n_splits=self.general_conf['hyperparam_cv_nfolds'],
								shuffle=True, random_state=np.random.get_state()[1][0])

		# Performing cross-validation phase
		optimal = GridSearchCV(estimator=classifier(), param_grid=parameters, scoring=scoring_function,
								n_jobs=self.general_conf['jobs'], cv=skf)

		optimal.fit(train_inputs, train_outputs)

		return optimal




	def write_report(self):

		"""
		Saves summarized information about experiment
		through Results class.
		"""

		if self.verbose:
			print("\nSaving Results...")

		# Names of each metric used (plus computational times)
		metrics_names = [x.strip().lower() for x in self.general_conf['metrics']] \
													+ ["cv_time", "time"]

		# Saving results through Results class
		self._results.save_summaries(metrics_names)



##########################
# END OF UTILITIES CLASS #
##########################


def check_packages_version():

	"""
	Checks if minimum version of packages used by this
	framework are installed.
	"""


	print("Checking packages version...")

	print("NumPy...", end=" ")
	if parse_version(get_distribution("numpy").version) < parse_version("1.15.2"):
		print("OUTDATED. Upgrade to 1.15.2 or newer")
	else:
		print("OK")

	print("Pandas...", end=" ")
	if parse_version(get_distribution("pandas").version) < parse_version("0.23.4"):
		print("OUTDATED. Upgrade to 0.23.4 or newer")
	else:
		print("OK")

	print("Sacred...", end=" ")
	if parse_version(get_distribution("sacred").version) < parse_version("0.7.3"):
		print("OUTDATED. Upgrade to 0.7.3 or newer")
	else:
		print("OK")

	print("Scikit-Learn...", end=" ")
	if parse_version(get_distribution("scikit-learn").version) < parse_version("0.20.0"):
		print("OUTDATED. Upgrade to 0.20.0 or newer")
	else:
		print("OK")

	print("SciPy...", end=" ")
	if parse_version(get_distribution("scipy").version) < parse_version("1.1.0"):
		print("OUTDATED. Upgrade to 1.1.0 or newer")
	else:
		print("OK")



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
		Parameters to initialize the classifier with. Used when loading
		a classifiers inside of an ensemble algorithm (base_classifier)


	Returns
	-------

	classifier: object
		Returns a loaded classifier, either from an scikit-learn
		module, or from a module of this framework.
		Depending if hyper-parameters are specified, the object will be
		instantiated or not.

	"""

	# Path to framework local classifier
	if (len(classifier_path.split('.')) == 1):
		classifier = __import__(classifier_path)
		classifier = getattr(classifier, classifier_path)

	# Path to Scikit-Learn classifier
	else:

		classifier = __import__(classifier_path.rsplit('.', 1)[0], fromlist="None")
		classifier = getattr(classifier, classifier_path.rsplit('.', 1)[1])

	# Instancing meta-classifier with given parameters
	if params is not None:
		classifier = classifier(**params)

	return classifier



def check_for_random_state(classifier):

	"""
	Checks if classifiers has an attribute named random_state

	Parameters
	----------
	classifier: object
		Instance of an sklearn compatible classifier

	Returns
	-------
	boolean
	"""

	try:

		load_classifier(classifier)().random_state
		return True

	except AttributeError:
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
	int or string
	"""

	try:
		return int(key)
	except ValueError:
		return key
