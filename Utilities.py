
import os, sys, collections
from itertools import product

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

from Results import Results

class Utilities:
	"""

	Utilities

	Class in charge of running an experiment over N datasets, where we
	apply M different configurations over each one of them.

	Configurations are composed of an algorithm and different parameters,
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
		Class used to manage and store all info obtained when testing
		the different built models during the run of an experiment.
	"""


	def __init__(self, general_conf, configurations):


		self.general_conf_ = general_conf
		self.configurations_ = configurations
		

	def runExperiment(self):

		"""
		Main function of this class. Runs an experiment.

		Builds one model for each possible combination of dataset and 
		config entry stated in it's corresponding configuration file.

		Loads every dataset, which can be fragmented in different partitions.
		Builds a model for every dataset, using cross-validation for finding
		the best possible values for the different parameters of that actual
		configuration entry.
		Get train and test metrics for each dataset-config pair.


		"""


		self.results_ = Results()
		# Adding algorithm folder to sys path. Needed to import modules from different folders
		sys.path.insert(0, 'Algorithms/')

		print "\n###############################"
		print "\tRunning Experiment"
		print "###############################"



		#TODO: Comprobar que dos configuraciones no tengan el mismo nombre, porque sino se sobreescribiran (Try catch exception)

		# Iterating over Datasets
		for x in self.general_conf_['datasets'].split(','):


			# Getting dataset name and path, stripped out of whitespaces
			dataset_name = x.strip()
			dataset_path = self._getDatasetPath(self.general_conf_['basedir'], dataset_name)

			# Loading dataset into a list of partitions. Each partition represented as a dictionary
			# containing train and test inputs/outputs. It also stores its partition number

			dataset = self._loadDataset(dataset_path)


			print "\nRunning", dataset_name, "dataset"
			print "--------------------------"

			# Iterating over all different Configurations
			for conf_name, configuration in self.configurations_.iteritems():
				print "Running", conf_name, "..."

				# TODO: Comprobar que los algoritmos dados son correctos (y el resto de parametros), sino parar la ejecucion
				#		Hacer que todas las metricas y algoritmos sean upper

				# Loading Algorithm stated in configuration
				algorithm = self._loadAlgorithm(configuration["algorithm"])
				# Iterating over all partitions in each dataset
				for partition in dataset:

					# Print number of actual partition if dataset is partitionated
					if partition["partition"] != "csv":
						print "  Running Partition", partition["partition"]

					# Finding optimal parameters
					optimal_estimator = self._getOptimalEstimator(partition["train_inputs"], partition["train_outputs"],\
																  algorithm, configuration["parameters"])


					# Creating tuples with each specified tuple and passing it to specified dataframe
					train_metrics = collections.OrderedDict()
					test_metrics = collections.OrderedDict()

					# Iterating over Metrics
					for metric_name in self.general_conf_['metrics'].split(','):

						# Loading metric from metrics file
						module = __import__("Metrics")
						metric = getattr(module, metric_name.strip().lower())

						# Get train scores
						train_predicted_y = optimal_estimator.predict(partition["train_inputs"])
						train_score = metric(partition["train_outputs"], train_predicted_y)

						# Get test scores
						test_predicted_y = optimal_estimator.predict(partition["test_inputs"])
						test_score = metric(partition["test_outputs"], test_predicted_y)

						# Add metric to dict of metrics
						train_metrics[metric_name.strip() + '_train'] = train_score
						test_metrics[metric_name.strip() + '_test'] = test_score

					# Save metrics scores for this partition
					self.results_.addRecord(dataset_name, conf_name, train_metrics, test_metrics,\
											optimal_estimator.best_params_)



	def _getDatasetPath(self, base_path, dataset_name):

		"""
		Gets path to actual dataset.

		Parameters
		----------

		base_path: string
			Base path in which dataset folder can be found

		dataset_name: string
			Name given to dataset folder


		Returns
		-------

		dataset_path: string
			Absolute path to folder containing dataset files
		"""

		# Check if path it's relative or absolute
		if base_path.startswith("~"):

			base_path = base_path.replace('~', os.path.expanduser('~'), 1)
			self.general_conf_['basedir'] = base_path

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
			Absolute path to dataset folder


		Returns
		-------

		partition_list: list of dicts
			List of partitions found inside a dataset folder.
			Each partition is stored into a dictionary, disjoining
			train and test inputs, and outputs
		"""

		#TODO: Comprobar que existe el fichero de test y de train, sino obrar en consecuencia

		# Looks for all files specified as part of dataset in given folder and orders them

		train_files = []; test_files = []
		for filename in os.listdir(dataset_path):

			if not os.path.isdir(filename):

				if filename.startswith("train_"):
					train_files.append(dataset_path + filename)

				elif filename.startswith("test_"):
					test_files.append(dataset_path + filename)

		train_files.sort(), test_files.sort()


		# Get input and output variables from dataset files
		partition_list = []
		for train_file, test_file in zip(train_files, test_files):


			#Declaring partition
			partition = {"partition": train_file[ train_file.find('.') + 1 : ], "path": dataset_path}

			# Get inputs and outputs from partition
			partition["train_inputs"], partition["train_outputs"] = self._readFile(train_file)
			partition["test_inputs"], partition["test_outputs"] = self._readFile(test_file)

			# Append to begining of list
			partition_list.append(partition)

		# Save info to dataset
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

		f = pd.read_csv(filename, header=None)

		inputs = f.values[:,0:(-1)]
		outputs = f.values[:,(-1)]

		return inputs, outputs



	def _loadAlgorithm(self, algorithm_path):

		"""
		Loads and returns a classifier.

		Parameters
		----------

		algorithm_path: string
			Relative path to which package the classifier class is located in.
			That module can be local if the classifier is built inside the
			framework, or relative to scikit-learn package.


		Returns
		-------

		algorithm: object
			Returns a loaded classifier, either from an scikit-learn module, or from
			a module of this framework.

		"""

		# Loading modules to execute algorithm given in configuration file
		modules = [x for x in algorithm_path.split('.')]

		if (len(modules) == 1):
			algorithm = __import__(modules[0])
			algorithm = getattr(algorithm, modules[0])

		elif (len(modules) == 3):
			algorithm = __import__(modules[0] + '.' + modules[1], fromlist=[str(modules[2])])
			algorithm = getattr(algorithm, modules[2])

		else:
			pass

		return algorithm



	def _getOptimalEstimator(self, train_inputs, train_outputs, algorithm, parameters):

		"""
		Perform cross-validation technique for finding best
		hyper-parameters out of the set given by configuration file.

		Parameters
		----------

		train_inputs: {array-like, sparse-matrix}, shape (n_samples, n_features)
			vector of features for each sample for this dataset.

		train_outputs: array-like, shape (n_samples)
			Target vector relative to train_inputs.

		Returns
		-------

		optimal: GridSearchCV object
			An already fitted model of the given classifier,
			with the best found parameters after performing cross-validation
			over train samples from given partition (or dataset)
		"""

		
		module = __import__("Metrics")
		metric = getattr(module, self.general_conf_['cv_metric'].lower().strip())

		gib = module.greater_is_better(self.general_conf_['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)
		# Checking if this configuration uses OrdinalDecomposition algorithm
		parameters = self._extractParams(parameters)


		#TODO: 	Parametro iid devuelve un Warning si no se especifica el valor (por defecto en warn)
		#		Que valor indicar, True o False??

		optimal = GridSearchCV(estimator=algorithm(), param_grid=parameters, scoring=scoring_function,\
					n_jobs=self.general_conf_['jobs'], cv=self.general_conf_['folds'], iid=True)

		optimal.fit(train_inputs, train_outputs)

		return optimal



	def _extractParams(self, parameters):

		"""
		Performs two different transformations over parameters dict
		when needed. Those consist of:

		- If one parameter's values are not inside a list, GridSearchCV will not be
		  able to handle them, so they must be enclosed into a list.

		- When an ensemble method, as OrderedPartitions, is chosen as classifier,
		  transforms the dict of lists in which the parameters for the internal
		  classifier are stated into a list of dicts (all possible combiantions of
		  those different parameters).

		Parameters
		----------

		parameters: dict of list
			Dictionary which contains parameters to cross-validate.

		Returns
		-------

		parameters: dict of list
			Dictionary properly formatted if necessary.
		"""

		for param_name, param in parameters.iteritems():

			# If parameter is not a list, convert it into one
			if (type(param) != list) and (type(param) != dict):
				parameters[param_name] = [param]

			# If parameter is a dict named 'parameters', then an ensemble method it's been used
			# we need to transform a dict of lists, into a list of dicts.
			elif (type(param) == dict) and (param_name == 'parameters'):

				# Creating a list for each parameter. Elements represented as 'parameterName_parameterValue'.
				p_list = [ [p_name + '_' + str(v) for v in p] for p_name, p in param.iteritems() ]
				# Permutations of all lists. Generates all possible combination of elements between lists.
				p_list = [ list(item) for item in list(product(*p_list)) ]
				# Creates a list of dictionaries, containing all combinations of given parameters
				p_list = [ dict( [item.split('_') for item in p] ) for p in p_list ]

				# Returns not string values back to it's normal self
				for d in p_list:
					for (k, v) in d.iteritems():

						if self._isFloat(v):
							d[k] = float(v)
						elif self._isBoolean(v):
							d[k] = bool(v)

				parameters[param_name] = p_list

		return parameters



	def _isFloat(self, value):

		"""
		Check if an string can be converted to float

		Parameters
		----------

		value: string

		Returns
		-------

		Boolean
		"""

		try:
			float(value)
			return True
		except ValueError:
			return False



	def _isBoolean(self, value):

		"""
		Check if an string can be converted to Boolean

		Parameters
		----------

		value: string

		Returns
		-------

		Boolean
		"""


		try:
			bool(value)
			return True
		except ValueError:
			return False



	def writeReport(self, fw_path):

		"""
		Saves information about experiment run into CSVs

		Parameters
		----------

		fw_path: string
			Path to framework folder
		"""

		# Info needed to save execution info properly
		# All possible pairs of dataset-configuration names
		summary_index = []
		for dataset_name in self.general_conf_['datasets'].split(','):
			for conf_name in self.configurations_.keys():

				# Names for every file (excepting summary ones)
				summary_index.append(dataset_name.strip() + "-" + conf_name)

		# Names of each metric used
		metrics_names = [x.strip().lower() for x in self.general_conf_['metrics'].split(',')]

		self.results_.saveResults(fw_path, summary_index, metrics_names)


