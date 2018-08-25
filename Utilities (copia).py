
import os, sys, csv, json, time, datetime, re, collections
from itertools import product

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

#from DSU import DSU
from Results import Results



class DSU:
	"""
	DSU - Data Storage Unit

	Abstraction level which we will work with.
	"""

	def __init__(self, dataset, partition):

		self.dataset = dataset
		self.partition = partition

		self.train_inputs = []
		self.train_outputs = []
		self.test_inputs = []
		self.test_outputs = []



class Utilities:
	"""

	"""


	def __init__(self, api_path, general_conf, configurations):

		"""

		"""

		self.api_path_ = api_path
		self.general_conf_ = general_conf
		self.configurations_ = configurations



	def runExperiment(self):
		"""

		"""

		#TODO: Comprobar que dos configuraciones no tengan el mismo nombre, porque sino se sobreescribiran

		# Loading and Storing all datasets
		self._processDatasets()

		print "\n###############################"
		print "\tRunning Experiment"
		print "###############################"


		# Adding algorithm folder to sys path. Needed to import modules from different folders
		sys.path.insert(0, 'Algorithms/')

		# Creates results object, that will store all different metrics for each configuration and dataset
		self.results_ = Results()


		# Iterating over all different datasets
		for dataset_name, dataset in self.datasets_.iteritems():

			print "\nRunning", dataset_name, "dataset"
			print "--------------------------"

			# Iterating over all different configurations
			for conf_name, configuration in self.configurations_.iteritems():

				print "Running", conf_name, "..."

				# TODO: Comprobar que los algoritmos dados son correctos (y el resto de parametros), sino parar la ejecucion
				#		Hacer que todas las metricas y algoritmos sean upper
				module = __import__(configuration["algorithm"])
				algorithm = getattr(module, configuration["algorithm"])


				# Iterating over all partitions in each dataset
				for partition in dataset:

					if partition.partition != "csv":
						print "  Running Partition", partition.partition

					# Finding optimal parameters
					optimal_estimator = self._getOptimalEstimator(partition.train_inputs, partition.train_outputs,\
																algorithm, configuration["parameters"])


					# Creating tuples with each specified tuple and passing it to specified dataframe
					train_metrics = collections.OrderedDict()
					test_metrics = collections.OrderedDict()

					for metric_name in self.general_conf_['metrics'].split(','):

						module = __import__("Metrics")
						metric = getattr(module, metric_name.strip().lower())

						train_predicted_y = optimal_estimator.predict(partition.train_inputs)
						train_score = metric(partition.train_outputs, train_predicted_y)

						test_predicted_y = optimal_estimator.predict(partition.test_inputs)
						test_score = metric(partition.test_outputs, test_predicted_y)

						train_metrics[metric_name.strip() + '_train'] = train_score
						test_metrics[metric_name.strip() + '_test'] = test_score


					self.results_.addRecord(dataset_name, conf_name, train_metrics, test_metrics,\
											optimal_estimator.best_params_)

	def _processDatasets(self):

		print "\n###############################"
		print "\tLoading Start"
		print "###############################\n"


		# Process each dataset provided by user

		self.datasets_ = {}
		for x in self.general_conf_['datasets'].split(','):

			dataset_name = x.strip()

			#Check if basedir has a final backslash or not
			if self.general_conf_['basedir'][-1] == '/':
				file_path = self.general_conf_['basedir'] + dataset_name + '/'
			else:
				file_path = self.general_conf_['basedir'] + "/" + dataset_name + '/'


			print "Loading dataset", dataset_name, "info..."
			self.datasets_[os.path.basename(os.path.normpath(file_path))] = self._loadDataset(file_path)



	def _loadDataset(self, file_path):

		"""

		"""

		# Looks for all files specified as part of dataset in given folder and orders them

		train_files = []; test_files = []
		for filename in os.listdir(file_path):

			if not os.path.isdir(filename):

				if filename.startswith("train_"):
					train_files.append(file_path + filename)

				elif filename.startswith("test_"):
					test_files.append(file_path + filename)

		train_files.sort(), test_files.sort()


		# Get input and output variables from dataset files
		partition_list = []
		for train_file, test_file in zip(train_files, test_files):


			#Declaring partition DSU
			partition = DSU(file_path, train_file[ train_file.find('.') + 1 : ])

			# Get inputs and outputs from partition
			partition.train_inputs, partition.train_outputs = self._readFile(train_file)
			partition.test_inputs, partition.test_outputs = self._readFile(test_file)

			# Append DSU to begining of list
			partition_list.append(partition)

		# Save info to dataset
		return partition_list



	def _readFile(self, filename):
		"""

		"""

		f = pd.read_csv(filename, header=None)

		inputs = f.values[:,0:(-1)]
		outputs = f.values[:,(-1)]

		return inputs, outputs



	def _getOptimalEstimator(self, train_inputs, train_outputs, algorithm, parameters):

		"""

		"""
		module = __import__("Metrics")
		metric = getattr(module, self.general_conf_['cv_metric'].lower().strip())

		gib = module.greater_is_better(self.general_conf_['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)
		# Checking if this configuration uses OrdinalDecomposition algorithm
		parameters = self._extractParams(parameters)


		# TODO: What if jobs or folds are not given?
		optimal = GridSearchCV(estimator=algorithm(), param_grid=parameters, scoring=scoring_function,\
								n_jobs=self.general_conf_['jobs'], cv=self.general_conf_['folds'])
		optimal.fit(train_inputs, train_outputs)

		return optimal


	def _extractParams(self, parameters):

		for param_name, param in parameters.iteritems():

			if (type(param) != list) and (type(param) != dict):
				parameters[param_name] = [param]

			elif (type(param) == dict) and (param_name == 'parameters'):

				# Creating a list for each parameter. Elements represented as 'parameterName_parameterValue'.
				p_list = [ [p_name + '_' + str(v) for v in p] for p_name, p in param.iteritems() ]
				# Permutations of all lists. Generates all possible combination of elements between lists.
				p_list = [ list(item) for item in list(product(*p_list)) ]
				# Creates a list of dictionaries, containing all combinations of given parameters
				p_list = [ dict( [item.split('_') for item in p] ) for p in p_list ]

				# Returns stringfied numbers to floats
				for d in p_list:
					for (k, v) in d.iteritems():

						if self._isFloat(v):
							d[k] = float(v)
						elif self._isBoolean(v):
							d[k] = bool(v)

				parameters[param_name] = p_list


		return parameters

	def _isFloat(self, value):

		try:
			float(value)
			return True
		except ValueError:
			return False

	# If Boolean is not converted back from String, it may lead to an error
	def _isBoolean(self, value):

		try:
			bool(value)
			return True
		except ValueError:
			return False


	def writeReport(self):

		"""


		"""

		# Info needed to save execution info properly
		summary_index = []
		for dataset_name in self.datasets_.keys():
			for conf_name in self.configurations_.keys():

				# Names for every file (excepting summary ones)
				summary_index.append(dataset_name + "-" + conf_name)

		# Names of each metric used
		metrics_names = [x.strip().lower() for x in self.general_conf_['metrics'].split(',')]

		self.results_.saveResults(self.api_path_, summary_index, metrics_names)


