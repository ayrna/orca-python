
import os, sys, csv, json, time, datetime, re, collections
import pandas as pd
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer

from DSU import DSU
from Results import Results

class Utilities:
	"""

	"""


	def __init__(self, api_path, general_conf, configurations):

		"""

		"""

		#TODO: Mejorar forma de obtener el path hasta la carpeta base (para que no dependa de la ubicacion del fichero config)
		#		Por ejemplo, del api path, quitando carpetas por la derecha hasta llegar a el nombre de la carpeta .../orca_python/
		self.api_path_ = api_path
		self.general_conf_ = general_conf
		self.configurations_ = configurations


		#TODO: Obtener el numero de salidas para cada dataset sin que se tenga que especificar en el fichero de configuracion
		# ---> En principio no es necesario, pero se puede conseguir tras leer las salidas de los datos, restando al mayor valor
		#		de las salidas el menor valor y se le suma 1.

		print "\n###############################"
		print "\tLoading Start"
		print "###############################\n"

		self.datasets_ = {}
		# Process each dataset provided by user
		for x in general_conf['datasets'].split(','):

			dataset_name = x.strip()

			#Check if basedir has a final backslash or not
			if general_conf['basedir'][-1] == '/':
				file_path = general_conf['basedir'] + dataset_name + '/'
			else:
				file_path = general_conf['basedir'] + "/" + dataset_name + '/'


			print "Loading dataset", dataset_name, "info..."
			self.datasets_[os.path.basename(os.path.normpath(file_path))] = self._loadDataset(file_path)



	def _loadDataset(self, file_path):

		"""

		"""

		# Looks for all files specified as part of dataset in given folder and orders 'em
		train_files = []
		test_files = []

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



	def runExperiment(self):
		"""

		"""


		print "\n###############################"
		print "\tRunning Experiment"
		print "###############################"


		# Adding algorithm folder to sys path. Needed to import modules from different folders
		sys.path.insert(0, 'Algorithms/')

		self.results_ = Results()	# Creates results object, that will store all different metrics for each configuration and dataset


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


				train_metrics_list = []
				test_metrics_list = []
				best_params_list = []

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

						train_metrics['train_' + metric_name.strip()] = train_score
						test_metrics['test_' + metric_name.strip()] = test_score

					train_metrics_list.append(train_metrics)
					test_metrics_list.append(test_metrics)
					best_params_list.append(optimal_estimator.best_params_)

				self.results_.addRecord(dataset_name, conf_name, train_metrics_list, test_metrics_list,\
										best_params_list, self.general_conf_['metrics'].split(','))



	def _getOptimalEstimator(self, train_inputs, train_outputs, algorithm, parameters):

		"""

		"""
		module = __import__("Metrics")
		metric = getattr(module, self.general_conf_['cv_metric'].lower().strip())

		#TODO: ADD ALL METRICS THAT ARE GIB TO THE LIST
		gib = module.greater_is_better(self.general_conf_['cv_metric'].lower().strip())
		scoring_function = make_scorer(metric, greater_is_better=gib)


		#TODO: Other way of doing this w/out using a for loop ?
		for param_name, param in parameters.iteritems():

			if type(param) != list:
				parameters[param_name] = [param]


		# TODO: What if jobs or folds are not given?
		optimal = GridSearchCV(estimator=algorithm(), param_grid=parameters, scoring=scoring_function,\
								n_jobs=self.general_conf_['jobs'], cv=self.general_conf_['folds'])
		optimal.fit(train_inputs, train_outputs)

		return optimal




	def writeReport(self):

		"""


		"""
		summary_index = []
		for dataset_name in self.datasets_.keys():
			for conf_name in self.configurations_.keys():

				summary_index.append(dataset_name + "-" + conf_name)

		self.results_.saveResults(self.api_path_, summary_index)


