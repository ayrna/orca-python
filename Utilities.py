
import os, sys, csv, json, time, datetime, re
import pandas as pd
import numpy as np

from sklearn.grid_search import GridSearchCV

import DSU
from Results import Results

class Utilities:
	"""

	"""


	def __init__(self, api_path, general_conf, algorithms):

		"""

		"""

		#TODO: Mejorar forma de obtener el path hasta la carpeta base
		self.api_path_ = api_path
		self.general_conf_ = general_conf
		self.algorithms_ = algorithms


		#TODO: Obtener el numero de salidas para cada dataset sin que se tenga que especificar en el fichero de
		# configuracion, y almacenarlo como una lista o como un diccionario

		print "\n###############################"
		print "\tLoading Start"
		print "###############################"

		self.datasets_ = {}
		# Process each dataset provided by user
		for x in general_conf['datasets'].split(','):

			dataset_name = x.strip()

			#TODO: Check if basedir has a final backslash or not
			file_path = general_conf['basedir'] + dataset_name + '/'


			# TODO: Hacer que la funcion devuelva un solo dataset cargado y aqui hacer el append (Para mejor encapsulacion)
			print "Loading dataset", dataset_name, "info..."
			self._loadDataset(file_path)


	def _loadDataset(self, file_path):

		"""

		"""

		train_files = []
		test_files = []

		for filename in os.listdir(file_path):

			if not os.path.isdir(filename):

				if filename.startswith("train_"):
					train_files.append(file_path + filename)

				elif filename.startswith("test_"):
					test_files.append(file_path + filename)



		partition_list = []

		for train_file, test_file in zip(train_files, test_files):

			#TODO: En vez de comprobar si los ficheros dentro del zip comparten el nombre, se podrian ordenar
			# con anterioridad las listas train_file y test_file alfabeticamente, de esa manera estaria seguro


			# Check if both sufixes are similar
			if ( train_file[ train_file.find('.') : ] != test_file[ test_file.find('.') : ] ):
				print train_file[ train_file.find('.') : ],  " =/= ", test_file[ test_file.find('.') : ]


			else:

				#Declaring partition DSU
				partition = DSU.DSU(file_path, train_file[ train_file.find('.') + 1 : ])

				# Get inputs and outputs from partition
				partition.train_inputs, partition.train_outputs = self._readFile(train_file)

				partition.test_inputs, partition.test_outputs = self._readFile(test_file)

				# Append DSU to list
				partition_list.insert(0, partition)


		# Save info to dataset
		self.datasets_[os.path.basename(os.path.normpath(file_path))] = partition_list




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

		self.results_ = Results()	# Creates results object, that will store al different metrics for each algorithm and dataset


		# Iterating over all different datasets
		for dataset_name, dataset in self.datasets_.iteritems():

			print "\nRunning", dataset_name, "dataset..."
			print "--------------------------"

			# Iterating over all different algorithm configurations
			for conf_name, configuration in self.algorithms_.iteritems():

				print "Running", configuration["algorithm"], "algorithm"

				# TODO: Comprobar que los algoritmos dados son correctos (y el resto de parametros), sino parar la ejecucion
				module = __import__(configuration["algorithm"])
				algorithm = getattr(module, configuration["algorithm"])

				optimal_params = self._getOptimalParameters(dataset, algorithm, configuration["parameters"])

				# TODO: Si no estan ordenados correctamente en el fichero de configuracion no funcionara como es debido
				algorithm_model = algorithm(*optimal_params.values())
				metrics_list = []

				# Iterating over all partitions in each dataset
				for partition in dataset:

					print "  Running Partition", partition.partition
					algorithm_model.fit(partition.train_inputs, partition.train_outputs)


					# Creating tuples with each specified tuple and passing it to specified dataframe

					metrics = {}
					for metric_name in self.general_conf_['metrics'].split(','):

						module = __import__("Metrics")
						metric = getattr(module, metric_name.strip())

						predicted_y = algorithm_model.predict(partition.test_inputs)
						partition_score = metric(partition.test_outputs, predicted_y)

						metrics[metric_name] = partition_score

					metrics_list.append(metrics)

				self.results_.addRecord(dataset_name, configuration['algorithm'], metrics_list)


		for dfs in self.results_.dataframes_:

			print dfs.dataset_, "-", dfs.algorithm_
			print dfs.df_



	def _getOptimalParameters(self, dataset, algorithm, parameters):

		"""

		"""

		# TODO: Dividir el conjunto de entrenamiento en entrenamiento y validacion (o ya lo hace gridsearch solo ?? )
		best_params = []
		for partition in dataset:

			algorithm_model = algorithm()

			# TODO: Seleccionar medida de precision en base a fichero de configuracion (make_scorer para crearla uno mismo)
 
			optimal = GridSearchCV(estimator=algorithm_model, param_grid=parameters, scoring='accuracy', n_jobs=1, cv=5)
			optimal.fit(partition.train_inputs, partition.train_outputs)

			params = optimal.best_params_
			score =  optimal.score(partition.train_inputs, partition.train_outputs)
			partition_params = {'params': params, 'score': score}
			best_params.append(partition_params)

		optimal_params = self._selectOptimalParameters(best_params)

		return optimal_params



	def _selectOptimalParameters(self, best_params):

		"""
		For now, selecting only the combination of parameters that gave best score value for crossvalidation_metric
		between all different partitions
		"""

		optimal_params = best_params.pop()
		for partition_params in best_params:

			if partition_params['score'] > optimal_params['score']:
				optimal_params = partition_params


		return optimal_params['params']



	def writeReport(self):

		"""


		"""



		# Check if experiments folder exists
		if not os.path.exists(self._api_path + "my_runs/"):
			os.makedirs(self._api_path + "my_runs/")


		# Getting name of folder where we will store info about the Experiment
		folder_name = "exp-" + datetime.date.today().strftime("%y-%m-%d") + "-" + datetime.datetime.now().strftime("%H-%M-%S") + "/"

		# Check if folder already exists
		folder_path = self._api_path + "my_runs/" + folder_name
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)


		#TODO: Una vez implementadas las diferentes metricas y que se puedan decidir cuales usar desde el fichero
		# de configuracion, hacer que esta lista sean unicamente las metricas a utilizar. Ademas habra que contemplar
		# los distintos nombres de las variables utilizadas (esto ultimo con un for y un append)

		for dataset_label, dsu_list in self.datasets_.iteritems():

			# Creates subfolders for each dataset
			os.makedirs(folder_path + dataset_label)

			# One file per partition
			for local_dsu in dsu_list:

				for algorithm_name, metrics in local_dsu._metrics.iteritems():

					# Get name of csv file
					if (local_dsu._partition == "csv"):
						file_path_train = folder_path + dataset_label + "/train-" + dataset_label + "-" + \
											algorithm_name + "." + local_dsu._partition
						file_path_test = folder_path + dataset_label + "/test-" + dataset_label + "-" + \
											algorithm_name + "." + local_dsu._partition

					else:
						file_path_train = folder_path + dataset_label + "/train-" + dataset_label + "-" + \
											algorithm_name + "." + local_dsu._partition + ".csv"
						file_path_test = folder_path + dataset_label + "/test-" + dataset_label + "-" + \
											algorithm_name + "." + local_dsu._partition + ".csv"



					# Writing results metrics to CSV
					with open(file_path_train, 'w') as train_csv, open(file_path_test, 'w') as test_csv:


						metrics_header = list(metrics[0]._hyper_parameters)
						metrics_header.append("MSE")
						metrics_header.append("CCR")


						train_writer = csv.DictWriter(train_csv, fieldnames=metrics_header)
						test_writer = csv.DictWriter(test_csv, fieldnames=metrics_header)

						train_writer.writeheader()
						test_writer.writeheader()

						# Writing one row per different configuration of hyper parameters
						for param_metrics in metrics:

							train_writer.writerow( dict(param_metrics._hyper_parameters, **param_metrics._train_metrics) )
							test_writer.writerow( dict(param_metrics._hyper_parameters, **param_metrics._test_metrics) )




