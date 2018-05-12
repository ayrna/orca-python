
import os, sys, csv, json, time, datetime, re
import pandas as pd
import numpy as np

import DSU

class Utilities:
	"""

	"""


	_api_path = ""
	_datasets = {}
	_general_conf = {}


	def __init__(self, api_path, general_conf, algorithms):

		"""

		"""

		#TODO: Mejorar forma de obtener el path hasta la carpeta base
		self._api_path = api_path
		self._general_conf = general_conf


		#TODO: Obtener el numero de salidas para cada dataset sin que se tenga que especificar en el fichero de
		# configuracion, y almacenarlo como una lista o como un diccionario

		print "\n###############################"
		print "\tLoading Start"
		print "###############################"

		# Process each dataset provided by user
		for x in general_conf['datasets'].split(','):

			dataset_name = x.strip()

			#TODO: Check if basedir has a final backslash or not
			file_path = general_conf['basedir'] + dataset_name + '/'

			print "Loading dataset", dataset_name, "info..."
			self.loadDataset(file_path)


	def loadDataset(self, file_path):

		"""

		"""

		dsu_list = []
		train_files = []
		test_files = []

		for filename in os.listdir(file_path):

			if not os.path.isdir(filename):

				if filename.startswith("train_"):
					train_files.append(file_path + filename)

				elif filename.startswith("test_"):
					test_files.append(file_path + filename)



		for train_file, test_file in zip(train_files, test_files):

			#TODO: En vez de comprobar si los ficheros dentro del zip comparten el nombre, se podrian ordenar
			# con anterioridad las listas train_file y test_file alfabeticamente, de esa manera estaria seguro


			# Check if both sufixes are similar
			if ( train_file[ train_file.find('.') : ] == test_file[ test_file.find('.') : ] ):


				#Declaring partition DSU
				local_dsu = DSU.DSU(file_path, train_file[ train_file.find('.') + 1 : ])

				# Get inputs and outputs from partition
				local_dsu._train_inputs, local_dsu._train_outputs = self.readFile(train_file)

				local_dsu._test_inputs, local_dsu._test_outputs = self.readFile(test_file)

				# Append DSU to list
				dsu_list.append(local_dsu)

			else:

				print train_file[ train_file.find('.') : ],  " =/= ", test_file[ test_file.find('.') : ]


		# Save info to dataset
		self._datasets[os.path.basename(os.path.normpath(file_path))] = dsu_list




	def readFile(self, filename):
		"""

		"""

		f = pd.read_csv(filename, header=None)

		inputs = f.values[:,0:(-1)]
		outputs = f.values[:,(-1)]

		return inputs, outputs



	def runExperiment(self):
		"""

		"""


		












		"""
		print "\n###############################"
		print "\tRunning Experiment"
		print "###############################"


		# Dinamically importing algorithm modules needed for actual experiment
		sys.path.insert(0, 'Algorithms/') # Import modules from different directory

		# TODO: Comprobar que los algoritmos dados son correctos (y el resto de parametros), sino parar la ejecucion
		algorithm_names = [x.strip(' ').upper() for x in self._algorithm_parameters['algorithms'].split(',')]
		algorithms = map(__import__, algorithm_names)


		# Running different datasets and partitions
		for dataset_label, dsu_list in self._datasets.iteritems():
			for local_dsu in dsu_list:

				if (local_dsu._partition == "-1"):
					print "\nRunning", dataset_label, "dataset..."
					print "--------------------------"
				else:
					print "\nRunning", dataset_label, "dataset, Partition", local_dsu._partition, "..."
					print "--------------------------"

				train_set = {'inputs': local_dsu._train_inputs, 'outputs': local_dsu._train_outputs}
				test_set = {'inputs': local_dsu._test_inputs, 'outputs': local_dsu._test_outputs}
				metrics_dict = {}

				# Running different algorithms
				for algorithm_name, algorithm in zip(algorithm_names, algorithms):

					print "Running algorithm", algorithm_name


					# TODO: Todos los algoritmos tienen que tener la funcion principal con el mismo nombre (runAlgorithm) y 
					# recibir los mismos parametros: los dos diccionarios con los diferentes parametros fijos en 
					# esta ejecucion (algorithm_parameters), las metricas a utilizar (como una lista) y los 
					# hiperparametros a optimizar (como dict y ya obtener los valores dentro ?? )


					metrics_dict[algorithm_name] = algorithm.entrenar_rbf_total(train_set, test_set,\
															self._algorithm_hyper_parameters,\
															self._general_conf["clasification"])


				local_dsu._metrics = metrics_dict
		"""



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

		for dataset_label, dsu_list in self._datasets.iteritems():

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




