
import os, sys, csv, json, time, datetime, re
import pandas as pd
import numpy as np

import DSU

class Utilities:
	"""

	"""

	#TODO: EN VEZ DE TANTAS ESTRUCTURAS DE DATOS ANIDADAS, CREAR UNA CLASE QUE SE LLAME DSU (DATA STORAGE UNIT)
	# O COMO SEA, QUE CONTENGA LOS CAMPOS QUE SEAN NECESARIOS PARA MANEJAR TODO APROPIADAMENTE
	#
	#
	#	CAMPO_1: DATASET
	#	CAMPO_2: NUMERO PARTICION
	#	CAMPO_3: TRAIN_INPUTS
	#	CAMPO_4: TRAIN_OUTPUTS
	#	CAMPO_5: TEST_INPUTS
	#	CAMPO_6: TEST_OUTPUTS
	#
	# COMO PODEMOS OBSERVAR, OPERA A NIVEL DE PARTICION DE DATASET
	# DE ESTA FORMA, SOLO NECESITAREMOS ALMACENAR UNA LISTA DE DSUs

	_api_path = ""
	_train_datasets = _test_datasets = {}
	_train_results = _test_results = {}


	def __init__(self, api_path, general_conf, algorithm_parameters, algorithm_hyper_parameters):

		"""

		"""

		self._api_path = api_path

		train_partitions = []
		test_partitions = []


		#TODO GENERAL GORDO: UNA VEZ SE TENGA EL DICCIONARIO, SE PODRA ITERAR SOBRE EL Y POSTERIORMENTE SOBRE
		#LAS LISTAS, LLAMANDO EN CADA OCASION AL ALGORITMO, QUE DEVOLVERA EL RESULTADO PARA DICHA PARTICION
		#COMO UNA LISTA CON LOS VALORES PARA LAS DISTINTAS METRICAS
		
		#AL FINAL, TENDREMOS QUE TENER UN DICCIONARIO (DATASETS) DE LISTAS (PARTICIONES) DE LISTAS (METRICAS)


		"""
		Orden Jerarquico:

			Diferentes Datasets (uno para train otro para test) - diccionario - clave: nombres datasets

				Particiones del Dataset - lista de elementos del siguiente nivel

					Entradas y/o Salidas - diccionario - clave: input y output

						Entradas y salidas - lista - valores concretos obtenidos al leer el fichero

		"""

		#TODO: Obtener el numero de salidas para cada dataset sin que se tenga que especificar en el fichero de
		# configuracion, y almacenarlo como una lista o como un diccionario

		# Process each dataset provided by user
		for x in general_conf['datasets'].split(','):

			dataset_name = x.strip()

			#TODO: Check if basedir has a final backslash or not
			file_path = general_conf['basedir'] + dataset_name + '/'


			#TODO: Cuidado con los subdirectorios

			# Check if dataset is fragmented
			p = re.compile("\.[0-9]+$")
			if all( p.search(filename) for filename in os.listdir(file_path) ):


				# Proccess each partition
				for filename in os.listdir(file_path):

					if filename.startswith("train_"):

						inputs, outputs = self.loadFile(file_path + filename, algorithm_parameters['clasification'],\
														algorithm_parameters['outputs'])
						train_partitions.append({'inputs': inputs, 'outputs': outputs})

					elif filename.startswith("test_"):

						inputs, outputs = self.loadFile(file_path + filename, algorithm_parameters['clasification'],\
														algorithm_parameters['outputs'])
						test_partitions.append({'inputs': inputs, 'outputs': outputs})



			# No partitions found
			else:

				for filename in os.listdir(file_path):

					if filename.startswith("train_"):

						inputs, outputs = self.loadFile(file_path + filename, algorithm_parameters['clasification'],\
														algorithm_parameters['outputs'])
						train_partitions.append({'inputs': inputs, 'outputs': outputs})

					elif filename.startswith("test_"):

						inputs, outputs = self.loadFile(file_path + filename, algorithm_parameters['clasification'],\
														algorithm_parameters['outputs'])
						test_partitions.append({'inputs': inputs, 'outputs': outputs})



			# Save all loaded data into anidated data structure
			self._train_datasets[dataset_name] = train_partitions
			self._test_datasets[dataset_name] = test_partitions




	def loadFile(self, filename, clasification, outputs):
		"""

		"""

		f = pd.read_csv(filename, header=None)

		if clasification:

			inputs = f.values[:,0:(-1)]
			outputs = f.values[:,(-1)]

		else:

			inputs = f.values[:,0:(-outputs)]
			outputs = f.values[:,(-outputs):]


		return inputs, outputs



	def runExperiment(self, algorithm_parameters, algorithm_hyper_parameters):
		"""

		"""

		#TODO: Para importar distintos algoritmos, habra que poner la parte del import correspondiente
		# dentro del for que itere sobre los algoritmos
		sys.path.insert(0, 'Algorithms/') # Import modules from different directory
		import rbf


		
		train_dataset_results = test_dataset_results = []

		#TODO: REMEMBER - Como ambos diccionarios (train y test) comparten claves, se podria iterar solo sobre uno
		# 				 y obtener el contenido del otro mediante las claves

		# Iterate over train and test dicts at same time
		for (train_label, train_dataset), (test_label, test_dataset)\
			in zip(self._train_datasets.items(), self._test_datasets.items()):

			# Iterate over train and test partitions
			for train_partition, test_partition in zip(train_dataset, test_dataset):

				# Running especified alogorithm
				train_metrics, test_metrics = rbf.entrenar_rbf_total(train_partition, test_partition, \
																	algorithm_parameters["clasification"],\
																	algorithm_parameters["ratio_rbf"], \
																	algorithm_parameters["l2"],\
																	algorithm_parameters["eta"], \
																	algorithm_parameters["outputs"])

				train_dataset_results.append(train_metrics)
				test_dataset_results.append(test_metrics)


		


	def writeReport(self, folder_name, train_metrics, test_metrics):

		# Getting preliminar info to start experiment

		folder_name = "exp-" + datetime.date.today().strftime("%y-%m-%d") + "-" + datetime.datetime.now().strftime("%H-%M-%S") + "/"

		
		"""
		writeReport function will concentrare and write all results in a subdirectory inside my_runs subfolder

		Information writed will be similar to that of orca API

			train_metrics and test_metrics will be:

				- A dict of the different metrics used
				- A list of dicts of the different metrics used (when there are more than one partition)
		"""

		# Check if results directory exists, if not, create it
		folder_path = os.path.dirname(os.path.abspath(__file__)) + "/my_runs/" + folder_name
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)


		file_path = folder_path + "results.csv"

		# Writing results metrics to CSV
		with open(file_path, 'w') as csvfile:

			# Obtain name of the different metrics used in this experiment
			metrics = []
			for keys in train_metrics:
				metrics.append(keys)
			writer = csv.DictWriter(csvfile, fieldnames=metrics)

			writer.writeheader()
			writer.writerow(train_metrics)
			writer.writerow(test_metrics)









