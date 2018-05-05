
import os, sys, csv, json, time, datetime, re
import pandas as pd
import numpy as np

import DSU

class Utilities:
	"""

	"""


	_api_path = ""
	_general_conf = {}
	_algorithm_parameters = {}
	_algorithm_hyper_parameters = {}
	_datasets = {}


	def __init__(self, api_path, general_conf, algorithm_parameters, algorithm_hyper_parameters):

		"""

		"""

		#TODO: Mejorar forma de obtener el path hasta la carpeta base
		self._api_path = api_path
		self._general_conf = general_conf
		self._algorithm_parameters = algorithm_parameters
		self._algorithm_hyper_parameters = algorithm_hyper_parameters


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

		#TODO: Cuidado con los subdirectorios, Permitirlos o no permitirlos?

		# Check if dataset is fragmented
		p = re.compile("\.[0-9]+$")
		if all( p.search(filename) for filename in os.listdir(file_path) ):


			# Segregate partition files by content
			train_files = []
			test_files = []

			for filename in os.listdir(file_path):

				if filename.startswith("train_"):
					train_files.append(file_path + filename)

				elif filename.startswith("test_"):
					test_files.append(file_path + filename)


			for train_file, test_file in zip(train_files, test_files):

				# Check if both sufixes are similar
				if ( train_file[ train_file.find('.') : ] == test_file[ test_file.find('.') : ] ):

					#Declaring partition DSU
					local_dsu = DSU.DSU(file_path, train_file[ train_file.find('.') + 1 : ])

					# Get inputs and outputs from partition
					local_dsu._train_inputs, local_dsu._train_outputs = self.readFile(train_file,\
																					self._general_conf['clasification'],
								  													self._general_conf['outputs'])

					local_dsu._test_inputs, local_dsu._test_outputs = self.readFile(test_file,\
																					self._general_conf['clasification'],
								  													self._general_conf['outputs'])
					# Append DSU to list
					dsu_list.append(local_dsu)

				else:

					print train_file[ train_file.find('.') : ],  " =/= ", test_file[ test_file.find('.') : ]



		# No partitions found
		else:

			# Declaring partition DSU
			local_dsu = DSU.DSU(file_path, "-1")


			for filename in os.listdir(file_path):


				if filename.startswith("train_"):

					local_dsu._train_inputs, local_dsu._train_outputs = self.readFile(file_path + filename,\
																					self._general_conf['clasification'],
								  													self._general_conf['outputs'])

				elif filename.startswith("test_"):

					local_dsu._test_inputs, local_dsu._test_outputs = self.readFile(file_path + filename,\
																					self._general_conf['clasification'],
								  													self._general_conf['outputs'])

			# Append DSU to list
			dsu_list.append(local_dsu)


		# Save info to dataset
		self._datasets[os.path.basename(os.path.normpath(file_path))] = dsu_list




	def readFile(self, filename, clasification, outputs):
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



	def runExperiment(self):
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


				# Running different algorithms
				for algorithm_name, algorithm in zip(algorithm_names, algorithms):

					print "Running algorithm", algorithm_name


					# TODO: Todos los algoritmos tienen que tener la funcion principal con el mismo nombre (runAlgorithm) y 
					# recibir los mismos parametros: los dos diccionarios con los diferentes parametros fijos en 
					# esta ejecucion (algorithm_parameters), las metricas a utilizar (como una lista) y los 
					# hiperparametros a optimizar (como dict y ya obtener los valores dentro ?? )


					local_dsu._metrics = algorithm.entrenar_rbf_total(train_set, test_set, self._algorithm_hyper_parameters,\
																		self._general_conf["clasification"])


	def writeReport(self):

		"""

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
		"""

		# Check if experiments folder exists
		if not os.path.exists(self._api_path + "my_runs/"):
			os.makedir(self._api_path + "my_runs/")


		# Getting name of folder where we will store info about the Experiment
		folder_name = "exp-" + datetime.date.today().strftime("%y-%m-%d") + "-" + datetime.datetime.now().strftime("%H-%M-%S") + "/"

		# Check if folder already exists
		folder_path = self._api_path + "my_runs/" + folder_name
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)



		for dataset_label, dsu_list in self._datasets.iteritems():

			# Creates subfolders for each dataset
			os.makedirs(folder_path + dataset_label)
			
			for local_dsu in dsu_list:

				pass







