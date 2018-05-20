
import os, datetime

import pandas as pd


#TODO: Ordenar mejor la informacion en estas dos clases:
#		Hacer que DFS alamcene una lista de dataframes (uno por cada algoritmo para un dataset)
#		De manera que Results solo tenga que almacenar una lista de DFS (uno por cada dataset)
#		(Si lo consideramos al reves daria lo mismo)

class DataFrameStorage:

	"""

	"""

	def __init__(self, dataset_name, algorithm_name):

		"""

		"""

		self.dataset_ = dataset_name
		self.algorithm_ = algorithm_name
		self.df_ = None


class Results:

	"""
	"""

	def __init__(self):

		"""

		"""

		self.dataframes_ = []


	def getDataFrame(self, dataset, algorithm):

		"""

		"""

		for dfs in self.dataframes_:

			if dfs.dataset_ == dataset and dfs.algorithm_ == algorithm:
				return dfs

		return False


	def addRecord(self, dataset, algorithm, metrics):

		"""

		"""


		if not self.getDataFrame(dataset, algorithm):

			dfs = DataFrameStorage(dataset, algorithm)
			dfs.df_ = pd.DataFrame(metrics)

			self.dataframes_.append(dfs)



	def saveResults(self, api_path):

		"""

		"""

		# Check if experiments folder exists
		if not os.path.exists(api_path + "my_runs/"):
			os.makedirs(api_path + "my_runs/")

		# Getting name of folder where we will store info about the Experiment
		folder_name = "exp-" + datetime.date.today().strftime("%y-%m-%d") + "-" + datetime.datetime.now().strftime("%H-%M-%S") + "/"

		# Check if folder already exists
		folder_path = api_path + "my_runs/" + folder_name
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)


		for dataframe in self.dataframes_:
		
			# Creates subfolders for each dataset
			dataset_folder = folder_path + dataframe.dataset_ + "/"
	
			if not os.path.exists(dataset_folder):
				os.makedirs(dataset_folder)

			dataframe.df_.to_csv(dataset_folder + dataframe.dataset_ + "-" + dataframe.algorithm_ + ".csv")










