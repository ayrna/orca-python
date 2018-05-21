
import os, datetime

import pandas as pd


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
			Stores all info about the run of a dataset with specified algorithm.

			The info will be stored as a pandas DataFrame in a class named DataFrameStorage built in
			for the purpose of keeping additional information.

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










