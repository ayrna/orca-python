
import os, datetime

import pandas as pd


class DataFrameStorage:

	"""

	"""

	def __init__(self, dataset_name, configuration_name):

		"""

		"""

		self.dataset_ = dataset_name
		self.configuration_ = configuration_name
		self.df_ = None


class Results:

	"""
	"""

	def __init__(self):

		"""

		"""

		self.dataframes_ = []
		self.train_summary_ = []
		self.test_summary_ = []


	def getDataFrame(self, dataset, configuration):

		"""

		"""

		for dfs in self.dataframes_:

			if dfs.dataset_ == dataset and dfs.configuration_ == configuration:
				return dfs

		return False



	def addRecord(self, dataset, configuration, train_metrics, test_metrics, best_parameters, metrics_names):

		"""
			Stores all info about the run of a dataset with specified configuration.

			The info will be stored as a pandas DataFrame in a class named DataFrameStorage built in
			for the purpose of keeping additional information.

		"""

		# Summarizing information from all partitions of configuration into one line

		index_mean = ['mean_' + mn.strip() for mn in metrics_names]
		index_std = ['std_' + mn.strip() for mn in metrics_names]

		train_df = pd.DataFrame(train_metrics)
		test_df = pd.DataFrame(test_metrics)

		train_avg, train_std = train_df.mean(), train_df.std();
		test_avg, test_std = test_df.mean(), test_df.std();

		train_avg.index, train_std.index = index_mean, index_std; train_series = train_avg.append(train_std)
		test_avg.index, test_std.index = index_mean, index_std; test_series = test_avg.append(test_std)

		self.train_summary_.append(train_series)
		self.test_summary_.append(test_series)


		# Mixing train and test metrics in one only dataframe - Will show info for each partition for configuration and DB
		list_of_series = []
		for train_row, test_row, param_row in zip(train_metrics, test_metrics, best_parameters):

			train_row.update(test_row)
			param_row.update(train_row)
			full_row = pd.Series(param_row)
			list_of_series.append(full_row)

		dfs = DataFrameStorage(dataset, configuration)
		dfs.df_ = pd.concat(list_of_series, axis=1).transpose()

		self.dataframes_.append(dfs)


	def saveResults(self, api_path, summary_index):

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


		# Saving summaries from every combination of DB and Configuration
		train_summary = pd.concat(self.train_summary_, axis=1).transpose(); train_summary.index = summary_index
		test_summary = pd.concat(self.test_summary_, axis=1).transpose(); test_summary.index = summary_index

		train_summary.to_csv(folder_path + "/" + "train_summary.csv")
		test_summary.to_csv(folder_path + "/" + "test_summary.csv")


		for dataframe in self.dataframes_:
		
			# Creates subfolders for each dataset
			dataset_folder = folder_path + dataframe.dataset_ + "/"
	
			if not os.path.exists(dataset_folder):
				os.makedirs(dataset_folder)

			dataframe.df_.to_csv(dataset_folder + dataframe.dataset_ + "-" + dataframe.configuration_ + ".csv")







