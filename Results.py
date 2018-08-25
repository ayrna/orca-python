
import os, datetime, collections

import pandas as pd


class DataFrameStorage:

	"""

	"""

	def __init__(self, dataset_name, configuration_name):

		"""

		"""

		self.dataset_ = dataset_name
		self.configuration_ = configuration_name
		self.df_ = []


class Results:

	"""
	"""

	def __init__(self):

		"""

		"""

		self.dataframes_ = []


	def getDataFrame(self, dataset, configuration):

		"""
			Method that looks if a dataframe for a given dataset and configuration already exists, if not, creates it

			Receives:

				- Dataset
				- Configuration

			Returns:

				- DFS: DataFrameStorage object which contains information about execution from that combination
		"""

		for dfs in self.dataframes_:

			if dfs.dataset_ == dataset and dfs.configuration_ == configuration:
				return dfs

		dfs = DataFrameStorage(dataset, configuration)
		self.dataframes_.append(dfs)

		return dfs



	def addRecord(self, dataset, configuration, train_metrics, test_metrics, best_params):

		"""
			Stores all info about the run of a dataset with specified configuration.

			The info will be stored as a pandas DataFrame in a class named DataFrameStorage built in
			for the purpose of keeping additional information.

		"""

		dfs = self.getDataFrame(dataset, configuration)

		dataframe_row = collections.OrderedDict()
		# Adding best parameters as first columns in dataframe
		for p_name, p_value in best_params.items():

			# If some ensemble method has been used, then one of its parameters will be a dict containing
			# the best parameters found for the internal algorithm
			if type(p_value) == dict:
				for (k, v) in p_value.iteritems():
					dataframe_row[k] = v
			else:
				dataframe_row[p_name] = p_value

		# Concatenating train and test metrics
		for (tm_name, tm_value), (ts_name, ts_value) in zip(train_metrics.items(), test_metrics.items()):

			dataframe_row[tm_name] = tm_value
			dataframe_row[ts_name] = ts_value

		dfs.df_.append(dataframe_row)


	def createSummary(self, df, avg_index, std_index):

		"""
			Summarizing information from all partitions of configuration into one line

		"""

		# Dissociating train and test metrics

		n_parameters = len(df.columns) - len(avg_index)*2	#Number of parameters used in this configuration
		train_df = df.iloc[:,n_parameters::2].copy() 		#Even columns from dataframe (train metrics)
		test_df = df.iloc[:,(n_parameters+1)::2].copy()		#Odd columns (test metrics)

		# Computing mean and standard deviation for metrics
		train_avg, train_std = train_df.mean(), train_df.std()
		test_avg, test_std = test_df.mean(), test_df.std()
		# Naming indexes for summary dataframes
		train_avg.index, train_std.index = avg_index, std_index
		test_avg.index, test_std.index = avg_index, std_index
		# Merging avg and std into one dataframe
		train_summary_row = pd.concat([train_avg, train_std])
		test_summary_row = pd.concat([test_avg, test_std])

		# Mixing avg and std dataframe columns results from metrics summaries
		train_summary_row = train_summary_row[list(sum(zip(	train_summary_row.iloc[:len(avg_index)].keys(),\
											 				train_summary_row.iloc[len(std_index):].keys()), ()))]
		test_summary_row = test_summary_row[list(sum(zip(test_summary_row.iloc[:len(avg_index)].keys(),\
											 			 test_summary_row.iloc[len(std_index):].keys()), ()))]
		return train_summary_row, test_summary_row
		

	def saveResults(self, api_path, summary_index, metrics_names):

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
		train_summary = []; test_summary = []

		# Name of columns for summary dataframes
		avg_index, std_index = [mn + '_mean' for mn in metrics_names], [mn + '_std' for mn in metrics_names]
		for dataframe in self.dataframes_:

			# Creates subfolders for each dataset
			dataset_folder = folder_path + dataframe.dataset_ + "/"
			if not os.path.exists(dataset_folder):
				os.makedirs(dataset_folder)

			# Saving each dataframe
			df = pd.DataFrame(dataframe.df_)
			df.to_csv(dataset_folder + dataframe.dataset_ + "-" + dataframe.configuration_ + ".csv")

			# Creating one entry for dataframe in summaries
			tr_sr, ts_sr = self.createSummary(df, avg_index, std_index)
			train_summary.append(tr_sr); test_summary.append(ts_sr)

		# Naming each row in datasets
		train_summary = pd.concat(train_summary, axis=1).transpose(); train_summary.index = summary_index
		test_summary = pd.concat(test_summary, axis=1).transpose(); test_summary.index = summary_index

		# Save summaries to csv
		train_summary.to_csv(folder_path + "/" + "train_summary.csv")
		test_summary.to_csv(folder_path + "/" + "test_summary.csv")





