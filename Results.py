
import os, datetime, collections

import numpy as np
import pandas as pd
import cPickle as pickle


class DataFrameStorage:

	"""
	DataFrameStorage

	Stores all metrics scores (train and test) for a given combiantion
	of dataset and configuration. It will contain a dataframe for each
	partition in which the dataset could be divided. If its not 
	partitionated, there'll be just one dataframe.
	
	Parameters
	----------

	dataset_name: string
		Name of dataset used

	configuration_name: string
		Name of configuration used


	Attributes
	----------

	df_: dict of OrderedDict
		Each dict contains the parameter's values with which the cross-validation
		metrics has been maximized (best parameters) during cross-validation
		phase, besides train and test scores for all different metrics specified.
		There will be as many dicts in the list as partitions the dataset is
		fragmented in.

	models_: dict
		Dictionary containing best found model for each partition, where the
		number of partition will act as key.

	predictions_: dict of dicts
		Dictionary containing array of predicted labels for train and test sets.
		The number of partition will act as key.

	"""

	def __init__(self, dataset_name, configuration_name):

		self.dataset_ = dataset_name
		self.configuration_ = configuration_name

		self.df_ = {}

		self.models_ = {}
		self.predictions_ = {}


class Results:

	"""
	Results

	Class that handles all info from an experiment that needs to be saved.
	This info will be saved in CSV's inside a dedicated folder

	Attributes
	----------

	dataframes_: list of DataFrameStorage objects
		Each object will storage information about a pair of 
		dataset-configuration. There will be as many as the number of
		combinations of different datasets and configurations.

	"""

	def __init__(self):

		self.dataframes_ = []


	def getDataFrame(self, dataset_name, configuration_name):

		"""
		Look if a dataframe for a given dataset and configuration already
		exists, if not, creates it.

		Parameters
		----------

		dataset_name : string
			Name of dataset used

		configuration_name : string
			Name of configuration used

		Returns
		-------

		dfs : DataFrameStorage object 
			Contains (or will contain) train and test metrics for 'dataset'
			and 'configuration' given values

		"""

		# Searchs if this combination of 'dataset' and 'configuration'
		# has already been used 
		for dfs in self.dataframes_:

			if dfs.dataset_ == dataset_name and dfs.configuration_ == configuration_name:
				return dfs

		# If the dataframe has yet to be added, we create it
		dfs = DataFrameStorage(dataset_name, configuration_name)
		self.dataframes_.append(dfs)

		return dfs



	def addRecord(self, partition, best_params, best_model, configuration, metrics, predictions):

		"""
		Stores information about the run of a partition into a
		DataFrameStorage object.

		Parameters
		----------

		partition: int ot string
			Number of partition to store.

		best_params: dictionary
			Best parameters found during cross-validation for this
			classifier.

		best_model: estimator
			Best found model of classifier during cross-validation.

		configuration: dict
			Dictionary containing the name used for this pair of
			dataset and configuration

		metrics: dict of dictionaries
			Dictionary containing the metrics for train and test for this
			particular configuration.

		predictions: dict of dictionaries
			Dictionary that stores train and test class predictions.

		"""

		# Get or create a DataFrameStorage object for this dataset and configuration
		dfs = self.getDataFrame(configuration['dataset'], configuration['config'])


		dataframe_row = collections.OrderedDict()
		# Adding best parameters as first elements in OrderedDict
		for p_name, p_value in best_params.items():

			# If some ensemble method has been used, then one of its parameters will 
			# be a dict containing the best parameters found for the meta classifier
			if type(p_value) == dict:
				for (k, v) in p_value.iteritems():
					dataframe_row[k] = v
			else:
				dataframe_row[p_name] = p_value


		# Concatenating train and test metrics
		for (tm_name, tm_value), (ts_name, ts_value) in zip(metrics['train'].items(), metrics['test'].items()):

			dataframe_row[tm_name] = tm_value
			dataframe_row[ts_name] = ts_value

		# Adding others variables
		dataframe_row['refit_time'] = metrics['train']['refit_time']

		# Adding this OrderedDict as a new entry to DataFrameStorage object
		dfs.df_[str(partition)] = dataframe_row


		# Storing models and predictions for this partition
		dfs.models_[str(partition)] = best_model
		dfs.predictions_[str(partition)] = predictions


	def saveResults(self, runs_folder, summary_index, metrics_names):

		"""
		Method used for saving experiment info to CSV's.

		By default, there will be a dedicated subfolder inside framework's one.

		Each time a experiment has been run successfully, this method will 
		generate a new subfolder inside that subfolder, named 
		'exp-YY-MM-DD-hh-mm-ss'.

		This new generated folder will store the train and test summaries 
		as CSV, as well as so many subfolders as datasets, named after them.

		Last, inside this dataset subfolders, there will be one CSV for each
		configuration used, containing info about metrics and partitions.


		"""


		# Transforming given path to absolute path if neccesary
		if not runs_folder.startswith("/"):

			fw_path = os.path.dirname(os.path.abspath(__file__)) + "/"
			runs_folder = fw_path + runs_folder

		if not runs_folder.endswith("/"):
			runs_folder += "/"
		

		# Check if experiments folder exists
		if not os.path.exists(runs_folder):
			os.makedirs(runs_folder)



		# Getting name of folder where we will store info about the Experiment
		folder_name = "exp-" + datetime.date.today().strftime("%y-%m-%d") + "-" \
				+ datetime.datetime.now().strftime("%H-%M-%S") + "/"


		# Check if folder already exists
		folder_path = runs_folder + folder_name
		try: os.makedirs(folder_path)
		except OSError: raise OSError("Could not create folder %s to store results. It already exists" % folder_path)


		# Saving summaries from every combination of DB and Configuration
		train_summary = []; test_summary = []

		# Name of columns for summary dataframes
		avg_index = [mn + '_mean' for mn in metrics_names]
		std_index = [mn + '_std' for mn in metrics_names]

		for dataframe in self.dataframes_:

			# Creates subfolders for each dataset
			dataset_folder = folder_path + dataframe.dataset_ + "/"
			try: os.makedirs(dataset_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % dataset_folder)

			# Saving each dataframe
			df = pd.DataFrame([row for partition,row in sorted(dataframe.df_.items())])
			df.to_csv(dataset_folder + dataframe.dataset_ + "-" + dataframe.configuration_ + ".csv")

			# Creating one entry for dataframe in summaries
			tr_sr, ts_sr = self.createSummary(df, avg_index, std_index)
			train_summary.append(tr_sr); test_summary.append(ts_sr)

			# Saving models generated for each partition in one folder
			models_folder = dataset_folder + "models/"
			try: os.makedirs(models_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % models_folder)

			for part, model in dataframe.models_.iteritems():

				model_filename = dataframe.dataset_ + "-" + dataframe.configuration_ + "." + part
				with open(models_folder + model_filename, 'wb') as output:

					pickle.dump(model, output)


			# Saving predictions
			predictions_folder = dataset_folder + "predictions/"
			try: os.makedirs(predictions_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % predictions_folder)

			for part, predictions in dataframe.predictions_.iteritems():

				pred_filename = dataframe.dataset_ + "-" + dataframe.configuration_ + "." + part
				np.savetxt(predictions_folder + 'train_' + pred_filename, predictions['train'], fmt='%d')
				np.savetxt(predictions_folder + 'test_' + pred_filename, predictions['test'], fmt='%d')


		# Naming each row in datasets
		train_summary = pd.concat(train_summary, axis=1).transpose(); train_summary.index = summary_index
		test_summary = pd.concat(test_summary, axis=1).transpose(); test_summary.index = summary_index

		# Save summaries to csv
		train_summary.to_csv(folder_path + "/" + "train_summary.csv")
		test_summary.to_csv(folder_path + "/" + "test_summary.csv")




	def createSummary(self, df, avg_index, std_index):

		"""
		Summarizing information from all partitions stored in a DataFrameStorage 
		object into one line of a DataFrame


		Parameters
		----------

			df: DataFrame
				Object that stores train and test metrics, as well as the parameters
				used to obtain them, for all partitions of a dataset with a given
				configuration

			avg_index: list of strings
				Includes all names of metrics calculated ending with '_mean'

			std_index: list of strings
				Includes all names of metrics calculated ending with '_std'
			

		Returns
		-------
	
			train_summary_row: DataFrame
				DataFrame with only one row, containing mean and standard deviation
				for all metrics calculated across partitions.

				Initial column will indicate dataset-configuration pair from which
				we are summarizing info from.

			test_summary_row: DataFrame
				Simmilar to train_summary_row, but storing only info about test scores

		"""

		# Dissociating train and test metrics (last 3 columns are computational times)
		n_parameters = len(df.columns) - len(avg_index)*2 - 3				# Number of parameters used in this configuration
		train_df = df.iloc[:,n_parameters:len(df.columns)-3:2].copy() 		# Even columns from dataframe (train metrics)
		test_df = df.iloc[:,(n_parameters+1):len(df.columns)-3:2].copy()	# Odd columns (test metrics)


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
		



