import os
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle


class ReportUnit:

	"""
	ReportUnit

	Stores all metric's scores (train and test) for an unique 
	combination of dataset and configuration, besides best models found
	during cross-validation, predictions obtained with them and 
	computational times.

	Only created in add_record function of Results class.
	
	Parameters
	----------

	dataset_name: string
		Name of dataset used

	configuration_name: string
		Name of configuration used


	Attributes
	----------

	metrics: dict of OrderedDict
		Each ordered dictionary contains: the set of best 
		hyper-parameters found during cross-validation, the train and
		test scores for each metrics and the computational times.
		One per partition.
		The number of partition will act as key.

	models: dict
		Dictionary containing best found model. One per partition.
		The number of partition will act as key.

	predictions: dict of dicts
		Each inner dictionary contains two dicts, which, in turn,
		includes an array of predicted labels for train and test.
		The number of partition will act as key.

	"""

	def __init__(self, dataset_name, configuration_name):

		self.dataset = dataset_name
		self.configuration = configuration_name

		self.metrics = {}

		self.models = {}
		self.predictions = {}


class Results:

	"""
	Results

	Class that handles all information from an experiment that needs
	to be saved. This info will be saved into an specified folder.

	Attributes
	----------

	reports_: list of ReportUnit objects
		Each object will store information about a pair of 
		dataset-configuration. There will be as many as the number of
		combinations of these.
	"""

	def __init__(self):

		self._reports = []


	def _get_report_unit(self, dataset_name, configuration_name):

		"""
		Searches if a given configuration-dataset ReportUnit has been
		created. If not, creates it with the given values.

		Parameters
		----------

		dataset_name : string
			Name of dataset used

		configuration_name : string
			Name of configuration used

		Returns
		-------

		ru : ReportUnit object
		"""

		# Searchs for this combination of 'dataset' and 'configuration'
		for ru in self._reports:

			if ru.dataset == dataset_name and \
				ru.configuration == configuration_name:

				return ru

		# If the ReportUnit has yet to be added, creates it
		ru = ReportUnit(dataset_name, configuration_name)
		self._reports.append(ru)

		return ru



	def add_record(self, partition, best_params, best_model, configuration, metrics, predictions):

		"""
		Stores information about the run of a partition into a
		ReportUnit object.

		Parameters
		----------

		partition: int or string
			Number of partition to store.

		best_params: dictionary
			Best hyper-parameters found for this configuration and
			dataset during cross-validation. If an ensemble method
			has been used, there'll exist a parameter called
			'parameters' that will store a dict with the best
			hyper-parameters found for the base classifier.
			Keys are the name of each parameter

		best_model: estimator
			Best found classifier model during cross-validation.

		configuration: dict
			Dictionary containing the name used for this pair of
			dataset and configuration. Keys are 'dataset' and
			'configuration'.

		metrics: dict of dictionaries
			Dictionary containing the metrics for train and test for
			this particular configuration. Keys are 'train' and 'test'

		predictions: dict of lists
			Dictionary that stores train and test class predictions.
			Keys are 'train' and 'test'.

		"""

		# Get or create a ReportUnit object for this dataset and configuration
		ru = self._get_report_unit(configuration['dataset'],\
									configuration['config'])


		dataframe_row = OrderedDict()
		# Adding best parameters as first elements in OrderedDict
		for p_name, p_value in best_params.items():

			"""If some ensemble method has been used, then one of its
			parameters will be a dictionary containing the best parameters
			found for the meta classifier.
			"""
			if type(p_value) == dict:
				for (k, v) in p_value.items():
					dataframe_row[k] = v
			else:
				dataframe_row[p_name] = p_value


		# Concatenating train and test metrics
		for (tm_name, tm_value), (ts_name, ts_value) \
		in zip(metrics['train'].items(), metrics['test'].items()):

			dataframe_row[tm_name] = tm_value
			dataframe_row[ts_name] = ts_value


		# Adding this OrderedDict as a new entry to ReportUnit object
		ru.metrics[str(partition)] = dataframe_row

		# Storing models and predictions for this partition
		ru.models[str(partition)] = best_model
		ru.predictions[str(partition)] = predictions


	def save_results(self, output_folder, metrics_names):

		"""
		Method used for writing all the experiment information to files

		By default, there will be a dedicated subfolder inside 
		framework's main one. This default folder can be changed in
		Config.py or through configuration files.

		Each time a experiment has been run successfully, this method
		will generate a new subfolder inside that folder, named 
		'exp-YY-MM-DD-hh-mm-ss'. Where everything bar 'exp' is the
		date and hour the experiment finished running, respectively.

		This new generated folder will store the train and test
		summaries as CSV, as well as so many subfolders as 
		datasets-configurations pairs, named after them.

		Inside this specifics subfolders, there will be:

				- A CSV with one entry per partition, where there'll be
				stored the best found parameters during 
				cross-validation, train and test metrics and 
				computational times for building each model.

				- Models subfolder where it'll be stored the best model 
				built for each partition, writed as a Pickle.

				- Predictions subfolder with train and test label 
				predictions obtained with the best found model.

		Parameters
		----------

		output_folder: string
			Relative or absolute path where results will be stored.

		metrics_names: list of strings
			List with the names of all metrics used during the
			execution of the experiment.
		"""


		# Check if experiments folder exists
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		# Getting name of folder where we will store info about the experiment
		folder_name = "exp-"+datetime.date.today().strftime("%y-%m-%d")+"-" \
				+datetime.datetime.now().strftime("%H-%M-%S")


		# Check if folder already exists
		folder_path = os.path.join(output_folder, folder_name)
		try: os.makedirs(folder_path)
		except OSError: raise OSError("Could not create folder %s to store results. It already exists" % folder_path)


		# Saving summaries from every combination of DB and Configuration
		train_summary = []; test_summary = []
		summary_index = []

		# Name of columns for summary dataframes
		avg_index = [mn + '_mean' for mn in metrics_names]
		std_index = [mn + '_std' for mn in metrics_names]

		for report in self._reports:

			# Creates subfolders for each dataset
			dataset_folder = os.path.join(folder_path, (report.dataset + "-" + report.configuration))
			try: os.makedirs(dataset_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % dataset_folder)

			# Saving each dataframe
			df = pd.DataFrame([row for partition,row in sorted(report.metrics.items())])
			df.to_csv(os.path.join(dataset_folder, (report.dataset + "-" + report.configuration + ".csv")))

			# Creating one entry per ReportUnit in summaries
			tr_sr, ts_sr = self._create_summary(df, avg_index, std_index)
			train_summary.append(tr_sr); test_summary.append(ts_sr)
			summary_index.append(report.dataset.strip() + "-" + report.configuration)

			# Saving models generated for each partition into one folder
			models_folder = os.path.join(dataset_folder, "models")
			try: os.makedirs(models_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % models_folder)

			for part, model in report.models.items():

				model_filename = report.dataset + "-" + report.configuration + "." + part
				with open(os.path.join(models_folder, model_filename), 'wb') as output:

					pickle.dump(model, output)


			# Saving predictions
			predictions_folder = os.path.join(dataset_folder, "predictions")
			try: os.makedirs(predictions_folder)
			except OSError: raise OSError("Could not create folder %s to store results. It already exists" % predictions_folder)

			for part, predictions in report.predictions.items():

				pred_filename = report.dataset + "-" + report.configuration + "." + part
				np.savetxt(os.path.join(predictions_folder, 'train_' + pred_filename), predictions['train'], fmt='%d')
				if predictions['test'] is not None:
					np.savetxt(os.path.join(predictions_folder, 'test_' + pred_filename), predictions['test'], fmt='%d')


		# Naming each row in datasets
		train_summary = pd.concat(train_summary, axis=1).transpose(); train_summary.index = summary_index
		test_summary = pd.concat(test_summary, axis=1).transpose(); test_summary.index = summary_index

		# Save summaries to csv
		train_summary.to_csv(os.path.join(folder_path, "train_summary.csv"))
		test_summary.to_csv(os.path.join(folder_path, "test_summary.csv"))




	def _create_summary(self, df, avg_index, std_index):

		"""
		Summarices information from all partitions stored in a 
		ReportUnit object into one line of a DataFrame.

		Parameters
		----------

			df: DataFrame object
				Dataframe representing one ReportUnit.
				Contains hyper-parameters, metric's scores and
				computational times.

			avg_index: list of strings
				Includes all names of metrics ending with '_mean'

			std_index: list of strings
				Includes all names of metrics ending with '_std'

		Returns
		-------
	
			train_summary_row: DataFrame object
				DataFrame with only one row, containing mean and
				standard deviation for all metrics calculated
				across partitions (including computational times).
				Stores only train information.

			test_summary_row: DataFrame object
				Stores only test information
		"""

		# Dissociating train and test metrics
		n_parameters = len(df.columns) - len(avg_index)*2		# Number of parameters used in this configuration
		train_df = df.iloc[:,n_parameters::2].copy() 			# Even columns from dataframe (train metrics)
		test_df = df.iloc[:,(n_parameters+1)::2].copy()			# Odd columns (test metrics)


		# Computing mean and standard deviation for metrics
		train_avg, train_std = train_df.mean(), train_df.std()
		test_avg, test_std = test_df.mean(), test_df.std()
		# Naming indexes for summary dataframes
		train_avg.index = avg_index; train_std.index = std_index
		test_avg.index = avg_index; test_std.index = std_index
		# Merging avg and std into one dataframe
		train_summary_row = pd.concat([train_avg, train_std])
		test_summary_row = pd.concat([test_avg, test_std])

		# Mixing avg and std dataframe columns results from metrics summaries
		train_summary_row = train_summary_row[list(sum(zip(	train_summary_row.iloc[:len(avg_index)].keys(),\
									train_summary_row.iloc[len(std_index):].keys()), ()))]
		test_summary_row = test_summary_row[list(sum(zip(test_summary_row.iloc[:len(avg_index)].keys(),\
								test_summary_row.iloc[len(std_index):].keys()), ()))]
		return train_summary_row, test_summary_row
		



