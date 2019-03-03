from sys import path as syspath
from os import path as ospath
from os import listdir
from os import walk
from shutil import rmtree
from collections import OrderedDict
from pickle import load

import unittest

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
from sklearn.svm import SVC

syspath.append('../')
from results import ReportUnit
from results import Results



class TestResults(unittest.TestCase):


	_results = Results()


	def test_get_report_unit(self):

		# Creating two different RU objects
		dfs1 = ReportUnit('toy', 'conf_1')
		dfs2 = ReportUnit('iris', 'conf_1')

		self._results._reports = [dfs1, dfs2]

		# Check that get_report_unit method returns returns the same object when using same parameters
		npt.assert_equal(self._results._get_report_unit('toy', 'conf_1'), self._results._reports[0])
		npt.assert_equal(self._results._get_report_unit('iris', 'conf_1'), self._results._reports[1])
		# Check that it creates a new object when using a not used combination of dataset and configuration
		npt.assert_equal(self._results._get_report_unit('test', 'conf_1'), self._results._reports[-1])

		# Resetting Results object
		self._results._reports = []



	def test_add_record(self):

		# Adding row to ReportUnit object
		partition = 0
		dataset = 'toy'
		configuration = 'conf_1'

		estimator = SVC()
		best_params = OrderedDict([('C', 0.1), ('gamma', 1)])

		train_metrics = OrderedDict([('ccr_train', 0.7222222222), ('mae_train', 0.2777777777)])
		test_metrics = OrderedDict([('ccr_test', 0.6666666666), ('mae_test', 0.3333333333)])

		train_predicted_y = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3])
		test_predicted_y = np.array([1,1,2,2,2,3,3])

		self._results.add_record(partition, best_params, estimator,
								{'dataset': dataset, 'config': configuration},
								{'train': train_metrics, 'test': test_metrics},
								{'train': train_predicted_y, 'test': test_predicted_y})



		# Adding second row (second partition) to ReportUnit object
		partition = 1
		dataset = 'toy'
		configuration = 'conf_1'

		best_params = OrderedDict([('C', 1), ('gamma', 1)])

		train_metrics = OrderedDict([('ccr_train', 0.9333333333), ('mae_train', 0.2777777777)])
		test_metrics = OrderedDict([('ccr_test', 1.0), ('mae_test', 0.3333333333)])

		train_predicted_y = np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3])
		test_predicted_y = np.array([1,1,2,1,2,3,3])

		self._results.add_record(partition, best_params, estimator,
								{'dataset': dataset, 'config': configuration},
								{'train': train_metrics, 'test': test_metrics},
								{'train': train_predicted_y, 'test': test_predicted_y})


		# Checking if actual and expected metrics are the same
		expected_data = {'0': OrderedDict([('C', 0.1), ('gamma', 1),\
											('ccr_train', 0.7222222222), ('ccr_test', 0.6666666666),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)]),\

						'1': OrderedDict([('C', 1), ('gamma', 1),\
											('ccr_train', 0.9333333333), ('ccr_test', 1.0),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)])}
		actual_data = self._results._get_report_unit(dataset, configuration).metrics
		npt.assert_equal(actual_data, expected_data)


		# Checking if actual and expected models are the same
		expected_data = {'0': estimator, '1': estimator}
		actual_data = self._results._get_report_unit(dataset, configuration).models
		npt.assert_equal(actual_data, expected_data)


		# Checking if actual and expected predictions are the same
		expected_data = {'0': {'train': np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3]), 'test': np.array([1,1,2,2,2,3,3])},
						'1': {'train': np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3]), 'test': np.array([1,1,2,1,2,3,3])}}
		actual_data = self._results._get_report_unit(dataset, configuration).predictions
		npt.assert_equal(actual_data, expected_data)


		# Resetting Results object
		self._results._reports = []



	def test_create_summary(self):


		dataset = 'toy'
		configuration = 'conf_1'

		best_params = OrderedDict([('C', 0.1), ('gamma', 1)])
		estimator = SVC()

		train_metrics = OrderedDict([('ccr_train', 0.7222222222), ('mae_train', 0.2777777777)])
		test_metrics = OrderedDict([('ccr_test', 0.6666666666), ('mae_test', 0.3333333333)])

		train_predicted_y = np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3])
		test_predicted_y = np.array([1,1,2,1,2,3,3])

		# Adding two identical rows as two partitions
		self._results.add_record(0, best_params, estimator,
								{'dataset': dataset, 'config': configuration},
								{'train': train_metrics, 'test': test_metrics},
								{'train': train_predicted_y, 'test': test_predicted_y})

		self._results.add_record(1, best_params, estimator,
								{'dataset': dataset, 'config': configuration},
								{'train': train_metrics, 'test': test_metrics},
								{'train': train_predicted_y, 'test': test_predicted_y})


		mean_index = ['ccr_mean', 'mae_mean']
		std_index = ['ccr_std', 'mae_std']

		data = self._results._get_report_unit(dataset, configuration).metrics
		df = pd.DataFrame(data=[row for partition,row in sorted(data.items())])
		train_row, test_row = self._results._create_summary(df, mean_index, std_index)

		# Desired row values and indexes
		desired_train_row = pd.Series(data=OrderedDict([('ccr_mean', 0.7222222222), ('ccr_std', 0.0),\
														('mae_mean', 0.2777777777), ('mae_std', 0.0)]),\
										index=['ccr_mean','ccr_std','mae_mean','mae_std'])

		desired_test_row = pd.Series(data=OrderedDict([('ccr_mean', 0.6666666666), ('ccr_std', 0.0),\
														('mae_mean', 0.3333333333), ('mae_std', 0.0)]),\
										index=['ccr_mean','ccr_std','mae_mean','mae_std'])

		# Check series similarity
		pdt.assert_series_equal(train_row, desired_train_row)
		pdt.assert_series_equal(test_row, desired_test_row)

		# Resetting Results object
		self._results._reports = []



	def test_save_results(self):


		# First ReportUnit object
		ru1 = ReportUnit("dataset1", "config1")
		ru1.metrics = {'0': OrderedDict([('C', 0.1), ('gamma', 0.1),\
											('ccr_train', 0.7222222222), ('ccr_test', 0.9666666666),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)]),\

						'1': OrderedDict([('C', 1), ('gamma', 1),\
											('ccr_train', 0.9333333333), ('ccr_test', 1.0),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)])}

		ru1.models = {'0': SVC(C=0.1, gamma=0.1),
					'1': SVC(C=1, gamma=1)}

		ru1.predictions = {'0': {'train': np.array([1,1,1,1,1,2,2,3,3,3,3,3,3,3]), 'test': np.array([1,1,2,2,2,3,3])},
						'1': {'train': np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3]), 'test': np.array([1,1,2,1,2,3,3])}}

		# Secong ReportUnit object
		ru2 = ReportUnit("dataset1", "config2")
		ru2.metrics = {'0': OrderedDict([('C', 0.1), ('gamma', 0.1),\
											('ccr_train', 0.7222222222), ('ccr_test', 0.6666666666),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)]),\

						'1': OrderedDict([('C', 1), ('gamma', 1),\
											('ccr_train', 0.8333333333), ('ccr_test', 1.0),\
											('mae_train', 0.5777777777), ('mae_test', 0.5333333333)])}

		ru2.models = {'0': SVC(C=0.1, gamma=0.1),
					'1': SVC(C=1, gamma=1)}

		ru2.predictions = {'0': {'train': np.array([1,1,1,1,1,2,2,3,3,3,3,3,3,3]), 'test': np.array([1,1,2,2,2,3,3])},
						'1': {'train': np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3]), 'test': np.array([1,1,2,1,2,3,3])}}


		# Saving results to sample folder
		main_folder = ospath.dirname(ospath.abspath(__file__)) + '/'
		output_folder = main_folder + "sample_run/"

		self._results._reports = [ru1, ru2]
		self._results.save_results(output_folder, ['ccr', 'mae'])

		# Checking if all elements stored are correct
		experiment_folder = output_folder + listdir(output_folder)[-1] + '/'


		# Checking summaries
		df_ru1 = pd.DataFrame(ru1.metrics).T
		df_ru2 = pd.DataFrame(ru2.metrics).T

		# Expected metrics mean and std
		expected_train_summary = pd.DataFrame({'ccr_mean': [np.mean(df_ru1['ccr_train']), np.mean(df_ru2['ccr_train'])],
												'ccr_std': [np.std(df_ru1['ccr_train'], ddof=1), np.std(df_ru2['ccr_train'], ddof=1)],
												'mae_mean': [np.mean(df_ru1['mae_train']), np.mean(df_ru2['mae_train'])],
												'mae_std': [np.std(df_ru1['mae_train'], ddof=1), np.std(df_ru2['mae_train'], ddof=1)]
												}, index=['dataset1-config1', 'dataset1-config2'])

		expected_test_summary = pd.DataFrame({'ccr_mean': [np.mean(df_ru1['ccr_test']), np.mean(df_ru2['ccr_test'])],
												'ccr_std': [np.std(df_ru1['ccr_test'], ddof=1), np.std(df_ru2['ccr_test'], ddof=1)],
												'mae_mean': [np.mean(df_ru1['mae_test']), np.mean(df_ru2['mae_test'])],
												'mae_std': [np.std(df_ru1['mae_test'], ddof=1), np.std(df_ru2['mae_test'], ddof=1)]
												}, index=['dataset1-config1', 'dataset1-config2'])


		# Reading summaries from CSVs (index is not readed properly)
		actual_train_summary = pd.read_csv(experiment_folder + 'train_summary.csv').iloc[:,1:]
		actual_train_summary.index = ['dataset1-config1', 'dataset1-config2']
		actual_test_summary = pd.read_csv(experiment_folder + 'test_summary.csv').iloc[:,1:]
		actual_test_summary.index = ['dataset1-config1', 'dataset1-config2']


		pdt.assert_frame_equal(actual_train_summary, expected_train_summary)
		pdt.assert_frame_equal(actual_test_summary, expected_test_summary)


		# Going through all subdirectories (pairs dataset-configuration)
		subdirectories = sorted(next(walk(experiment_folder))[1])
		subdirectories = [experiment_folder + s + '/' for s in subdirectories]

		for subdir, ru in zip(subdirectories, self._results._reports):

			# Checking metrics
			actual_metrics = pd.read_csv(subdir + ru.dataset + "-" + ru.configuration + ".csv").iloc[:,1:]
			actual_metrics.index = ['0', '1']
			expected_metrics = pd.DataFrame(ru.metrics, index=actual_metrics.columns).T

			pdt.assert_frame_equal(actual_metrics, expected_metrics)

			# Checking models
			models_folder = subdir + "models/"
			for expected_model_key, actual_model in zip(sorted(ru.models.keys()), sorted(listdir(models_folder))):

				model_file = open(models_folder + actual_model, 'rb')
				model_object = load(model_file)

				npt.assert_equal(ru.models[expected_model_key].get_params(), model_object.get_params())
				model_file.close()

			# Checking predictions
			predictions_folder = subdir + "predictions/"
			for i in range(len(self._results._reports)):

				train_prediction_file = open(predictions_folder + 'train_' + ru.dataset + '-' + ru.configuration + \
											'.' + str(i), 'rb')
				test_prediction_file = open(predictions_folder + 'test_' + ru.dataset + '-' + ru.configuration + \
											'.' + str(i), 'rb')

				actual_train_predictions = np.loadtxt(train_prediction_file)
				actual_test_predictions = np.loadtxt(test_prediction_file)

				npt.assert_array_equal(actual_train_predictions, ru.predictions[str(i)]['train'])
				npt.assert_array_equal(actual_test_predictions, ru.predictions[str(i)]['test'])

				train_prediction_file.close()
				test_prediction_file.close()



		rmtree(output_folder)
		# Ressetting results object
		self._results._reports = []


# Running all tests
if __name__ == "__main__":
	unittest.main()
