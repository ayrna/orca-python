import os, sys, collections
import unittest

import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np
import pandas as pd

from sklearn.svm import SVC

sys.path.append('../')
from Results import DataFrameStorage
from Results import Results



class TestResults(unittest.TestCase):


	r_ = Results()


	def test_get_dataframe(self):

		# Creating two different DFS objects
		data_row = {'C': 0.1, 'gamma': 1, 'ccr_train': 0.7222222222, 'ccr_test': 0.6666666666, \
					'mae_train': 0.2777777777, 'mae_test': 0.3333333333}

		dfs1 = DataFrameStorage('toy', 'conf_1'); 	dfs1.df_[0] = data_row
		dfs2 = DataFrameStorage('iris', 'conf_1'); 	dfs2.df_[0] = data_row

		self.r_.dataframes_ = [dfs1, dfs2]

		# Check that getDataFrame method returns returns the same object when using same parameters
		npt.assert_equal(self.r_.getDataFrame('toy', 'conf_1'), self.r_.dataframes_[0])
		npt.assert_equal(self.r_.getDataFrame('iris', 'conf_1'), self.r_.dataframes_[1])
		# Check that it creates a new object when using a not used combination of dataset and configuration
		npt.assert_equal(self.r_.getDataFrame('test', 'conf_1'), self.r_.dataframes_[-1])

		self.r_.dataframes_ = []



	def test_add_record(self):

		# Adding row to DataFrameStorage object
		partition = 0
		dataset = 'toy'
		configuration = 'conf_1'

		estimator = SVC()
		best_params = collections.OrderedDict([('C', 0.1), ('gamma', 1)])

		train_metrics = collections.OrderedDict([('ccr_train', 0.7222222222), ('mae_train', 0.2777777777)])
		test_metrics = collections.OrderedDict([('ccr_test', 0.6666666666), ('mae_test', 0.3333333333)])

		train_predicted_y = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3])
		test_predicted_y = np.array([1,1,2,2,2,3,3])

		self.r_.addRecord(partition, best_params, estimator,
						{'dataset': dataset, 'config': configuration},
						{'train': train_metrics, 'test': test_metrics},
						{'train': train_predicted_y, 'test': test_predicted_y}))



		# Adding second row (second partition) to DataFrameStorage object
		partition = 1
		dataset = 'toy'
		configuration = 'conf_1'

		estimator = SVC()
		best_params = collections.OrderedDict([('C', 0.1), ('gamma', 1)])

		train_metrics = collections.OrderedDict([('ccr_train', 0.9333333333), ('mae_train', 0.2777777777)])
		test_metrics = collections.OrderedDict([('ccr_test', 1.0), ('mae_test', 0.3333333333)])

		train_predicted_y = np.array([1,1,1,1,1,2,2,3,3,2,3,3,3,3])
		test_predicted_y = np.array([1,1,2,1,2,3,3])

		self.r_.addRecord(partition, best_params, estimator,
						{'dataset': dataset, 'config': configuration},
						{'train': train_metrics, 'test': test_metrics},
						{'train': train_predicted_y, 'test': test_predicted_y}))


		# Expected data consists of two rows with given values (stored as a list of OrderedDicts)
		expected_data = [collections.OrderedDict([('C', 0.1), ('gamma', 1),\
											('ccr_train', 0.7222222222), ('ccr_test', 0.6666666666),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)]),\

						collections.OrderedDict([('C', 1), ('gamma', 1),\
											('ccr_train', 0.9333333333), ('ccr_test', 1.0),\
											('mae_train', 0.2777777777), ('mae_test', 0.3333333333)])]

		# Comparing than both lists are similar
		actual_data = self.r_.getDataFrame(dataset, configuration).df_
		npt.assert_equal(actual_data, expected_data)

		self.r_.dataframes_ = []



	def test_create_summary(self):


		dataset = 'toy'
		configuration = 'conf_1'

		train_metrics = collections.OrderedDict([('ccr_train', 0.7222222222), ('mae_train', 0.2777777777)])
		test_metrics = collections.OrderedDict([('ccr_test', 0.6666666666), ('mae_test', 0.3333333333)])
		best_params = collections.OrderedDict([('C', 0.1), ('gamma', 1)])

		# Adding two identical rows as two partitions
		self.r_.addRecord(dataset, configuration, train_metrics, test_metrics, best_params)
		self.r_.addRecord(dataset, configuration, train_metrics, test_metrics, best_params)

		mean_index = ['ccr_mean', 'mae_mean']
		std_index = ['ccr_std', 'mae_std']

		train_row, test_row = self.r_.createSummary( pd.DataFrame(self.r_.getDataFrame(dataset, configuration).df_),\
													 mean_index, std_index )

		# Desired row values and indexes
		desired_train_row = pd.Series(data=collections.OrderedDict([ ('ccr_mean', 0.7222222222), ('ccr_std', 0.0),\
													('mae_mean', 0.2777777777), ('mae_std', 0.0) ]), \
										index=['ccr_mean','ccr_std','mae_mean','mae_std'])

		desired_test_row = pd.Series(data=collections.OrderedDict([ ('ccr_mean', 0.6666666666), ('ccr_std', 0.0),\
													('mae_mean', 0.3333333333), ('mae_std', 0.0) ]), \
										index=['ccr_mean','ccr_std','mae_mean','mae_std'])

		# Check series similarity
		pdt.assert_series_equal(train_row, desired_train_row)
		pdt.assert_series_equal(test_row, desired_test_row)

		self.r_.dataframes_ = []



	def test_save_results(self):

		#TODO: Hacer que se ejecute al completo y luego comprobar los resultados, cargando la informacion
		#		guardada de nuevo y compobando que es similar a lo esperado.

		pass



# Running all tests
if __name__ == "__main__":
	unittest.main()
