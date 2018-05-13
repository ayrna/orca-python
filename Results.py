
import pandas as pd



class DataFrameStorage:

	"""

	"""

	def __init__(self, dataset, algorithm):

		"""

		"""

		self.dataset_ = dataset
		self.algorithm_ = algorithm
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

		pass














