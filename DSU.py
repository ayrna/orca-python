

class DSU:
	"""
		DSU - Data Storage Unit

		Abstraction level which we will work with during all API.
	"""

	# Information about partition
	_dataset = ""
	_partition = ""

	# Data loaded from file
	_train_inputs = []
	_train_outputs = []
	_test_inputs = []
	_test_outputs = []

	# Predicted outputs from algorithms
	_train_predicted = []
	_test_predicted = []

	# Obtained metrics (As many as algorithms)
	_metrics = None

	def __init__(self, dataset, partition):

		self._dataset = dataset
		self._partition = partition


	def printInfo(self):

		print "Dataset:", self._dataset
		print "Partition:", self._partition

		print "Train Inputs:\t", self._train_inputs
		print "Train Outputs:\t", self._train_outputs
		print "Test Inputs:\t", self._test_inputs
		print "Test Outputs:\t", self._test_outputs


class ParamMetrics:

	"""

	"""

	_hyper_parameters = {}
	_train_metrics = []
	_test_metrics = []


	def __init__(self, hyper_parameters, train_metrics, test_metrics):


		"""

		"""

		self._hyper_parameters = hyper_parameters
		self._train_metrics = train_metrics
		self._test_metrics = test_metrics




