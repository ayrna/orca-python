

class DSU:
	"""
	DSU - Data Storage Unit

	Abstraction level which we will work with.
	"""

	def __init__(self, dataset, partition):

		self.dataset = dataset
		self.partition = partition

		self.train_inputs = []
		self.train_outputs = []
		self.test_inputs = []
		self.test_outputs = []


