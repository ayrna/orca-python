
import os, sys, collections
import unittest

import numpy.testing as npt
import pandas.util.testing as pdt
import numpy as np

sys.path.append('../Algorithms')
from OrdinalDecomposition import OrdinalDecomposition

# TODO: Testear este metodo como lo hacen en los tests del clasificador de naive bayes en scikit learn ???

class TestOrdinalDecomposition(unittest.TestCase):

	# Data is just 6 separable points in the plane
	X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
	y = np.array([1, 1, 1, 2, 2, 2])



	def test_coding_matrix(self):

		od = OrdinalDecomposition()

		# Checking OrderedPartitions (with a 5 class, 4 classifiers example)
		expected_cm = np.array([[-1,-1,-1,-1], [1,-1,-1,-1], [1,1,-1,-1], [1,1,1,-1], [1,1,1,1]])
		actual_cm = od._codingMatrix(5, 'OrderedPartitions')

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking OneVsNext
		expected_cm = np.array([[-1,0,0,0], [1,-1,0,0], [0,1,-1,0], [0,0,1,-1], [0,0,0,1]])
		actual_cm = od._codingMatrix(5, 'OneVsNext')

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking OneVsFollowers
		expected_cm = np.array([[-1,0,0,0], [1,-1,0,0], [1,1,-1,0], [1,1,1,-1], [1,1,1,1]])
		actual_cm = od._codingMatrix(5, 'OneVsFollowers')

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking OneVsPrevious
		expected_cm = np.array([[1,1,1,1], [1,1,1,-1], [1,1,-1,0], [1,-1,0,0], [-1,0,0,0]])
		actual_cm = od._codingMatrix(5, 'OneVsPrevious')

		npt.assert_array_equal(actual_cm, expected_cm)




	def test_od(self):

		od = OrdinalDecomposition("OrderedPartitions", "sklearn.svm.SVC", {'C': 1.0, 'gamma': "scale", 'probability': True})
		y_pred = od.fit(self.X, self.y).predict(self.X)
		npt.assert_array_equal(y_pred, self.y)


	"""
	def test_positive_class(self):

		np.random.seed(0)
		rng = np.random.RandomState(0)
		X2 = rng.randint(5, size=(9, 10))
		y2 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

		X3 = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
		y3 = np.array([1, 1, 1, 2, 2, 2])

		X4 = np.array([[0, 5], [-1, 4], [1, 4], [5, 0], [4, 1], [4, -1], [0, -5], [-1, -4], [1, -4], [-5, 0], [-4, -1], [-4, 1]])
		y4 = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

		od = OrdinalDecomposition("OneVsPrevious", "sklearn.svm.SVC", {'C': 1, 'gamma': "scale", 'probability': True})
		y_pred = od.fit(X4, y4).predict(X4)

		print '\nPredicted Y:\n', y_pred
	"""


if __name__ == '__main__':
	unittest.main()
