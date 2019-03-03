import os
import sys
import collections

import unittest

from numpy import array
import numpy.testing as npt
import pandas.util.testing as pdt

sys.path.append('../classifiers')

from OrdinalDecomposition import OrdinalDecomposition

# TODO: Testear este metodo como lo hacen en los tests del clasificador de naive bayes en scikit learn ???

class TestOrdinalDecomposition(unittest.TestCase):

	# Data is just 6 separable points in the plane
	X = array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
	y = array([1, 1, 1, 2, 2, 2])


	def test_coding_matrix(self):

		od = OrdinalDecomposition()

		# Checking ordered_partitions (with a 5 class, 4 classifiers example)
		od.dtype = 'ordered_partitions'
		expected_cm = array([[-1,-1,-1,-1], [1,-1,-1,-1], [1,1,-1,-1], [1,1,1,-1], [1,1,1,1]])
		actual_cm = od._coding_matrix(5)

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking one_vs_next
		od.dtype = 'one_vs_next'
		expected_cm = array([[-1,0,0,0], [1,-1,0,0], [0,1,-1,0], [0,0,1,-1], [0,0,0,1]])
		actual_cm = od._coding_matrix(5)

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking one_vs_followers
		od.dtype = 'one_vs_followers'
		expected_cm = array([[-1,0,0,0], [1,-1,0,0], [1,1,-1,0], [1,1,1,-1], [1,1,1,1]])
		actual_cm = od._coding_matrix(5)

		npt.assert_array_equal(actual_cm, expected_cm)

		# Checking one_vs_previous
		od.dtype = 'one_vs_previous'
		expected_cm = array([[1,1,1,1], [1,1,1,-1], [1,1,-1,0], [1,-1,0,0], [-1,0,0,0]])
		actual_cm = od._coding_matrix(5)

		npt.assert_array_equal(actual_cm, expected_cm)


	#TODO: Test decision methods outputs and check if they are correct (compute them on paper)
	def test_decision_method(self):

		# Checking Frank and Hall method cannot be used whitout ordered partitions
		od = OrdinalDecomposition(dtype="one_vs_next", decision_method="frank_hall")
		npt.assert_raises(AttributeError, od._frank_hall_method, self.X)

		# 
		od = OrdinalDecomposition(dtype="", decision_method="")




	def test_ordinal_decomposition(self):

		od = OrdinalDecomposition(dtype="ordered_partitions", decision_method="frank_hall",\
								base_classifier="sklearn.svm.SVC",\
								parameters={'C': 1.0, 'gamma': "scale", 'probability': True})

		y_pred = od.fit(self.X, self.y).predict(self.X)
		npt.assert_array_equal(y_pred, self.y)




if __name__ == '__main__':
	unittest.main()
