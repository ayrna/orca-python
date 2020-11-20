from sys import path as syspath
from os import path as ospath
import collections

import unittest

from numpy import array
import numpy.testing as npt
import pandas.testing as pdt

syspath.append(ospath.join('..', 'classifiers'))

from OrdinalDecomposition import OrdinalDecomposition

class TestOrdinalDecomposition(unittest.TestCase):
	"""
	Class testing OrdinalDecomposition's functionality.
	This class will make use of a toy problem, consisting of
	6 points in a plane, divided into two different classes.

	This classifier is built in classifiers/OrdinalDecomposition.py.
	"""

	# Data is just 6 separable points in the plane
	X = array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
	y = array([1, 1, 1, 2, 2, 2])


	def test_ordinal_decomposition(self):
		"""
		Check if this algorithm can correctly classify a toy problem.
		"""

		od = OrdinalDecomposition(dtype="ordered_partitions", decision_method="frank_hall",\
								base_classifier="sklearn.svm.SVC",\
								parameters={'C': 1.0, 'gamma': "scale", 'probability': True})


		y_pred = od.fit(self.X, self.y).predict(self.X)
		npt.assert_array_equal(y_pred, self.y)



	def test_coding_matrix(self):
		"""
		Checking if the coding matrix is built properly for
		each type of ordinal decomposition.
		"""

		od = OrdinalDecomposition()

		# Checking ordered_partitions (with a 5 class, 4 classifiers example)
		od.dtype = 'ordered_partitions'
		expected_cm = array([[-1,-1,-1,-1], 
							 [ 1,-1,-1,-1], 
							 [ 1, 1,-1,-1], 
							 [ 1, 1, 1,-1], 
							 [ 1, 1, 1, 1]])

		actual_cm = od._coding_matrix(od.dtype, 5)

		npt.assert_array_equal(actual_cm, expected_cm)


		# Checking one_vs_next
		od.dtype = 'one_vs_next'
		expected_cm = array([[-1, 0 ,0, 0],
							 [ 1,-1, 0, 0], 
							 [ 0, 1,-1, 0], 
							 [ 0, 0, 1,-1], 
							 [ 0, 0, 0, 1]])

		actual_cm = od._coding_matrix(od.dtype, 5)

		npt.assert_array_equal(actual_cm, expected_cm)


		# Checking one_vs_followers
		od.dtype = 'one_vs_followers'
		expected_cm = array([[-1, 0, 0, 0], 
							 [ 1,-1, 0, 0], 
							 [ 1, 1,-1, 0], 
							 [ 1, 1, 1,-1], 
							 [ 1, 1, 1, 1]])

		actual_cm = od._coding_matrix(od.dtype, 5)

		npt.assert_array_equal(actual_cm, expected_cm)


		# Checking one_vs_previous
		od.dtype = 'one_vs_previous'
		expected_cm = array([[ 1, 1, 1, 1], 
							 [ 1, 1, 1,-1], 
							 [ 1, 1,-1, 0], 
							 [ 1,-1, 0, 0], 
							 [-1, 0, 0, 0]])

		actual_cm = od._coding_matrix(od.dtype, 5)

		npt.assert_array_equal(actual_cm, expected_cm)



	def test_frank_hall_method(self):
		"""
		Check that frank and hall method returns expected values 
		for one toy problem (starting off predicted probabilities
		given by each binary classifier).
		"""


		# Checking frank_hall cannot be used whitout ordered_partitions
		od = OrdinalDecomposition(dtype="one_vs_next", decision_method="frank_hall")
		npt.assert_raises(AttributeError, od._frank_hall_method, self.X)


		od = OrdinalDecomposition(dtype="ordered_partitions")
		od.coding_matrix_ = od._coding_matrix(od.dtype, 5)

		# Predicted probabilities from a 5 class ordinal dataset (positive class)
		predictions = array(
							[[0.07495, 0.00003, 0.06861, 0.00005],
							[0.00017 , 0.	  , 0.03174, 0.00011],
							[0.99235 , 0.04285, 0.0485 , 0.00004],
							[0.95376 , 0.16388, 0.03857, 0.00028],
							[0.99726 , 0.20159, 0.61801, 0.00037],
							[1.		 , 0.90501, 0.44459, 0.00011],
							[1.		 , 0.97307, 0.99424, 0.14627],
							[1.		 , 0.64663, 0.45326, 0.06143],
							[1.		 , 0.83569, 0.9175 , 0.94988],
							[1.		 , 0.93172, 0.6774 , 0.43379]])



		actual_predicted_probabilities = od._frank_hall_method(predictions)
		expected_predicted_probabilities = array([
					[0.92505,	0.07492,	-0.06858,	 0.06856,	 0.00005],
					[0.99983,	0.00017,	-0.03174,	 0.03163,	 0.00011],
					[0.00765,	0.94950,	-0.00565,	 0.04846,	 0.00004],
					[0.04624,	0.78988,	 0.12531,	 0.03829,	 0.00028],
					[0.00274,	0.79567,	-0.41642,	 0.61764,	 0.00037],
					[0.,		0.09499,	 0.46042,	 0.44448,	 0.00011],
					[0.,		0.02693,	-0.02117,	 0.84797,	 0.14627],
					[0.,		0.35337,	 0.19337,	 0.39183,	 0.06143],
					[0.,		0.16431,	-0.08181,	-0.03238,	 0.94988],
					[0.,		0.06828,	 0.25432,	 0.24361,	 0.43379]])

		# Asserting similarity
		npt.assert_allclose(actual_predicted_probabilities, expected_predicted_probabilities, rtol=1e-04, atol=0)


	def test_exponential_loss_method(self):
		"""
		Check that exponential loss method returns expected values
		for one toy problem (starting off predicted probabilities
		given by each binary classifier).
		"""

		od = OrdinalDecomposition(dtype="ordered_partitions")
		od.coding_matrix_ = od._coding_matrix(od.dtype, 5)

		# Predicted probabilities from a 5 class ordinal dataset (positive class)
		predictions = array(
							[[0.07495, 0.00003, 0.06861, 0.00005],
							[0.00017 , 0.	  , 0.03174, 0.00011],
							[0.99235 , 0.04285, 0.0485 , 0.00004],
							[0.95376 , 0.16388, 0.03857, 0.00028],
							[0.99726 , 0.20159, 0.61801, 0.00037],
							[1.		 , 0.90501, 0.44459, 0.00011],
							[1.		 , 0.97307, 0.99424, 0.14627],
							[1.		 , 0.64663, 0.45326, 0.06143],
							[1.		 , 0.83569, 0.9175 , 0.94988],
							[1.		 , 0.93172, 0.6774 , 0.43379]])

		# Interpoling values from [0, 1] range to [-1, 1]
		predictions = (2*predictions) - 1



		actual_elosses = od._exponential_loss(predictions)
		expected_elosses = array(	[[1.5852 , 3.49769, 5.8479 , 7.79566, 10.14575],
									 [1.49583, 3.84519, 6.19559, 8.35469, 10.70441],
									 [3.85107, 1.54761, 3.64184, 5.70348, 8.05364],
									 [3.7542 , 1.67955, 3.12761, 5.24671, 7.59538],
									 [4.88834, 2.55481, 3.82059, 3.34415, 5.69227],
									 [6.2293 , 3.87889, 2.07579, 2.29788, 4.64761],
									 [8.47407, 6.12367, 3.93616, 1.62115, 3.15709],
									 [5.3858 , 3.0354 , 2.44043, 2.62767, 4.61571],
									 [9.43904, 7.08864, 5.64271, 3.77177, 1.71942],
									 [7.39145, 5.04105, 3.09146, 2.36688, 2.63249]])

		# Asserting similarity
		npt.assert_allclose(actual_elosses, expected_elosses, rtol=1e-04, atol=0)


	def test_logarithmic_loss_method(self):
		"""
		Check that exponential loss method returns expected values
		for one toy problem (starting off predicted probabilities
		given by each binary classifier).
		"""

		od = OrdinalDecomposition(dtype="ordered_partitions")
		od.coding_matrix_ = od._coding_matrix(od.dtype, 5)

		# Predicted probabilities from a 5 class ordinal dataset (positive class)
		predictions = array(
							[[0.07495, 0.00003, 0.06861, 0.00005],
							[0.00017 , 0.	  , 0.03174, 0.00011],
							[0.99235 , 0.04285, 0.0485 , 0.00004],
							[0.95376 , 0.16388, 0.03857, 0.00028],
							[0.99726 , 0.20159, 0.61801, 0.00037],
							[1.		 , 0.90501, 0.44459, 0.00011],
							[1.		 , 0.97307, 0.99424, 0.14627],
							[1.		 , 0.64663, 0.45326, 0.06143],
							[1.		 , 0.83569, 0.9175 , 0.94988],
							[1.		 , 0.93172, 0.6774 , 0.43379]])

		# Interpoling values from [0, 1] range to [-1, 1]
		predictions = (2*predictions) - 1


		actual_llosses = od._logarithmic_loss(predictions)
		expected_llosses = array(	[[0.58553, 2.28573, 4.28561, 6.01117, 8.01097],
									 [0.52385, 2.52317, 4.52317, 6.39621, 8.39577],
									 [2.52807, 0.55867, 2.38727, 4.19327, 6.19311],
									 [2.47122, 0.65618, 2.00066, 3.84638, 5.84526],
									 [3.46591, 1.47687, 2.67051, 2.19847, 4.19699],
									 [4.64297, 2.64297, 1.02293, 1.24457, 3.24413],
									 [6.48375, 4.48375, 2.59147, 0.61451, 2.02943],
									 [3.91936, 1.91936, 1.33284, 1.5198 , 3.27408],
									 [7.49674, 5.49674, 4.15398, 2.48398, 0.68446],
									 [5.69657, 3.69657, 1.96969, 1.26009, 1.52493]])

		# Asserting similarity
		npt.assert_allclose(actual_llosses, expected_llosses, rtol=1e-04, atol=0)


	def test_hinge_loss_method(self):
		"""
		Check that exponential loss method returns expected values
		for one toy problem (starting off predicted probabilities
		given by each binary classifier).
		"""

		od = OrdinalDecomposition(dtype="ordered_partitions")
		od.coding_matrix_ = od._coding_matrix(od.dtype, 5)

		# Predicted probabilities from a 5 class ordinal dataset (positive class)
		predictions = array(
							[[0.07495, 0.00003, 0.06861, 0.00005],
							[0.00017 , 0.	  , 0.03174, 0.00011],
							[0.99235 , 0.04285, 0.0485 , 0.00004],
							[0.95376 , 0.16388, 0.03857, 0.00028],
							[0.99726 , 0.20159, 0.61801, 0.00037],
							[1.		 , 0.90501, 0.44459, 0.00011],
							[1.		 , 0.97307, 0.99424, 0.14627],
							[1.		 , 0.64663, 0.45326, 0.06143],
							[1.		 , 0.83569, 0.9175 , 0.94988],
							[1.		 , 0.93172, 0.6774 , 0.43379]])

		# Interpoling values from [0, 1] range to [-1, 1]
		predictions = (2*predictions) - 1


		actual_hlosses = od._hinge_loss(predictions)
		expected_hlosses = array(	[[0.28728, 1.98748, 3.98736, 5.71292, 7.71272],
									 [0.06404, 2.06336, 4.06336, 5.9364 , 7.93596],
									 [2.16748, 0.19808, 2.02668, 3.83268, 5.83252],
									 [2.31298, 0.49794, 1.84242, 3.68814, 5.68702],
									 [3.63446, 1.64542, 2.83906, 2.36702, 4.36554],
									 [4.69942, 2.69942, 1.07938, 1.30102, 3.30058],
									 [6.22716, 4.22716, 2.33488, 0.35792, 1.77284],
									 [4.32264, 2.32264, 1.73612, 1.92308, 3.67736],
									 [7.40614, 5.40614, 4.06338, 2.39338, 0.59386],
									 [6.08582, 4.08582, 2.35894, 1.64934, 1.91418]])

		# Asserting similarity
		npt.assert_allclose(actual_hlosses, expected_hlosses, rtol=1e-04, atol=0)




if __name__ == '__main__':
	unittest.main()
