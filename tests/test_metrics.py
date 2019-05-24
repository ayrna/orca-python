
from sys import path

import unittest

from numpy import array
import numpy.testing as npt

path.append('..')

import metrics

class TestMetrics(unittest.TestCase):
	"""
	Class testing that every metric used to compute
	classifiers eficiency is correctly implemented.

	For this, a toy example will be used to compare
	the values given by the implemented functions against
	the previously calculated expected values for each metric.
	"""

	real_y = array([1,2,3,1,2,3,1,2,3,1,1,1,1,2,2,2,3,3,3,3])
	predicted_y = array([1,3,3,1,2,3,1,2,2,1,3,1,1,2,2,2,3,3,1,3])


	def test_ccr(self):

		real_ccr = 0.8000
		predicted_ccr = metrics.ccr(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_ccr, predicted_ccr, decimal=4)


	def test_amae(self):

		real_amae = 0.2937
		predicted_amae = metrics.amae(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_amae, predicted_amae, decimal=4)

	def test_gm(self):

		real_gm = 0.7991
		predicted_gm = metrics.gm(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_gm, predicted_gm, decimal=4)

	def test_mae(self):

		real_mae = 0.3000
		predicted_mae = metrics.mae(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_mae, predicted_mae, decimal=4)

	def test_mmae(self):

		real_mmae = 0.4286
		predicted_mmae = metrics.mmae(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_mmae, predicted_mmae, decimal=4)

	def test_ms(self):

		real_ms = 0.7143
		predicted_ms = metrics.ms(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_ms, predicted_ms, decimal=4)

	def test_mze(self):

		real_mze = 0.2000
		predicted_mze = metrics.mze(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_mze, predicted_mze, decimal=4)

	def test_tkendall(self):

		real_tkendall = 0.6240
		predicted_tkendall = metrics.tkendall(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_tkendall, predicted_tkendall, decimal=4)

	def test_wkappa(self):

		real_wkappa = 0.6703
		predicted_wkappa = metrics.wkappa(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_wkappa, predicted_wkappa, decimal=4)

	def test_spearman(self):

		real_spearman = 0.6429
		predicted_spearman = metrics.spearman(self.real_y, self.predicted_y)
		npt.assert_almost_equal(real_spearman, predicted_spearman, decimal=4)


if __name__ == '__main__':
	unittest.main()
