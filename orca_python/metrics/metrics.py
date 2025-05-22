from __future__ import division

import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
import scipy.stats

def greater_is_better(metric_name):
	"""
	Determines if greater values for one metric represent a better
	classification rate or vice versa. Needed when declaring a
	new scorer through make_scorer from sklearn.
	"""

	greater_is_better_metrics = ["ccr", "ms", "gm", "tkendall", "wkappa", "spearman"]
	if metric_name in greater_is_better_metrics:
		return True
	else:
		return False


def ccr(y, ypred):
	"""
	CCR - Correctly Classified Ratio

	Also named Accuracy, it's the percentage of well
	classified patterns among all patterns from a set.
	"""

	return np.count_nonzero(y == ypred) / float( len(y) )

def amae(y, ypred):
	"""
	AMAE - Average MAE

	Mean of the MAE metric among classes.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		cm = confusion_matrix(y, ypred)
		n_class = cm.shape[0]
		costs = np.reshape(np.tile(range(n_class),n_class),(n_class,n_class))
		costs = np.abs(costs - np.transpose(costs))
		errores = costs*cm
		amaes = np.sum(errores,axis=1)/np.sum(cm,axis=1).astype('double')
		amaes = amaes[~np.isnan(amaes)]
		return np.mean(amaes)

def gm(y, ypred):
	"""

	GM - Geometric Mean

	Geometric mean of the sensitivy (accuracy) for each class
	
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		cm = confusion_matrix(y, ypred)
		sum_byclass = np.sum(cm,axis=1)
		sensitivities = np.diag(cm)/sum_byclass.astype('double')
		sensitivities[sum_byclass==0] = 1
		gm_result = pow(np.prod(sensitivities),1.0/cm.shape[0])
		return gm_result

def mae(y, ypred):
	"""
	MAE - Mean Absolute Error

	Average absolute deviation of the predicted class
	from the actual true class.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		y = np.asarray(y)
		ypred = np.asarray(ypred)
		return abs(y - ypred).sum() / len(y)

def mmae(y, ypred):
	"""
	MMAE - Maximum MAE

	MAE value of the class with higher distance from the
	true values to the predicted ones.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		cm = confusion_matrix(y, ypred)
		n_class = cm.shape[0]
		costes=np.reshape(np.tile(range(n_class),n_class),(n_class,n_class))
		costes = np.abs(costes - np.transpose(costes))
		errores = costes*cm
		amaes = np.sum(errores,axis=1)/np.sum(cm,axis=1).astype('double')
		amaes = amaes[~np.isnan(amaes)]
		return amaes.max()

def ms(y, ypred):
	"""
	MS - Minimum Sensitivity

	Lowest percentage of patterns correctly predicted as
	belonging to each class, with respect to the total number
	of examples in the corresponding class.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		cm = confusion_matrix(y, ypred)
		sum_byclass = np.sum(cm,axis=1)
		sensitivities = np.diag(cm)/sum_byclass.astype('double')
		sensitivities[sum_byclass==0] = 1
		ms = np.min(sensitivities)

		return ms

def mze(y, ypred):
	"""
	MZE - Mean Zero-one Error

	Better known as error rate, is the complementary measure of CCR.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		confusion = confusion_matrix(y, ypred)
		return 1 - np.diagonal(confusion).sum() / confusion.sum()

def tkendall(y, ypred):
	"""
	The Kendalls t is a statistic used to measure
	the association between two measured quantities.
	It is a measure of rank correlation.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		corr, pvalue = scipy.stats.kendalltau(y, ypred)
		return corr

def wkappa(y, ypred):
	"""
	The Weighted Kappa is a modified version of the Kappa
	statistic calculated to allow as signing different weights
	to different levels of aggregation between two variables.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		cm = confusion_matrix(y, ypred)
		n_class = cm.shape[0]
		costes=np.reshape(np.tile(range(n_class),n_class),(n_class,n_class))
		costes = np.abs(costes - np.transpose(costes))
		f = 1 - costes

		n = cm.sum()
		x = cm/n

		r = x.sum(axis=1) # Row sum
		s = x.sum(axis=0) # Col sum
		Ex = r.reshape(-1, 1) * s
		po = (x * f).sum()
		pe = (Ex * f).sum()
		return (po - pe) / (1 - pe)

def spearman(y, ypred):
	"""
	The Spearmans rank correlation coefficient is
	a non-parametric measure of statistical dependence
	between two variables.
	"""

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")

		n = len(y)
		num = ((y - np.repeat(np.mean(y), n)) * (ypred - np.repeat(np.mean(ypred), n))).sum()
		div = np.sqrt((pow(y - np.repeat(np.mean(y), n), 2)).sum()
						* (pow(ypred - np.repeat(np.mean(ypred), n), 2)).sum())

		if num == 0:
			return 0
		else:
			return num / div
