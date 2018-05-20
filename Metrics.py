

import numpy as np


def CCR(real_y, predicted_y):

	"""

	"""

	if(len(real_y) != len(predicted_y)):
		print "Real and Predicted outputs lists have different sizes"

	return np.count_nonzero(real_y == predicted_y) / float( len(real_y) )


def MAE(real_y, predicted_y):
	"""

	"""

	if(len(real_y) != len(predicted_y)):
		print "Real and Predicted outputs lists have different sizes"

	return np.count_nonzero(real_y == predicted_y) / float( len(real_y) )

