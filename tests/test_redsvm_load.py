from sys import path as syspath
from os import path as ospath

import unittest

import numpy as np
from sklearn.model_selection import GridSearchCV
import ntpath

syspath.append(ospath.join('..', 'classifiers'))

from REDSVM import REDSVM


class TestRedsvmLoad(unittest.TestCase):
	"""
	Class testing REDSVM's functionality.

	This classifier is built in classifiers/REDSVM.py.
	"""

	dataset_path = ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_redsvm_svorex_load_dataset")
		
	def test_redsvm_load(self):

		print("\n")
		print("++++++++++++++++")
		print("REDSVM load test")
		print("++++++++++++++++")
		print()

		datasets_names = [ospath.join(self.dataset_path,"train_automobile.0"),
						ospath.join(self.dataset_path,"train_balance-scale.0"),
						ospath.join(self.dataset_path,"train_bondrate.0"),
						ospath.join(self.dataset_path,"train_car.0"),
						ospath.join(self.dataset_path,"train_contact-lenses.0"),
						ospath.join(self.dataset_path,"train_ERA.0"),
						ospath.join(self.dataset_path,"train_ESL.0"),
						ospath.join(self.dataset_path,"train_eucalyptus.0"),
						ospath.join(self.dataset_path,"train_LEV.0"),
						ospath.join(self.dataset_path,"train_newthyroid.0"),
						ospath.join(self.dataset_path,"train_pasture.0"),
						ospath.join(self.dataset_path,"train_squash-stored.0"),
						ospath.join(self.dataset_path,"train_squash-unstored.0"),
						ospath.join(self.dataset_path,"train_SWD.0"),
						ospath.join(self.dataset_path,"train_tae.0"),
						ospath.join(self.dataset_path,"train_toy.0"),
						ospath.join(self.dataset_path,"train_winequality-red.0")]

		classifiers = [REDSVM(t=0, d=2),
					REDSVM(t=1, d=2), 
					REDSVM(t=2, d=2), 
					REDSVM(t=3, d=2), 
					REDSVM(t=4, d=2), 
					REDSVM(t=5, d=2), 
					REDSVM(t=6, d=2), 
					REDSVM(t=7, d=2)]

		parameters = {'c': np.full(7,10.)**np.arange(-3,4), 'g': np.full(7,10.)**np.arange(-3,4)}
		
		for dataset_name in datasets_names:
			dataset = np.loadtxt(dataset_name)

			X_train = dataset[:,0:(-1)]
			y_train = dataset[:,(-1)]
			
			print("-------------")
			print("Dataset {}...".format(ntpath.basename(dataset_name)))

			for classifier in classifiers:
				grid = GridSearchCV(classifier, parameters, n_jobs=-1, cv=3)
				grid.fit(X_train, y_train)

			print("Done!")


if __name__ == '__main__':
	unittest.main()