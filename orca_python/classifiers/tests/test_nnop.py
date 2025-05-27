from sys import path as syspath
from os import path as ospath

import pytest
import numpy as np
import numpy.testing as npt

# syspath.append(ospath.join('..', 'classifiers'))

# from NNOP import NNOP
from orca_python.classifiers.NNOP import NNOP
from orca_python.testing import TEST_DATASETS_DIR


@pytest.fixture
def dataset_path():
	return ospath.join(TEST_DATASETS_DIR, "balance-scale")

@pytest.fixture
def train_file(dataset_path):
	return np.loadtxt(ospath.join(dataset_path,"train_balance-scale.csv"), delimiter=",")

@pytest.fixture
def test_file(dataset_path):
	return np.loadtxt(ospath.join(dataset_path,"test_balance-scale.csv"), delimiter=",")

#	-----	NOT APPLIED	-----
# It doesn't apply to the because can't set seed to randomize model weights.
# def test_nnop_fit_correct(self):
# 	#Check if this algorithm can correctly classify a toy problem.
	
# 	#Test preparation
# 	X_train = self.train_file[:,0:(-1)]
# 	y_train = self.train_file[:,(-1)]

# 	X_test = self.test_file[:,0:(-1)]
	
	# expected_predictions = [ospath.join(self.dataset_path,"expectedPredictions.0")]
							# ospath.join(self.dataset_path,"expectedPredictions.1"),
							# ospath.join(self.dataset_path,"expectedPredictions.2"),
							# ospath.join(self.dataset_path,"expectedPredictions.3")]

# 	classifiers = [NNOP(epsilon_init = 0.5, hidden_n = 10, iterations = 500, lambda_value = 0.01)]
	#			   NNOP(epsilon_init = 0.5, hidden_n = 20, iterations = 500, lambda_value = 0.01),
	#			   NNOP(epsilon_init = 0.5, hidden_n = 10, iterations = 250, lambda_value = 0.01),
	#			   NNOP(epsilon_init = 0.5, hidden_n = 20, iterations = 500, lambda_value = 0.01)]


# 	#Test execution and verification
# 	for expected_prediction, classifier in zip(expected_predictions, classifiers):
# 		classifier.fit(X_train, y_train)
# 		predictions = classifier.predict(X_test)
# 		expected_prediction = np.loadtxt(expected_prediction)
# 		npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")

def test_nnop_fit_not_valid_parameter(train_file):

	#Test preparation
	X_train = train_file[:,0:(-1)]
	y_train = train_file[:,(-1)]

	classifiers = [NNOP(epsilon_init=0.5, hidden_n=-1, iterations=1000, lambda_value=0.01),
					NNOP(epsilon_init=0.5, hidden_n=10, iterations=-1, lambda_value=0.01)]

	#Test execution and verification
	for classifier in classifiers:
			model = classifier.fit(X_train, y_train)
			assert model is None, "The NNOP fit method doesnt return Null on error"

def test_nnop_fit_not_valid_data(train_file):
	#Test preparation
	X_train = train_file[:,0:(-1)]
	y_train = train_file[:,(-1)]
	X_train_broken = train_file[0:(-1),0:(-2)]
	y_train_broken = train_file[0:(-1),(-1)]

	#Test execution and verification
	classifier = NNOP(epsilon_init=0.5, hidden_n=10, iterations=1000, lambda_value=0.01)
	with pytest.raises(ValueError):
			model = classifier.fit(X_train, y_train_broken)
			assert model is None, "The NNOP fit method doesnt return Null on error"

	with pytest.raises(ValueError):
			model = classifier.fit([], y_train)
			assert model is None, "The NNOP fit method doesnt return Null on error"

	with pytest.raises(ValueError):
			model = classifier.fit(X_train, [])
			assert model is None, "The NNOP fit method doesnt return Null on error"

	with pytest.raises(ValueError):
			model = classifier.fit(X_train_broken, y_train)
			assert model is None, "The NNOP fit method doesnt return Null on error"


#	-----	NOT APPLIED	-----
# It doesn't apply to the because it has no internal model
# like in other classifiers like REDSVM or SVOREX. 
# def test_nnop_model_is_not_a_dict(self):
# 	#Test preparation
# 	X_train = self.train_file[:,0:(-1)]
# 	y_train = self.train_file[:,(-1)]

# 	X_test = self.test_file[:,0:(-1)]

# 	classifier = NNOP(epsilon_init = 0.5, hidden_n = 10, iterations = 500, lambda_value = 0.01)
# 	classifier.fit(X_train, y_train)

# 	#Test execution and verification
# 	with self.assertRaisesRegex(TypeError, "Model should be a dictionary!"):
# 			classifier.classifier_ = 1
# 			classifier.predict(X_test)


def test_nnop_predict_not_valid_data(train_file):
	#Test preparation
	X_train = train_file[:,0:(-1)]
	y_train = train_file[:,(-1)]

	classifier = NNOP(epsilon_init = 0.5, hidden_n = 10, iterations = 500, lambda_value = 0.01)
	classifier.fit(X_train, y_train)

	#Test execution and verification
	with pytest.raises(ValueError):
		classifier.predict([])
