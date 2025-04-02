from sys import path as syspath
from os import path as ospath

import pytest
import numpy as np
import numpy.testing as npt

syspath.append(ospath.join('..', 'classifiers'))

# from SVOREX import SVOREX
from orca_python.classifiers.SVOREX import SVOREX


@pytest.fixture
def dataset_path():
    return ospath.join(ospath.dirname(ospath.abspath(__file__)), "test_datasets", "test_svorex_dataset")

@pytest.fixture
def train_file(dataset_path):
    return np.loadtxt(ospath.join(dataset_path,"train.0"))

@pytest.fixture
def test_file(dataset_path):
    return np.loadtxt(ospath.join(dataset_path,"test.0"))

def test_svorex_fit_correct(dataset_path, train_file, test_file):
    #Check if this algorithm can correctly classify a toy problem.
    
    #Test preparation
    X_train = train_file[:,0:(-1)]
    y_train = train_file[:,(-1)]

    X_test = test_file[:,0:(-1)]

    expected_predictions = [ospath.join(dataset_path,"expectedPredictions.0"), 
                            ospath.join(dataset_path,"expectedPredictions.1"),
                            ospath.join(dataset_path,"expectedPredictions.2"),]

    classifiers = [SVOREX(kernel_type=0, t=0.002, c=0.5, k=0.1),
                   SVOREX(kernel_type=1, t=0.002, c=0.5, k=0.1),
                   SVOREX(kernel_type=2, p=4, t=0.002, c=0.5, k=0.1)]

    #Test execution and verification
    for expected_prediction, classifier in zip(expected_predictions, classifiers):
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        expected_prediction = np.loadtxt(expected_prediction)
        npt.assert_equal(predictions, expected_prediction, "The prediction doesnt match with the desired values")

def test_svorex_fit_not_valid_parameter(train_file):

    #Test preparation
    X_train = train_file[:,0:(-1)]
    y_train = train_file[:,(-1)]

    classifiers = [SVOREX(c=0.1, k=1, t=0),
                SVOREX(c=0, k=1),
                SVOREX(c=0.1, k=0),
                SVOREX(kernel_type=2, p=0, c=0.1, k=1),
                SVOREX(kernel_type=0, c=0.1, k=-1)]
    
    error_msgs = ["- T is invalid",
                "- C is invalid",
                "- K is invalid",
                "- P is invalid",
                "-1 is invalid"]

    #Test execution and verification
    for classifier, error_msg in zip(classifiers, error_msgs):
        with pytest.raises(ValueError, match=error_msg):
            model = classifier.fit(X_train, y_train)
            assert model is None, "The SVOREX fit method doesnt return Null on error"

def test_svorex_fit_not_valid_data(train_file):
    #Test preparation
    X_train = train_file[:,0:(-1)]
    y_train = train_file[:,(-1)]
    X_train_broken = train_file[0:(-1),0:(-2)]
    y_train_broken = train_file[0:(-1),(-1)]

    #Test execution and verification
    classifier = SVOREX(k=0.1, c=1)
    with pytest.raises(ValueError):
            model = classifier.fit(X_train, y_train_broken)
            assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
            model = classifier.fit([], y_train)
            assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
            model = classifier.fit(X_train, [])
            assert model is None, "The SVOREX fit method doesnt return Null on error"

    with pytest.raises(ValueError):
            model = classifier.fit(X_train_broken, y_train)
            assert model is None, "The SVOREX fit method doesnt return Null on error"

def test_svorex_model_is_not_a_dict(train_file, test_file):
    #Test preparation
    X_train = train_file[:,0:(-1)]
    y_train = train_file[:,(-1)]

    X_test = test_file[:,0:(-1)]

    classifier = SVOREX(k=0.1, c=1)
    classifier.fit(X_train, y_train)

    #Test execution and verification
    with pytest.raises(TypeError, match="Model should be a dictionary!"):
            classifier.classifier_ = 1
            classifier.predict(X_test)

def test_svorex_predict_not_valid_data(train_file):
    #Test preparation
    X_train = train_file[:,0:(-1)]
    y_train = train_file[:,(-1)]

    classifier = SVOREX(k=0.1, c=1)
    classifier.fit(X_train, y_train)

    #Test execution and verification
    with pytest.raises(ValueError):
        classifier.predict([])
