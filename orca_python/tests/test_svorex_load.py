from sys import path as syspath
from os import path as ospath
import ntpath
from shutil import rmtree
import gc

import pytest

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

# syspath.append('..')
# syspath.append(ospath.join('..', 'classifiers'))

# from utilities import Utilities
from orca_python.utilities import Utilities
from orca_python.testing import TEST_DATASETS_DIR


@pytest.fixture
def dataset_path():
    return ospath.join(TEST_DATASETS_DIR)

@pytest.fixture
def values():
    return np.logspace(-3, 3, 7).tolist()

@pytest.fixture
def general_conf(dataset_path):
    return {"basedir": dataset_path,
            "datasets": ["balance-scale"],
            "input_preprocessing": "std",
            "hyperparam_cv_nfolds": 3,
            "jobs": 10,
            "output_folder": "my_runs/",
            "metrics": ["ccr", "mae", "amae", "mze"],
            "cv_metric": "mae"}

@pytest.fixture
def configurations(values):
    return {
        "svorex_gaussian": {
            "classifier": "orca_python.classifiers.SVOREX",
            "parameters": {
                "kernel_type": 0,
                "c": values,
                "k": values
            }
        }
    }

def test_redsvm_load(general_conf, configurations):
    gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)
    
    print("\n")
    print("###############################")
    print("SVOREX load test")
    print("###############################")

    # Declaring Utilities object and running the experiment
    util = Utilities(general_conf, configurations, verbose=True)
    util.run_experiment()
    # Saving results information
    util.write_report()

    #Delete all the test results after load test
    rmtree("my_runs")
