import os
from os import path as ospath
from shutil import rmtree

import numpy.testing as npt
import pandas as pd
import numpy as np
import pytest

# syspath.append('..')
# syspath.append(ospath.join('..', 'classifiers'))

# from utilities import Utilities
# from utilities import load_classifier
from orca_python.utilities import Utilities, load_classifier
from orca_python.testing import TEST_DATASETS_DIR


@pytest.fixture
def util():
    general_conf = {}
    configurations = {}
    return Utilities(general_conf, configurations)


def create_csv(path, filename):
    """Create a csv file with sample data."""
    sample_data = "1,2,3,0\n4,5,6,1"
    (path / filename).write_text(sample_data)


def test_load_complete_dataset(tmp_path, util):
    """Loading dataset composed of 5 partitions, each one of them composed of
    a train and test file.

    """
    dataset_path = tmp_path / "complete"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_complete.{i}")
        create_csv(dataset_path, f"test_complete.{i}")

    partition_list = util._load_dataset(dataset_path)

    # Check all partitions have been loaded
    npt.assert_equal(
        len(partition_list), (len([name for name in os.listdir(dataset_path)]) / 2)
    )
    # Check if every partition has train and test inputs and outputs (4 diferent dictionaries)
    npt.assert_equal(
        all([len(partition[1]) == 4 for partition in partition_list]), True
    )


def test_load_partitionless_dataset(tmp_path, util):
    """Loading dataset composed of only two csv files (train and test files)"""
    dataset_path = tmp_path / "partitionless"
    dataset_path.mkdir()

    create_csv(dataset_path, "train_partitionless.csv")
    create_csv(dataset_path, "test_partitionless.csv")

    partition_list = util._load_dataset(dataset_path)

    npt.assert_equal(len(partition_list), 1)
    npt.assert_equal(
        all([len(partition[1]) == 4 for partition in partition_list]), True
    )


def test_load_nontestfile_dataset(tmp_path, util):
    """Loading dataset composed of five train files."""
    dataset_path = tmp_path / "nontestfile"
    dataset_path.mkdir()

    for i in range(5):
        create_csv(dataset_path, f"train_nontestfile.{i}")

    partition_list = util._load_dataset(dataset_path)

    npt.assert_equal(
        len(partition_list), len([name for name in os.listdir(dataset_path)])
    )
    npt.assert_equal(
        all([len(partition[1]) == 2 for partition in partition_list]), True
    )


def test_load_nontrainfile_dataset(tmp_path, util):
    """Loading dataset with 2 partitions, one of them lacking its train file.
    This should raise an exception.

    """
    dataset_path = tmp_path / "nontrainfile"
    dataset_path.mkdir()

    for i in range(2):
        create_csv(dataset_path, f"test_nontrainfile.{i}")

    with pytest.raises(RuntimeError):
        util._load_dataset(dataset_path)


def test_normalize_data(tmp_path, util):
    # Test preparation
    dataset_path = ospath.join(TEST_DATASETS_DIR, "balance-scale")

    train_file = np.loadtxt(
        ospath.join(dataset_path, "train_balance-scale.csv"), delimiter=","
    )
    X_train = train_file[:, 0:(-1)]

    test_file = np.loadtxt(
        ospath.join(dataset_path, "test_balance-scale.csv"), delimiter=","
    )
    X_test = test_file[:, 0:(-1)]

    # Test execution
    norm_X_train, _ = util._normalize_data(X_train, X_test)

    # Test verification
    result = (norm_X_train >= 0).all() and (norm_X_train <= 1).all()
    npt.assert_equal(result, True)


def test_standardize_data(util):
    # Test preparation
    dataset_path = ospath.join(TEST_DATASETS_DIR, "balance-scale")

    train_file = np.loadtxt(
        ospath.join(dataset_path, "train_balance-scale.csv"), delimiter=","
    )
    X_train = train_file[:, 0:(-1)]

    test_file = np.loadtxt(
        ospath.join(dataset_path, "test_balance-scale.csv"), delimiter=","
    )
    X_test = test_file[:, 0:(-1)]

    # Test execution
    std_X_train, _ = util._standardize_data(X_train, X_test)

    # Test verification
    npt.assert_almost_equal(np.mean(std_X_train), 0)
    npt.assert_almost_equal(np.std(std_X_train), 1)


def test_load_algorithm():
    # Loading a method from within this framework
    from orca_python.classifiers import OrdinalDecomposition

    imported_class = load_classifier("orca_python.classifiers.OrdinalDecomposition")
    npt.assert_equal(imported_class, OrdinalDecomposition)

    # Loading a scikit-learn classifier
    from sklearn.svm import SVC

    imported_class = load_classifier("sklearn.svm.SVC")
    npt.assert_equal(imported_class, SVC)

    # Raising exceptions when the classifier cannot be loaded
    with pytest.raises(ImportError):
        load_classifier("sklearn.svm.SVC.submethod")
    with pytest.raises(AttributeError):
        load_classifier("sklearn.svm.SVCC")


def test_check_params(util):
    """Testing functionality of check_params method.

    It will test the 3 different scenarios contemplated within the framework
    for passing the configuration.

    """
    # Normal use of configuration file with a non nested method
    util.configurations = {
        "conf1": {
            "classifier": "sklearn.svm.SVC",
            "parameters": {
                "C": [0.1, 1, 10],
                "gamma": [0.1, 1, 100],
                "probability": "True",
            },
        }
    }

    # Getting formatted_params and expected_params
    util._check_params()
    formatted_params = util.configurations["conf1"]["parameters"]

    random_state = util.configurations["conf1"]["parameters"]["random_state"]
    expected_params = {
        "C": [0.1, 1, 10],
        "gamma": [0.1, 1, 100],
        "probability": ["True"],
        "random_state": random_state,
    }

    npt.assert_equal(formatted_params, expected_params)

    # Configuration file using an ensemble method
    util.configurations = {
        "conf2": {
            "classifier": "orca_python.classifiers.OrdinalDecomposition",
            "parameters": {
                "dtype": "OrderedPartitions",
                "base_classifier": "sklearn.svm.SVC",
                "parameters": {"C": [1, 10], "gamma": [1, 10], "probability": ["True"]},
            },
        }
    }

    # Getting formatted_params and expected_params
    util._check_params()
    formatted_params = util.configurations["conf2"]["parameters"]

    random_state = util.configurations["conf2"]["parameters"]["parameters"][0][
        "random_state"
    ]
    expected_params = {
        "dtype": ["OrderedPartitions"],
        "base_classifier": ["sklearn.svm.SVC"],
        "parameters": [
            {"C": 1, "gamma": 1, "probability": True, "random_state": random_state},
            {"C": 1, "gamma": 10, "probability": True, "random_state": random_state},
            {"C": 10, "gamma": 1, "probability": True, "random_state": random_state},
            {"C": 10, "gamma": 10, "probability": True, "random_state": random_state},
        ],
    }

    # Ordering list of parameters from formatted_params to prevent inconsistencies
    formatted_params["parameters"] = sorted(
        formatted_params["parameters"], key=lambda k: k["C"]
    )

    npt.assert_equal(expected_params, formatted_params)

    # Configuration file where it's not necessary to perform cross-validation
    util.configurations = {
        "conf3": {
            "classifier": "orca_python.classifiers.OrdinalDecomposition",
            "parameters": {
                "dtype": "OrderedPartitions",
                "base_classifier": "sklearn.svm.SVC",
                "parameters": {"C": [1], "gamma": [1]},
            },
        }
    }

    # Getting formatted_params and expected_params
    util._check_params()
    formatted_params = util.configurations["conf3"]["parameters"]

    random_state = util.configurations["conf3"]["parameters"]["parameters"][
        "random_state"
    ]
    expected_params = {
        "dtype": "OrderedPartitions",
        "base_classifier": "sklearn.svm.SVC",
        "parameters": {"C": 1, "gamma": 1, "random_state": random_state},
    }

    npt.assert_equal(formatted_params, expected_params)

    # Resetting configurations to not interfere with other experiments
    util.configurations = {}


@pytest.fixture
def main_folder():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def dataset_folder(main_folder):
    return os.path.join(main_folder, "datasets", "data")


@pytest.fixture
def general_conf(dataset_folder):
    return {
        "basedir": dataset_folder,
        "datasets": ["tae", "contact-lenses"],
        "input_preprocessing": "std",
        "hyperparam_cv_nfolds": 3,
        "jobs": 10,
        "output_folder": "my_runs/",
        "metrics": ["ccr", "mae", "amae", "mze"],
        "cv_metric": "mae",
    }


@pytest.fixture
def configurations():
    return {
        "SVM": {
            "classifier": "sklearn.svm.SVC",
            "parameters": {"C": [0.001, 0.1, 1, 10, 100], "gamma": [0.1, 1, 10]},
        },
        "SVMOP": {
            "classifier": "orca_python.classifiers.OrdinalDecomposition",
            "parameters": {
                "dtype": "ordered_partitions",
                "decision_method": "frank_hall",
                "base_classifier": "sklearn.svm.SVC",
                "parameters": {
                    "C": [0.01, 0.1, 1, 10],
                    "gamma": [0.01, 0.1, 1, 10],
                    "probability": ["True"],
                },
            },
        },
    }


def test_run_experiment(main_folder, general_conf, configurations):
    """To test the main method, a configuration will be run until the end.
    Next we will check that every expected result file has been created,
    having all of them the proper dimensions and types.

    """
    # Declaring Utilities object and running the experiment
    util = Utilities(general_conf, configurations, verbose=False)
    util.run_experiment()
    # Saving results information
    util.write_report()

    # Checking if all outputs have been generated and are correct
    outputs_folder = "my_runs"
    npt.assert_equal(os.path.exists(outputs_folder), True)

    experiment_folder = sorted(os.listdir(outputs_folder))
    experiment_folder = os.path.join(outputs_folder, experiment_folder[-1])

    for dataset in util.general_conf["datasets"]:
        for conf_name, _ in util.configurations.items():

            # Check if the folder for that dataset-configurations exists
            conf_folder = os.path.join(experiment_folder, (dataset + "-" + conf_name))
            npt.assert_equal(os.path.exists(conf_folder), True)

            # Checking CSV containning all metrics for that configuration
            metrics_csv = pd.read_csv(
                os.path.join(conf_folder, (dataset + "-" + conf_name + ".csv"))
            )
            metrics_csv = metrics_csv.iloc[:, -12:]

            npt.assert_equal(metrics_csv.shape, (30, 12))
            npt.assert_equal(all(str(c) == "float64" for c in metrics_csv.dtypes), True)

            # Checking that all models have been saved
            models_folder = os.path.join(conf_folder, "models")
            npt.assert_equal(os.path.exists(models_folder), True)
            npt.assert_equal(len(os.listdir(models_folder)), 30)

            # Checking that all predictions have been saved
            predictions_folder = os.path.join(conf_folder, "predictions")
            npt.assert_equal(os.path.exists(predictions_folder), True)
            npt.assert_equal(len(os.listdir(predictions_folder)), 60)

    # Checking if summaries are correct
    train_summary = pd.read_csv(os.path.join(experiment_folder, "train_summary.csv"))
    npt.assert_equal(train_summary.shape, (4, 13))
    npt.assert_equal(
        all(str(c) == "float64" for c in train_summary.dtypes.iloc[1:]), True
    )

    test_summary = pd.read_csv(os.path.join(experiment_folder, "test_summary.csv"))
    npt.assert_equal(test_summary.shape, (4, 13))
    npt.assert_equal(
        all(str(c) == "float64" for c in test_summary.dtypes.iloc[1:]), True
    )

    rmtree(outputs_folder)
