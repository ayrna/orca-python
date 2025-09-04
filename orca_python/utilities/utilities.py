"""Utility class for running experiments."""

from __future__ import print_function

from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from sys import path as syspath
from time import time

import numpy as np
import pandas as pd
from pkg_resources import get_distribution, parse_version
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from orca_python.metrics import compute_metric, load_metric_as_scorer
from orca_python.model_selection import (
    get_classifier_by_name,
    is_searchcv,
    prepare_param_grid,
)
from orca_python.results import Results


class Utilities:
    """Run experiments over N datasets with M different configurations.

    Configurations are composed of a classifier method and different parameters, where
    it may be multiple values for every one of them.

    Running the main function of this class will perform cross-validation for each
    partition per dataset-configuration pairs, obtaining the most optimal model, after
    what will be used to infer the labels for the test sets.

    Parameters
    ----------
    general_conf : dict
        Dictionary containing values needed to run the experiment. It gives this class
        information about where are located the different datasets, which one are going
        to be tested, the metrics to use, etc.

    configurations : dict
        Dictionary in which are stated the different classifiers to build methods upon
        the selected datasets, as well as the different values for the hyper-parameters
        used to optimize the model during cross-validation phase.

    verbose : bool
        Variable used for testing purposes. Silences all prints.

    Attributes
    ----------
    _results : Results
        Class used to manage and store all information obtained during the run of an
        experiment.

    """

    def __init__(self, general_conf, configurations, verbose=True):
        self.general_conf = deepcopy(general_conf)
        self.configurations = deepcopy(configurations)
        self.verbose = verbose

        syspath.append("classifiers")

    def _resolve_estimator(self, identifier):
        """Resolve and return a classifier given its identifier."""
        if not isinstance(identifier, str):
            return identifier

        try:
            return get_classifier_by_name(identifier)
        except Exception:
            pass

        if "." in identifier:
            module_path, class_name = identifier.rsplit(".", 1)
            return getattr(import_module(module_path), class_name)
        raise ValueError(f"Unknown classifier identifier: {identifier}")

    def run_experiment(self):
        """Run an experiment. Main method of this framework.

        Loads all datasets, which can be fragmented in partitions. Builds a model per
        partition, using cross-validation to find the optimal values among the
        hyper-parameters to compare from.

        Uses the built model to get train and test metrics, storing all the information
        into a Results object.

        Raises
        ------
        ValueError
            If the dataset list is inconsistent.

        AttributeError
            If the input preprocessing is unknown.

        TypeError
            If the parameters for base_classifier must be list.

        """
        self._results = Results(self.general_conf["output_folder"])

        self._check_dataset_list()
        self._check_params()

        if self.verbose:
            print("\n###############################")
            print("\tRunning Experiment")
            print("###############################")

        # Iterating over Datasets
        for x in self.general_conf["datasets"]:

            dataset_name = x.strip()
            dataset_path = Path(self.general_conf["basedir"]) / dataset_name

            dataset = self._load_dataset(dataset_path)

            if self.verbose:
                print("\nRunning", dataset_name, "dataset")
                print("--------------------------")

            # Iterating over Configurations
            for conf_name, configuration in self.configurations.items():

                if self.verbose:
                    print("Running", conf_name, "...")

                classifier = self._resolve_estimator(configuration["classifier"])

                # Iterating over partitions
                for part_idx, partition in dataset:

                    if self.verbose:
                        print("  Running Partition", part_idx)

                    # Normalization or Standardization of the partition if requested
                    if (
                        self.general_conf["input_preprocessing"].strip().lower()
                        == "norm"
                    ):
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._normalize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
                        )
                    elif (
                        self.general_conf["input_preprocessing"].strip().lower()
                        == "std"
                    ):
                        partition["train_inputs"], partition["test_inputs"] = (
                            self._standardize_data(
                                partition["train_inputs"], partition["test_inputs"]
                            )
                        )

                    elif self.general_conf["input_preprocessing"].strip().lower() != "":
                        raise AttributeError(
                            "Input preprocessing named '%s' unknown"
                            % self.general_conf["input_preprocessing"].strip().lower()
                        )

                    optimal_estimator = self._get_optimal_estimator(
                        partition["train_inputs"],
                        partition["train_outputs"],
                        classifier,
                        configuration["parameters"],
                    )

                    # Getting train and test predictions
                    train_predicted_y = optimal_estimator.predict(
                        partition["train_inputs"]
                    )

                    test_predicted_y = None
                    elapsed = np.nan
                    if "test_outputs" in partition:
                        start = time()
                        test_predicted_y = optimal_estimator.predict(
                            partition["test_inputs"]
                        )
                        elapsed = time() - start

                    # Obtaining train and test metrics values.
                    train_metrics = OrderedDict()
                    test_metrics = OrderedDict()
                    for metric_name in self.general_conf["metrics"]:
                        # Get train scores
                        train_score = compute_metric(
                            metric_name,
                            partition["train_outputs"],
                            train_predicted_y,
                        )
                        train_metrics[metric_name.strip() + "_train"] = train_score

                        # Get test scores
                        test_metrics[metric_name.strip() + "_test"] = np.nan
                        if "test_outputs" in partition:
                            test_score = compute_metric(
                                metric_name, partition["test_outputs"], test_predicted_y
                            )
                            test_metrics[metric_name.strip() + "_test"] = test_score

                    # Cross-validation was performed to tune hyper-parameters
                    if isinstance(optimal_estimator, GridSearchCV):
                        train_metrics["cv_time_train"] = optimal_estimator.cv_results_[
                            "mean_fit_time"
                        ].mean()
                        test_metrics["cv_time_test"] = optimal_estimator.cv_results_[
                            "mean_score_time"
                        ].mean()
                        train_metrics["time_train"] = optimal_estimator.refit_time_
                        test_metrics["time_test"] = elapsed

                    else:
                        optimal_estimator.best_params_ = configuration["parameters"]
                        optimal_estimator.best_estimator_ = optimal_estimator

                        train_metrics["cv_time_train"] = np.nan
                        test_metrics["cv_time_test"] = np.nan
                        train_metrics["time_train"] = optimal_estimator.refit_time_
                        test_metrics["time_test"] = elapsed

                    # Saving the results for this partition
                    self._results.add_record(
                        part_idx,
                        optimal_estimator.best_params_,
                        optimal_estimator.best_estimator_,
                        {"dataset": dataset_name, "config": conf_name},
                        {"train": train_metrics, "test": test_metrics},
                        {"train": train_predicted_y, "test": test_predicted_y},
                    )

    def _load_dataset(self, dataset_path):
        """Load all dataset's files, divided into train and test.

        Parameters
        ----------
        dataset_path : Path
            Path to dataset folder.

        Returns
        -------
        partition_list : list of tuples
            List of partitions found inside a dataset folder. Each partition is stored
            into a dictionary, disjoining train and test inputs and outputs.

        Raises
        ------
        ValueError
            If the dataset path does not exist.

        RuntimeError
            If a partition is found without train files.

        """

        def get_partition_index(filename):
            # Extracts the index between the last "_" and ".csv"
            return filename.rsplit("_", 1)[-1].replace(".csv", "")

        try:
            partition_list = {
                get_partition_index(filename.name): {}
                for filename in dataset_path.iterdir()
                if filename.name.startswith("train_")
            }

            # Loading each dataset
            for filename in dataset_path.iterdir():
                if filename.name.startswith("train_"):
                    idx = get_partition_index(filename.name)
                    train_inputs, train_outputs = self._read_file(filename)
                    partition_list[idx]["train_inputs"] = train_inputs
                    partition_list[idx]["train_outputs"] = train_outputs

                elif filename.name.startswith("test_"):
                    idx = get_partition_index(filename.name)
                    test_inputs, test_outputs = self._read_file(filename)
                    partition_list[idx]["test_inputs"] = test_inputs
                    partition_list[idx]["test_outputs"] = test_outputs

        except OSError:
            raise ValueError(f"No such file or directory: '{dataset_path}'")

        except KeyError:
            raise RuntimeError(
                f"Found partition without train files: partition {filename.name}"
            )

        # Saving partitions as a sorted list of (index, partition) tuples
        partition_list = sorted(partition_list.items(), key=(lambda t: get_key(t[0])))

        return partition_list

    def _read_file(self, filename):
        """Read a CSV containing partitions, or full datasets.

        Train and test files must be previously divided for the experiment to run.

        Parameters
        ----------
        filename : str or Path
            Full path to train or test file.

        Returns
        -------
        inputs : {array-like, sparse-matrix} of shape (n_samples, n_features)
            Vector of sample's features.

        outputs : array-like of shape (n_samples)
            Target vector relative to inputs.

        """
        # Separator is automatically found
        f = pd.read_csv(filename, header=None, engine="python")

        inputs = f.values[:, 0:(-1)]
        outputs = f.values[:, (-1)]

        return inputs, outputs

    def _check_dataset_list(self):
        """Check if there is some inconsistency in the dataset list.

        It also simplifies running all datasets inside one folder.

        Raises
        ------
        ValueError
            If the dataset list is inconsistent or contains non-string values.

        """
        base_path = Path(self.general_conf["basedir"])
        dataset_list = self.general_conf["datasets"]

        # Check if home path is shortened
        if str(base_path).startswith("~"):
            base_path = Path.home() / str(base_path)[1:]

        # Compatibility between python 2 and 3
        try:
            basestring = (unicode, str)
        except NameError:
            basestring = str

        # Check if 'all' is the only value, and if it is, expand it
        if len(dataset_list) == 1 and dataset_list[0] == "all":
            dataset_list = [item.name for item in base_path.iterdir() if item.is_dir()]

        elif not all(isinstance(item, basestring) for item in dataset_list):
            raise ValueError("Dataset list can only contain strings")

        self.general_conf["basedir"] = str(base_path)
        self.general_conf["datasets"] = dataset_list

    def _normalize_data(self, train_data, test_data):
        """Normalize the data.

        Test data normalization will be based on train data.

        Parameters
        ----------
        train_data : 2d array
            Contain the train data features.

        test_data : 2d array
            Contain the test data features.

        Returns
        -------
        train_normalized : np.ndarray
            Normalized training data.

        test_normalized : np.ndarray
            Normalized test data.

        """
        mm_scaler = preprocessing.MinMaxScaler().fit(train_data)

        return mm_scaler.transform(train_data), mm_scaler.transform(test_data)

    def _standardize_data(self, train_data, test_data):
        """Standardize the data.

        Test data standardization will be based on train data.

        Parameters
        ----------
        train_data : 2d array
            Contain the train data features.

        test_data : 2d array
            Contain the test data features.

        Returns
        -------
        train_standardized : np.ndarray
            Standardized training data.

        test_standardized : np.ndarray
            Standardized test data.

        """
        std_scaler = preprocessing.StandardScaler().fit(train_data)

        return std_scaler.transform(train_data), std_scaler.transform(test_data)

    def _check_params(self):
        """Check if all given configurations are syntactically correct.

        Performs two different transformations over parameter dictionaries when needed:

        - If one parameter's values are not inside a list, GridSearchCV
          will not be able to handle them, so they must be enclosed into one.

        - When an ensemble method, as OrderedPartitions, is chosen as classifier,
          transforms the dict of lists in which the parameters for the internal
          classifier are stated into a list of dicts (all possible combinations of
          those different parameters).

        Raises
        ------
        TypeError
            If any parameter value for the base_classifier is not a list.

        """
        random_seed = np.random.get_state()[1][0]
        for _, conf in self.configurations.items():

            parameters = conf["parameters"]

            try:
                estimator_cls = self._resolve_estimator(conf["classifier"])
            except Exception as e:
                raise ValueError(
                    f"Unable to resolve classifier '{conf['classifier']}': {e}"
                )

            prepared = prepare_param_grid(
                estimator=estimator_cls, param_grid=parameters, random_state=random_seed
            )

            conf["parameters"] = prepared

    def _get_optimal_estimator(
        self, train_inputs, train_outputs, classifier, parameters
    ):
        """Perform cross-validation over one dataset and configuration.

        Each configuration consists of one classifier and none, one or multiple
        hyper-parameters, that, in turn, can contain one or multiple values used to
        optimize the resulting model.

        At the end of cross-validation phase, the model with the specific combination of
        values from the hyper-parameters that achieved the best metrics from all the
        combinations will remain.

        Parameters
        ----------
        train_inputs : {array-like, sparse-matrix} of shape (n_samples, n_features)
            Vector of features for each sample for this dataset.

        train_outputs : array-like of shape (n_samples)
            Target vector relative to train_inputs.

        classifier : object
            Class implementing a mathematical model able to be trained and to perform
            predictions over given datasets.

        parameters : dict
            Dictionary containing parameters to optimize as keys, and the list of
            values that we want to compare as values.

        Returns
        -------
        optimal: GridSearchCV object or classifier object
            An already fitted model of the given classifier, with the best found
            parameters after cross-validation. If cross-validation is not needed, it will
            return the classifier model already trained.

        Raises
        ------
        AttributeError
            If the metric name is not found or cv_metric is not a string.

        """
        # No need to cross-validate when there is just one value per parameter
        if not is_searchcv(parameters):
            optimal = classifier(**parameters)

            start = time()
            optimal.fit(train_inputs, train_outputs)
            elapsed = time() - start

            optimal.refit_time_ = elapsed
            return optimal

        metric_name = self.general_conf["cv_metric"].strip().lower()
        scoring_function = load_metric_as_scorer(metric_name)

        # Creating object to split train data for cross-validation
        # This will make GridSearch have a pseudo-random beheaviour
        skf = StratifiedKFold(
            n_splits=self.general_conf["hyperparam_cv_nfolds"],
            shuffle=True,
            random_state=np.random.get_state()[1][0],
        )

        # Performing cross-validation phase
        optimal = GridSearchCV(
            estimator=classifier(),
            param_grid=parameters,
            scoring=scoring_function,
            n_jobs=self.general_conf["jobs"],
            cv=skf,
        )

        optimal.fit(train_inputs, train_outputs)

        return optimal

    def write_report(self):
        """Save summarized information about experiment through Results class."""
        if self.verbose:
            print("\nSaving Results...")

        # Names of each metric used (plus computational times)
        metrics_names = [x.strip().lower() for x in self.general_conf["metrics"]] + [
            "cv_time",
            "time",
        ]

        # Saving results through Results class
        self._results.save_summaries(metrics_names)


##########################
# END OF UTILITIES CLASS #
##########################


def check_packages_version():
    """Check if minimum version of packages used by this framework are installed."""
    print("Checking packages version...")

    print("NumPy...", end=" ")
    if parse_version(get_distribution("numpy").version) < parse_version("1.15.2"):
        print("OUTDATED. Upgrade to 1.15.2 or newer")
    else:
        print("OK")

    print("Pandas...", end=" ")
    if parse_version(get_distribution("pandas").version) < parse_version("0.23.4"):
        print("OUTDATED. Upgrade to 0.23.4 or newer")
    else:
        print("OK")

    print("Sacred...", end=" ")
    if parse_version(get_distribution("sacred").version) < parse_version("0.7.3"):
        print("OUTDATED. Upgrade to 0.7.3 or newer")
    else:
        print("OK")

    print("Scikit-Learn...", end=" ")
    if parse_version(get_distribution("scikit-learn").version) < parse_version(
        "0.20.0"
    ):
        print("OUTDATED. Upgrade to 0.20.0 or newer")
    else:
        print("OK")

    print("SciPy...", end=" ")
    if parse_version(get_distribution("scipy").version) < parse_version("1.1.0"):
        print("OUTDATED. Upgrade to 1.1.0 or newer")
    else:
        print("OK")


def get_key(key):
    """Convert dict key to int if possible, otherwise return as string.

    Parameters
    ----------
    key : str
        Dictionary key to convert.

    Returns
    -------
    converted_key : int or str
        Integer if conversion is possible, original string otherwise.

    """
    try:
        return int(key)
    except ValueError:
        return key
