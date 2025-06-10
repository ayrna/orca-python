"""Tests for the utility functions in the datasets module."""

import numpy as np
import pytest
from pathlib import Path

import orca_python.datasets.data
from orca_python.testing.testing_utils import TEST_RANDOM_STATE
from orca_python.datasets.datasets import (
    get_data_path,
    dataset_exists,
    is_undivided,
    has_unseeded_split,
    has_seeded_split,
    check_ambiguity,
    load_datafile,
    load_dataset,
    shuffle_data,
)


def create_tmp_dataset(tmp_path, dataset_name, filename):
    data = """
        134.0,1.9,0.6,18.4,8.2,1
        115.0,6.3,1.2,4.7,14.4,1
        136.0,1.4,0.3,32.6,8.4,1
        141.0,2.5,1.3,8.5,7.5,1
        119.0,0.8,0.7,56.4,21.6,1
        139.0,4.2,0.7,4.3,6.3,1
        108.0,3.5,0.6,1.7,1.4,1
        """
    (tmp_path / dataset_name).mkdir(exist_ok=True)
    file_path = tmp_path / dataset_name / filename
    with open(file_path, "w") as f:
        f.write(data)


def test_get_data_path():
    assert get_data_path() == Path(orca_python.datasets.data.__file__).parent


def test_dataset_exists(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    assert dataset_exists("dataset", tmp_path)
    assert not dataset_exists("dataset", tmp_path / "datapath")
    assert not dataset_exists("dataset2", tmp_path)


def test_is_undivided(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    assert is_undivided("dataset", tmp_path)


def test_has_unseeded_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    assert has_unseeded_split("dataset", tmp_path)


def test_has_seeded_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", f"train_dataset_{TEST_RANDOM_STATE}.csv")
    create_tmp_dataset(tmp_path, "dataset", f"test_dataset_{TEST_RANDOM_STATE}.csv")
    assert has_seeded_split("dataset", TEST_RANDOM_STATE, tmp_path)


def test_check_ambiguity_divided(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    assert is_undivided("dataset", tmp_path)

    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    assert has_unseeded_split("dataset", tmp_path)

    with pytest.raises(ValueError):
        check_ambiguity("dataset", tmp_path, seed=TEST_RANDOM_STATE)


def test_check_ambiguity_with_seed(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    assert has_unseeded_split("dataset", tmp_path)

    create_tmp_dataset(tmp_path, "dataset", f"train_dataset_{TEST_RANDOM_STATE}.csv")
    assert has_seeded_split("dataset", TEST_RANDOM_STATE, tmp_path)

    with pytest.raises(ValueError):
        check_ambiguity("dataset", tmp_path, seed=TEST_RANDOM_STATE)


def test_load_datafile_not_found(tmp_path):
    dataset_name = "dataset"
    error_msg = f"No dataset found for '{dataset_name}' in '{tmp_path}'."
    with pytest.raises(FileNotFoundError, match=error_msg):
        load_datafile(dataset_name, split="undivided", data_path=tmp_path)

    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    assert dataset_exists(dataset_name, tmp_path)


def test_load_datafile_undivided(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    X, y = load_datafile("dataset", split="undivided", data_path=tmp_path)
    assert X is not None and y is not None


def test_load_datafile_train_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    X, y = load_datafile("dataset", split="train", data_path=tmp_path)
    assert X is not None and y is not None


def test_load_datafile_test_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    X, y = load_datafile("dataset", split="test", data_path=tmp_path)
    assert X is not None and y is not None


def test_load_datafile_with_valid_seed(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset_0.csv")
    X, y = load_datafile("dataset", split="train", seed=0, data_path=tmp_path)
    assert X is not None and y is not None


def test_load_datafile_with_invalid_seed(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "dataset.csv")
    X, y = load_datafile("dataset", split="train", seed=1, data_path=tmp_path)
    assert X is None and y is None


def test_load_dataset_not_found(tmp_path):
    dataset_name = "dataset"
    error_msg = f"No dataset found for '{dataset_name}' in '{tmp_path}'."

    with pytest.raises(FileNotFoundError, match=error_msg):
        load_dataset(dataset_name, data_path=tmp_path)

    (tmp_path / dataset_name).mkdir(exist_ok=True)
    assert dataset_exists(dataset_name, tmp_path)

    with pytest.raises(FileNotFoundError, match=error_msg):
        load_dataset(dataset_name, data_path=tmp_path)

    create_tmp_dataset(tmp_path, "dataset", "dataset1.csv")
    with pytest.raises(FileNotFoundError, match=error_msg):
        load_dataset(dataset_name, data_path=tmp_path)


def test_load_dataset_split_without_seed(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    X_train, y_train, X_test, y_test = load_dataset("dataset", data_path=tmp_path)

    assert X_train is not None and y_train is not None
    assert X_test is not None and y_test is not None


def test_load_dataset_split_with_seed(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", f"train_dataset_{TEST_RANDOM_STATE}.csv")
    create_tmp_dataset(tmp_path, "dataset", f"test_dataset_{TEST_RANDOM_STATE}.csv")
    X_train, y_train, X_test, y_test = load_dataset(
        "dataset", data_path=tmp_path, seed=TEST_RANDOM_STATE
    )

    assert X_train is not None and y_train is not None
    assert X_test is not None and y_test is not None


def test_load_dataset_train_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    X_train, y_train, X_test, y_test = load_dataset("dataset", data_path=tmp_path)

    assert X_train is not None and y_train is not None
    assert X_test is None and y_test is None


def test_load_dataset_test_split(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    X_train, y_train, X_test, y_test = load_dataset("dataset", data_path=tmp_path)

    assert X_train is None and y_train is None
    assert X_test is not None and y_test is not None


def test_load_dataset_missing_data(tmp_path):
    error_msg = f"No dataset found for 'dataset' in '{tmp_path}'."

    with pytest.raises(FileNotFoundError, match=error_msg):
        load_dataset("dataset", data_path=tmp_path)


def test_shuffle_data(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    X_train, y_train, X_test, y_test = load_dataset("dataset", data_path=tmp_path)
    X_train_new, y_train_new, X_test_new, y_test_new = shuffle_data(
        X_train, y_train, X_test, y_test, seed=TEST_RANDOM_STATE
    )

    assert X_train_new.shape == X_train.shape
    assert X_test_new.shape == X_test.shape
    assert len(y_train_new) == len(y_train)
    assert len(y_test_new) == len(y_test)

    assert np.sum(y_train_new == 0) == np.sum(y_train == 0)
    assert np.sum(y_test_new == 0) == np.sum(y_test == 0)


def test_shuffle_data_invalid_train_size():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
        shuffle_data(X, y, None, None, seed=TEST_RANDOM_STATE, train_size=-0.5)

    with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
        shuffle_data(X, y, None, None, seed=TEST_RANDOM_STATE, train_size=2)


def test_shuffle_data_no_data():
    with pytest.raises(ValueError, match="No data provided for shuffling"):
        shuffle_data(None, None, None, None, seed=TEST_RANDOM_STATE)


def test_shuffle_data_only_train(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "train_dataset.csv")
    X_train, y_train, _, _ = load_dataset("dataset", data_path=tmp_path)
    X_train_new, y_train_new, X_test_new, y_test_new = shuffle_data(
        X_train, y_train, None, None, seed=TEST_RANDOM_STATE
    )

    assert X_train_new.shape[0] + X_test_new.shape[0] == len(X_train)
    assert X_train_new.shape[1] == X_train.shape[1]
    assert len(y_train_new) + len(y_test_new) == len(y_train)


def test_shuffle_data_only_test(tmp_path):
    create_tmp_dataset(tmp_path, "dataset", "test_dataset.csv")
    _, _, X_test, y_test = load_dataset("dataset", data_path=tmp_path)
    X_train_new, y_train_new, X_test_new, y_test_new = shuffle_data(
        None, None, X_test, y_test, seed=TEST_RANDOM_STATE
    )

    assert X_train_new.shape[0] + X_test_new.shape[0] == len(X_test)
    assert X_train_new.shape[1] == X_test.shape[1]
    assert X_test_new.shape[1] == X_test.shape[1]
    assert len(y_train_new) + len(y_test_new) == len(y_test)


def test_shuffle_data_mismatched_dimensions():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])

    error_msg = "X_train and y_train dimensions don't match: 3 vs 2"
    with pytest.raises(ValueError, match=error_msg):
        shuffle_data(X, y, None, None, seed=TEST_RANDOM_STATE)

    error_msg = "X_test and y_test dimensions don't match: 3 vs 2"
    with pytest.raises(ValueError, match=error_msg):
        shuffle_data(None, None, X, y, seed=TEST_RANDOM_STATE)


def test_shuffle_data_non_stratificable():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 0])
    X_train, y_train, X_test, y_test = shuffle_data(
        X, y, None, None, seed=TEST_RANDOM_STATE
    )

    assert X_train.shape[0] + X_test.shape[0] == len(X)
    assert len(y_train) + len(y_test) == len(y)
