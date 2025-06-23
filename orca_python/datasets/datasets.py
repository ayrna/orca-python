"""Utility functions for datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import orca_python.datasets.data


def get_data_path():
    """Get the absolute path of the orca_python.datasets.data module.

    Returns
    -------
    data_path : path
        ORCA-python datasets path.

    Examples
    --------
    >>> from orca_python.datasets.datasets import get_data_path
    >>> get_data_path()
    PosixPath('/path/to/orca_python/datasets/data')

    """
    return Path(orca_python.datasets.data.__file__).parent


def dataset_exists(dataset_name, data_path):
    """Check if the dataset directory exists within the data path.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    data_path : str or path
        Root directory containing dataset files.

    Returns
    -------
    is_dir : bool
        True if both `data_path` and the dataset directory exist, False otherwise.

    Examples
    --------
    >>> from orca_python.datasets.datasets import dataset_exists, get_data_path
    >>> data_path = get_data_path()
    >>> dataset_exists("car", data_path)
    True
    >>> dataset_exists("tae", data_path)
    True
    >>> dataset_exists("nonexistent_dataset", data_path)
    False

    """
    data_path = Path(data_path)
    return data_path.is_dir() and (data_path / dataset_name).is_dir()


def is_undivided(dataset_name, data_path):
    """Check if there is a dataset file with no train/test split.

    Parameters
    ----------
    dataset_name : str
        Name for the specific dataset.

    data_path : str or path
        Root directory containing dataset files.

    Returns
    -------
    is_undivided : bool
        True if the full dataset file exists, False otherwise.

    Examples
    --------
    >>> from orca_python.datasets.datasets import is_undivided, get_data_path
    >>> data_path = get_data_path()
    >>> is_undivided("tae", data_path)
    False
    >>> is_undivided("nonexistent_dataset", data_path)
    False

    """
    data_path = Path(data_path)
    file_path = data_path / dataset_name / f"{dataset_name}.csv"
    return file_path.exists()


def has_unseeded_split(dataset_name, data_path):
    """Check if the dataset has train/test split files without a seed.

    Parameters
    ----------
    dataset_name : str
        Name for the specific dataset.

    data_path : str or path
        Root directory containing dataset files.

    Returns
    -------
    is_unseeded : bool
        True if train and/or test files without a seed exist, False otherwise.

    Examples
    --------
    >>> from orca_python.datasets.datasets import has_unseeded_split, get_data_path
    >>> data_path = get_data_path()
    >>> has_unseeded_split("tae", data_path)
    False
    >>> has_unseeded_split("nonexistent_dataset", data_path)
    False

    """
    data_path = Path(data_path)
    return any(
        (data_path / dataset_name / f"{split}_{dataset_name}.csv").exists()
        for split in ["train", "test"]
    )


def has_seeded_split(dataset_name, seed, data_path):
    """Check if the dataset has train/test split files with a specific seed.

    Parameters
    ----------
    dataset_name : str
        Name for the specific dataset.

    seed : int
        Numerical seed ensuring reproducible randomization.

    data_path : str or path
        Root directory containing dataset files.

    Returns
    -------
    is_seeded : bool
        True if train and/or test files with the specified seed exist, False otherwise.

    Examples
    --------
    >>> from orca_python.datasets.datasets import has_seeded_split, get_data_path
    >>> data_path = get_data_path()
    >>> has_seeded_split("tae", 0, data_path)
    True
    >>> has_seeded_split("tae", 99, data_path)
    False
    >>> has_seeded_split("nonexistent_dataset", 0, data_path)
    False

    """
    data_path = Path(data_path)
    return any(
        (data_path / dataset_name / f"{split}_{dataset_name}_{seed}.csv").exists()
        for split in ["train", "test"]
    )


def check_ambiguity(dataset_name, data_path, seed=None):
    """Check for ambiguity in dataset format.

    Parameters
    ----------
    dataset_name : str
        Name for the specific dataset.

    data_path : str or path
        Root directory containing dataset files.

    seed: int, optional
        Numerical seed ensuring reproducible randomization.

    Raises
    ------
    ValueError
        If an ambiguity exists, such as both full dataset and split files in the
        directory, or if both seeded and unseeded split files are present.

    Examples
    --------
    >>> from orca_python.datasets.datasets import check_ambiguity, get_data_path
    >>> data_path = get_data_path()
    >>> check_ambiguity("tae", data_path)
    >>> check_ambiguity("tae", data_path, seed=0)
    >>> check_ambiguity("nonexistent_dataset", data_path, seed=0)

    """
    data_path = Path(data_path)
    undivided = is_undivided(dataset_name, data_path)
    unseeded = has_unseeded_split(dataset_name, data_path)
    seeded = seed is not None and has_seeded_split(dataset_name, seed, data_path)

    if undivided and (unseeded or seeded):
        raise ValueError(
            f"Ambiguity detected: Both a undivided dataset '{dataset_name}.csv' and"
            " split files are present. Please ensure only one format is used."
        )

    if unseeded and seeded:
        raise ValueError(
            "Ambiguity detected: Split files with and without seed are both present"
            f" for '{dataset_name}'. Please remove one type to avoid conflict."
        )


def load_datafile(dataset_name, split="undivided", data_path=None, seed=None):
    """Load a dataset file based on split type and seed.

    Parameters
    ----------
    dataset_name : str
        Name for the specific dataset.

    split : str, default='undivided'
        Data division type ('undivided', 'train' or 'test').

    data_path : str or path, optional
        Root directory containing dataset files. If None, defaults to the orca-python
        datasets path.

    seed : int, optional
        Numerical seed ensuring reproducible randomization.

    Returns
    -------
    X, y : array or None
        Feature and target arrays. Both may be None if the file does not exist.

    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist.

    Examples
    --------
    >>> from orca_python.datasets.datasets import load_datafile, get_data_path
    >>> X_train, y_train = load_datafile("tae", split="train", seed=None)
    >>> X_train is None, y_train is None
    (True, True)
    >>> X_train, y_train = load_datafile("tae", split="train", seed=1)
    >>> X_train.shape, y_train.shape
    ((113, 54), (113,))
    >>> X_test, y_test = load_datafile("tae", split="test", seed=1)
    >>> X_test.shape, y_test.shape
    ((38, 54), (38,))

    """
    data_path = Path(data_path or get_data_path()).expanduser()

    if not dataset_exists(dataset_name, data_path):
        raise FileNotFoundError(
            f"No dataset found for '{dataset_name}' in '{data_path}'."
        )

    split_str = f"{split + '_' if split and split != 'undivided' else ''}"
    seed_str = f"{'_' + str(seed) if seed is not None else ''}"
    file_name = f"{split_str}{dataset_name}{seed_str}.csv"
    file_path = data_path / dataset_name / file_name

    try:
        df = pd.read_csv(file_path, header=None, engine="python")
        return df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None, None


def load_dataset(dataset_name, data_path=None, seed=None):
    """Load a dataset from the specified directory.

    The dataset can be stored in one of three formats:
        1. **Undivided dataset**: A single file `[dataset_name].csv` containing both
           features and target.
        2. **Train/Test split**: Two separate files, `train_[dataset_name].csv` and
           `test_[dataset_name].csv`.
        3. **Train/Test split with seed**: Seed-specific files,
           `train_[dataset_name]_[seed].csv` and
           `test_[dataset_name]_[seed].csv`.

    The function automatically detects the format and loads the data accordingly.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.

    data_path : str or path, optional
        Root directory containing dataset files. If None, defaults to the orca-python
        datasets path.

    seed : int, optional
        Numerical seed ensuring reproducible randomization.

    Returns
    -------
    X_train, y_train, X_test, y_test: array or None
        - If the dataset is undivided: X and y are returned, test data is None.
        - If the dataset has a split: train and test data arrays are returned. Any
        value may be None if not available.

    Raises
    ------
    FileNotFoundError
        If the data path does not exist.

    ValueError
        If an ambiguity exists, such as both full dataset and split files in the
        directory.

    Examples
    --------
    >>> from orca_python.datasets.datasets import load_dataset
    >>> X_train, y_train, X_test, y_test = load_dataset("tae", seed=0)
    >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
    ((113, 54), (113,), (38, 54), (38,))
    >>> X_train, y_train, X_test, y_test = load_dataset("car", seed=0)
    >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
    ((1296, 21), (1296,), (432, 21), (432,))

    """
    data_path = Path(data_path or get_data_path()).expanduser()

    if not dataset_exists(dataset_name, data_path):
        raise FileNotFoundError(
            f"No dataset found for '{dataset_name}' in '{data_path}'."
        )

    check_ambiguity(dataset_name, data_path, seed)

    X_train, y_train, X_test, y_test = None, None, None, None

    if is_undivided(dataset_name, data_path):
        X_train, y_train = load_datafile(dataset_name, "undivided", data_path)
    elif has_unseeded_split(dataset_name, data_path):
        X_train, y_train = load_datafile(dataset_name, "train", data_path)
        X_test, y_test = load_datafile(dataset_name, "test", data_path)
    elif seed is not None and has_seeded_split(dataset_name, seed, data_path):
        X_train, y_train = load_datafile(dataset_name, "train", data_path, seed)
        X_test, y_test = load_datafile(dataset_name, "test", data_path, seed)
    else:
        raise FileNotFoundError(
            f"No dataset found for '{dataset_name}' in '{data_path}'."
        )

    return X_train, y_train, X_test, y_test


def shuffle_data(X_train, y_train, X_test, y_test, seed, train_size=0.75):
    """Shuffle data by combining train and test sets and splitting them again.

    Handles cases where either train or test set may be None.

    Parameters
    ----------
    X_train : np.ndarray or None
        Feature matrix used specifically for model training.

    y_train : np.ndarray or None
        Target vector relative to X_train.

    X_test : np.ndarray or None
        Feature matrix for model evaluation and prediction.

    y_test : np.ndarray or None
        Target vector relative to X_test.

    seed : int
        Numerical seed ensuring reproducible randomization.

    train_size : float, default=0.75
        Proportion of the dataset to allocate to training. Default is 0.75.

    Returns
    -------
    X_train, y_train, X_test, y_test : array or None
        Shuffled training and test sets. All are arrays.

    Raises
    ------
    ValueError
        - If both training and test sets are None.
        - If X and y dimensions don't match.
        - If train_size is not between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> from orca_python.datasets.datasets import load_dataset, shuffle_data
    >>> X_train, y_train, X_test, y_test = load_dataset("tae", seed=0)
    >>> X_train.shape, y_train.shape, X_test.shape, y_test.shape
    ((113, 54), (113,), (38, 54), (38,))
    >>> X_train_sh, y_train_sh, X_test_sh, y_test_sh = shuffle_data(
    ...     X_train, y_train, X_test, y_test, seed=1)
    >>> X_train_sh.shape, y_train_sh.shape, X_test_sh.shape, y_test_sh.shape
    ((113, 54), (113,), (38, 54), (38,))
    >>> # Verify data is redistributed by comparing first rows
    >>> np.array_equal(X_train[:5], X_train_sh[:5])
    False

    """
    if train_size <= 0 or train_size >= 1:
        raise ValueError("train_size must be between 0 and 1")

    if X_train is None and X_test is None:
        raise ValueError("No data provided for shuffling")

    if X_train is not None and y_train is not None and len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train dimensions don't match: {X_train.shape[0]} vs {len(y_train)}"
        )

    if X_test is not None and y_test is not None and len(X_test) != len(y_test):
        raise ValueError(
            f"X_test and y_test dimensions don't match: {X_test.shape[0]} vs {len(y_test)}"
        )

    if X_train is None:
        X_train = (
            np.empty((0, X_test.shape[1])) if X_test is not None else np.empty((0, 0))
        )
        y_train = np.empty(0)

    if X_test is None:
        X_test = (
            np.empty((0, X_train.shape[1])) if X_train is not None else np.empty((0, 0))
        )
        y_test = np.empty(0)

    X_full = np.vstack((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))

    if X_train.size > 0 and X_test.size > 0:
        train_size = len(X_train) / (len(X_train) + len(X_test))

    stratify_param = y_full if len(np.unique(y_full)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        train_size=train_size,
        random_state=seed,
        stratify=stratify_param,
    )

    return X_train, y_train, X_test, y_test
