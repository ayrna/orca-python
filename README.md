# ORCA-python

## What is ORCA-python?

**ORCA-python** is an experimental framework built on Python that seamlessly integrates with scikit-learn and sacred modules to automate machine learning experiments through simple JSON configuration files. Initially designed for ordinal classification, it supports regular classification algorithms as long as they are compatible with scikit-learn, making it easy to run reproducible experiments across multiple datasets and classification methods.

## Table of Contents

- [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
    - [Testing Installation](#testing-installation)
- [Quick Start](#quick-start)
- [Configuration Files](#configuration-files)
    - [general-conf](#general-conf)
    - [configurations](#configurations)
- [Running Experiments](#running-experiments)
    - [Basic Usage](#basic-usage)
    - [Recommended Usage](#recommended-usage)
    - [Example Output](#example-output)

## Installation

### Requirements

ORCA-python requires Python 3.8 or higher and is tested on Python 3.8, 3.9, 3.10, and 3.11.

All dependencies are managed through `pyproject.toml` and include:
- numpy (>=1.24.4)
- pandas (>=2.0.3)
- sacred (>=0.8.7)
- scikit-learn (>=1.3.2)
- scipy (>=1.10.1)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ayrna/orca-python
   cd orca-python
   ```

2. **Install the framework**:
   ```bash
   pip install .
   ```

   For development purposes, use editable installation:
   ```bash
   pip install -e .
   ```

   Optional dependencies for development:
   ```bash
   pip install -e .[dev]
   ```

> **Note:** The editable mode is required for running tests due to automatic dependency resolution.

### Testing Installation

Test your installation with the provided example:

```bash
python config.py with orca_python/configurations/full_functionality_test.json -l ERROR
```

## Quick Start

ORCA-python includes sample datasets with pre-partitioned train/test splits using a 30-holdout experimental design.

**Basic experiment configuration:**

```json
{
    "general_conf": {
        "basedir": "orca_python/datasets/data",
        "datasets": ["balance-scale", "contact-lenses", "tae"],
        "hyperparam_cv_nfolds": 3,
        "output_folder": "results/",
        "metrics": ["ccr", "mae", "amae"],
        "cv_metric": "mae"
    },
    "configurations": {
        "SVM": {
            "classifier": "sklearn.svm.SVC",
            "parameters": {
                "C": [0.001, 0.1, 1, 10, 100],
                "gamma": [0.1, 1, 10]
            }
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
                    "probability": ["True"]
                }
            }
        }
    }
}
```

**Run the experiment:**
```bash
python config.py with my_experiment.json -l ERROR
```

Results are saved in `results/` folder with performance metrics for each dataset-classifier combination. The framework automatically performs cross-validation, hyperparameter tuning, and evaluation on test sets.

## Configuration Files

Experiments are defined using JSON configuration files with two main sections: general_conf for experiment settings and configurations for classifier definitions.

### general-conf

Controls global experiment parameters.

**Required parameters:**
- **`basedir`**: folder containing all dataset subfolders, it doesn't allow more than one folder at a time. It can be indicated using a full path, or a relative one to the framework folder.
- **`datasets`**: name of datasets that will be experimented with. A subfolder with the same name must exist inside `basedir`.

**Optional parameters:**
- **`hyperparam_cv_folds`**: number of folds used while cross-validating.
- **`jobs`**: number of jobs used for GridSearchCV during cross-validation.
- **`input_preprocessing`**: data preprocessing (`"std"` for standardization, `"norm"` for normalization, `""` for none)
- **`output_folder`**: name of the folder where all experiment results will be stored.
- **`metrics`**: name of the accuracy metrics to measure the train and test performance of the classifier.
- **`cv_metric`**: error measure used for GridSearchCV to find the best set of hyper-parameters.

### configurations

Defines classifiers and their hyperparameters for GridSearchCV. Each configuration has a name and consists of:

- **`classifier`**: scikit-learn path or built-in ORCA-python classifier
- **`parameters`**: hyperparameters for grid search (nested for ensemble methods)

## Running Experiments

### Basic Usage

```bash
python config.py with experiment_file.json
```

### Recommended Usage

For reproducible results with minimal output:

```bash
python config.py with experiment_file.json seed=12345 -l ERROR
```

**Parameters:**
- `seed`: fixed random seed for reproducibility
- `-l ERROR`: reduces Sacred framework verbosity

### Example Output

Results are stored in the specified output folder with detailed performance metrics and hyperparameter information for each dataset and configuration combination.
