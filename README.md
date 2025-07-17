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

This tutorial uses three small datasets (balance-scale, contact-lenses and tae) contained in "datasets" folder.
The datasets are already partitioned with a 30-holdout experimental design (train and test pairs for each partition).

### Configuration Files

All experiments are run through configuration files, which are written in JSON format, and consist of two well differentiated
sections:

  - **`general-conf`**: indicates basic information to run the experiment, such as the location to datasets, the names of the different datasets to run, etc.
  - **`configurations`**: tells the framework what classification algorithms to apply over all the datasets, with the collection of hyper-parameters to tune.

Each one of this sections will be inside a dictionary, having the said section names as keys.

For a better understanding of the way this files works, it's better to follow an example, that can be found in: [configurations/full_functionality_test.json](https://github.com/ayrna/orca-python/blob/master/configurations/full_functionality_test.json).

#### general-conf

```
"general_conf": {

    "basedir": "ordinal-datasets/ordinal-regression/",
    "datasets": ["tae", "balance-scale", "contact-lenses"],
    "hyperparam_cv_folds": 3,
    "jobs": 10,
    "input_preprocessing": "std",
    "output_folder": "my_runs/",
    "metrics": ["ccr", "mae", "amae", "mze"],
    "cv_metric": "mae"
}
```
*note that all the keys (variable names) must be strings, while all pair: value elements are separated by commas.*

- **`basedir`**: folder containing all dataset subfolders, it doesn't allow more than one folder at a time. It can be indicated using a full path, or a relative one to the framework folder.
- **`datasets`**: name of datasets that will be experimented with. A subfolder with the same name must exist inside `basedir`.
- **`hyperparam_cv_folds`**: number of folds used while cross-validating.
- **`jobs`**: number of jobs used for GridSearchCV during cross-validation.
- **`input_preprocessing`**: type of preprocessing to apply to the data, **`std`** for standardization and **`norm`** for normalization. Assigning an empty srtring will omit the preprocessing process.
- **`output_folder`**: name of the folder where all experiment results will be stored.
- **`metrics`**: name of the accuracy metrics to measure the train and test performance of the classifier.
- **`cv_metric`**: error measure used for GridSearchCV to find the best set of hyper-parameters.

Most of this variables do have default values (specified in [config.py](https://github.com/ayrna/orca-python/blob/master/config.py)), but "basedir" and "datasets" must always be written for the experiment to be run. Take into account, that all variable names in "general-conf" cannot be modified, otherwise the experiment will fail.


#### configurations

this dictionary will contain, at the same time, one dictionary for each configuration to try over the datasets during the experiment. This is, a classifier with some specific hyper-parameters to tune. (Keep in mind, that if two or more configurations share the same name, the later ones will be omitted)

```
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
	},
	"LR": {

		"classifier": "orca_python.classifiers.OrdinalDecomposition",
		"parameters": {
			"dtype": ["ordered_partitions", "one_vs_next"],
			"decision_method": "exponential_loss",
			"base_classifier": "sklearn.linear_model.LogisticRegression",
			"parameters": {
				"solver": ["liblinear"],
				"C": [0.01, 0.1, 1, 10],
				"penalty": ["l1","l2"]
			}

		}
	},
	"REDSVM": {

		"classifier": "orca_python.classifiers.REDSVM",
		"parameters": {
		    "t": 2,
			"c": [0.1, 1, 10],
			"g": [0.1, 1, 10],
			"r": 0,
			"m": 100,
			"e": 0.001,
			"h": 1
		}

	},
	"SVOREX": {

		"classifier": "orca_python.classifiers.SVOREX",
		"parameters": {
			"kernel_type": 0,
			"c": [0.1, 1, 10],
			"k": [0.1, 1, 10],
			"t": 0.001
		}

	}
}
```

Each configuration has a name (whatever you want), and consists of:

- **`classifier`**: tells the framework which classifier to use. Can be specified in two different ways:
    - A relative path to the classifier in sklearn module.
    - The name of a built-in class in Classifiers folder (found in the main folder of the project).
- **`parameters`**: hyper-parameters to tune, having each one of them a list of values to cross-validate (not really necessary, can be just one value).

*In ensemble methods, as `OrdinalDecomposition`, you must nest another classifier (the base classifier, which doesn't have a configuration name), with it's respective parameters to tune.*


### Running Experiments

As viewed in [Installation Testing](#installation-testing), running an experiment is as simple as executing Config.py
with the python interpreter, and tell what configuration file to use for this experiment, resulting in the next command:

  `$ python config.py with experiment_file.json`

Running an experiment this way has two problems though, one of them being an excessive verbosity from Sacred,
while the other consists of the non-reproducibility of the results of the experiment, due to the lack of a fixed seed.

Both problems can be easily fixed. The seed can be specified after "with" in the command:

  `$ python config.py with experiment_file.json seed=12345`

while we can silence Sacred just by adding "-l ERROR" at the end of the line (not necessarily at the end).

  `$ python config.py with experiment_file.json seed=12345 -l ERROR`
