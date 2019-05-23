<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:1 -->

1. [What is ORCA-python](#what-is-orca)
2. [Installing ORCA-python](#installing-orca)
	1. [Installation Requirements](#installation-requirements)
	2. [Download ORCA-python](#download-orca)
	3. [Installation Testing](#installation-testing)
3. [How to use ORCA-python](#how-to-use-orca)
	1. [Configuration Files](#configuration-files)
		1. [general-conf](#general-conf)
		2. [configurations](#configurations)
	2. [Running an Experiment](#running-an-experiment)

<!-- /TOC -->

## What is ORCA-python?

ORCA-python is an experimental framework, completely built on Python (integrated with scikit-learn and sacred modules), 
that seeks to automatize the run of machine learning experiments through simple-to-understand configuration files.

ORCA-python has been initially created to test ordinal classification, but it can handle regular classification algorithms,
as long as they are implemented in scikit-learn, or self-implemented following compatibility guidelines form scikit-learn.

In this README, we will explain how to use ORCA-python, and what you need to install in order to run it.


# Installing ORCA-python

ORCA-python has been developed and tested in GNU/Linux systems. It has been tested with Python 2.7.13 and Python 3.5.3.

## Installation Requirements

Besides the need of the aforementioned Python interpreter, you will need to install the next Python modules
in order to run an experiment (needs recent versions of scikit-learn >=0.20.0):

- numpy (tested with version 1.15.2)
- pandas (tested with version 0.23.4)
- sacred (tested with version 0.7.3)
- scikit-learn (tested with version 0.20.0)
- scipy (tested with version 1.1.0)

To install Python, you can use the package management system you like the most.\
For the modules installation, you may follow this [Python's Official Guide](https://docs.python.org/2/installing/index.html).

## Download ORCA-python

To download ORCA-python you can simply clone this GitHub repository by using the following commands:

  `$ git clone https://github.com/i22bomui/orca-python.git`
  
All the contents of the repository can also be downloaded from the GitHub site by using the "Download ZIP" button.

## Installation Testing

We provide a pre-made experiment (dataset and configuration file) to test if everything has been correctly installed.\
The way to run this test (and all experiments) is the following:

  ```
  # Go to framework main folder
  $ python config.py with configurations/single_test.json -l ERROR
  ```


# How to use ORCA-python


This tutorial uses three small datasets (balance-scale, contact-lenses and tae) contained in "datasets" folder.
The datasets are already partitioned with a 30-holdout experimental design (train and test pairs for each partition).

## Configuration Files

All experiments are run through configuration files, which are written in JSON format, and consist of two well differentiated 
sections:

  - **`general-conf`**: indicates basic information to run the experiment, such as the location to datasets, the different datasets names to run, etc. 
  - **`configurations`**: tells the framework what classification algorithms to apply over all the datasets, with the collection of hyper-parameters to tune.

Each one of this sections will be inside a dictionary, having the said section names as keys.


For a better understanding of the way this files works, it's better to follow an example, that can be found in: [configurations/full_functionality_test.json](https://github.com/i22bomui/orca-python/blob/master/configurations/full_functionality_test.json).

### general-conf

```
"general_conf": {

	"basedir": "ordinal-datasets/ordinal-regression/",
	"datasets": ["tae", "balance-scale", "contact-lenses"],
	"hyperparam_cv_folds": 3,
	"jobs": 10,
	"output_folder": "my_runs/",
	"metrics": ["ccr", "mae", "amae", "mze"],
	"cv_metric": "mae"
}
```
*note that all the keys (variable names) must be strings, while all pair: value elements are separated by commas.*

- **`basedir`**: folder containing all dataset subfolders, it doesn't allow more than one folder at a time. It can be indicated using a full path, or a relative one to the framework folder.
- **`datasets`**: name of datasets that will be experimented with. A subfolder with the same name must exist insise `basedir`.
- **`hyperparam_cv_folds`**: number of folds used while cross-validating.
- **`jobs`**: number of jobs used for GridSearchCV during cross-validation.
- **`output_folder`**: name of the folder where all experiment results will be stored.
- **`metrics`**: name of the accuracy metrics to measure the train and test performance of the classifier.
- **`cv_metric`**: error measure used for GridSearchCV to find the best set of hyper-parameters.

Most os this variables do have default values (specified in [config.py](https://github.com/i22bomui/orca-python/blob/master/config.py)), but "basedir" and "datasets" must always be written for the experiment to be run. Take into account, that all variable names in "general-conf" cannot be modified, otherwise the experiment will fail.


### configurations

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

		"classifier": "OrdinalDecomposition",
		"parameters": {
			"dtype": "OrderedPartitions",
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

		"classifier": "OrdinalDecomposition",
		"parameters": {
			"dtype": ["OrderedPartitions", "OneVsNext"],
			"decision_method": "exponential_loss",
			"base_classifier": "sklearn.linear_model.LogisticRegression",
			"parameters": {
				"C": [0.01, 0.1, 1, 10],
				"penalty": ["l1","l2"]
			}

		}
	}
}
```

Each configuration has a name (whatever you want), and consists of:

- **`classifier`**: tells the framework wich classifier to use. Can be specified in two different ways:
	- A relative path to the classifier in sklearn module.
	- The name of a built-in class in Classifiers folder (found in the main folder of the project).
- **`parameters`**: hyper-paramers to tune, having each one of them a list of values to cross-validate (not really necessary, can be just one value).

*In ensemble methods, as `OrdinalDecomposition`, you must nest another classifier (the base classifier, which doesn't have a configuration name), with it's respective parameters to tune.*



## Running an Experiment

As viewed in [Installation Testing](#installation-testing), running an experiment is as simple as executing Config.py
with the python interpreter, and tell what configuration file to use for this expetiment, resulting in the next command:

  `$ python config.py with experiment_file.json`

Running an experiment this way has two problems though, one of them being an excesive verbosity from Sacred's python module,
while the other consists in the non-reproducibility of the experiments results, due to the lack of a fixed seed.

Both problems can be easily fixed. The seed can be specified after "with" in the command:

  `$ python config.py with experiment_file.json seed=12345`
  
while we can silence Sacred just by adding "-l ERROR" at the end of the line (not necessarily at the end).

  `$ python config.py with experiment_file.json seed=12345 -l ERROR`


