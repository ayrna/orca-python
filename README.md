## What is ORCA?

ORCA is an experimental framework, completely built on Python (integrated with scikit-learn and sacred modules), 
that seeks to automatize the run of machine learning experiments through simple-to-understand configuration files.

ORCA has been initially created to test ordinal classification, but it can handle regular classification algorithms,
as long as they are implemented in scikit-learn, or self-implemented following compatibility guidelines form scikit-learn.

In this README, we will explain how to use ORCA, and what you need to install in order to run it.


# Installing ORCA

ORCA has been developed and tested in GNU/Linux systems. It has been tested with Python 2.7.13.

## Installation Requirements

Besides the need of the aforementioned Python interpreter, you will need to install the next Python modules
in order to run an experiment:

- numpy (tested with version 1.15.2)
- pandas (tested with version 0.23.4)
- sacred (tested with version 0.7.3)
- scikit-learn (tested with version 0.20.0)
- scipy (tested with version 1.1.0)

To install Python, you can use the package management system you like the most.\
For the modules installation, you may follow this [Python's Official Guide](https://docs.python.org/2/installing/index.html).

## Download ORCA

To download ORCA you can simply clone this GitHub repository by using the following commands:

  `$ git clone https://github.com/i22bomui/orca-python.git`
  
All the contents of the repository can also be downloaded from the GitHub site by using the "Download ZIP" button.

## Installation Testing

We provide a pre-made experiment (dataset and configuration file) to test if everything has been correctly installed.\
The way to run this test (and all experiments) is the following:

  ```
  # Go to framework main folder
  $ python Config.py with Configurations/single_test.json -l ERROR
  ```


# How to use ORCA


This tutorial uses three small datasets (balance-scale, contact-lenses and tae) contained in "Datasets" folder.
The datasets are already partitioned with a 30-holdout experimental design (train and test pairs for each partition).

## Configuration Files

All experiments are run through configuration files, which are written in JSON format, and consist of two well differentiated 
sections:

  - **general-conf**: indicates basic information to run the experiment, such as the location to datasets, the different datasets names to run, etc. 
  - **configurations**: tells the framework what classification algorithms to apply over all the datasets, with the collection of hyper-parameters to tune.

Each one of this sections will be inside a dictionary, having the said section names as keys.


For a better understanding of the way this files works, it's better to follow an example:

### general-conf

```
"general_conf": {

	"basedir": "ordinal-datasets/ordinal-regression/",
	"datasets": ["tae", "balance-scale", "contact-lenses"],
	"folds": 3,
	"jobs": 10,
	"runs_folder": "my_runs/",
	"metrics": ["ccr", "mae", "amae", "mze"],
	"cv_metric": "mae"
}
```


*note that all the keys (variable names) must be strings, while all dictionaries are separated by commas.*

This example can be found in [Configurations/full_functionality_test.json](https://github.com/i22bomui/orca-python/blob/master/Configurations/full_functionality_test.json).

## Running an Experiment

As viewed in [Installation Testing](#installation-testing), running an experiment is as simple as executing Config.py
with the python interpreter, and tell what configuration file to use for this expetiment, resulting in the next command:

  `$ python Config.py with runtest.json`

Running an experiment this way has two problems though, one of them being an excesive verbosity from Sacred's python module,
while the other consists in the non-reproducibility of the experiments results, due to the lack of a fixed seed.

Both problems can be easily fixed. The seed can be specified after "with" in the command:

  `$ python Config.py with runtest.json seed=12345`
  
while we can silence Sacred just by adding "-l ERROR" at the end of the line (not necessarily).

  `$ python Config.py with runtest.json seed=12345 -l ERROR`








