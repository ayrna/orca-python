## What is ORCA?

ORCA is an experimental framework, completely built on Python (integrated with scikit-learn and sacred modules), 
that seeks to automatize the run of machine learning experiments through simple-to-understand configuration files.

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

## Installation testing

We provide a pre-made experiment (dataset and configuration file) to test if everything has been correctly installed.\
The way to run this test (and all experiments) is the following:

  `# Go to framework main folder`\
  `$ python Config.py with Configurations/testing-config.json`




# How to use ORCA
