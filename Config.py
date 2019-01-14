
import os
from sacred import Experiment

from Utilities import Utilities

ex = Experiment('Experiment Configuration')

@ex.config
def default_config():

	general_conf = {"basedir": "",
					"dataset": "",
					"folds": 3,
					"jobs": 1,
					"metrics": "ccr",
					"cv_metric": "ccr",
					"runs_folder": "my_runs/"
					}

	algorithms = {}


@ex.automain
def main(general_conf, configurations):


	interface = Utilities(general_conf, configurations)
	interface.runExperiment()
	interface.writeReport()



