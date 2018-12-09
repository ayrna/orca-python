
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
					"cv_metric": "ccr"
					}

	algorithms = {}


@ex.automain
def main(general_conf, configurations):


	fw_path = os.path.dirname(os.path.abspath(__file__)) + "/"

	interfaz = Utilities(general_conf, configurations)
	interfaz.runExperiment()
	interfaz.writeReport(fw_path)



