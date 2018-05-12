
import os
from sacred import Experiment

from Utilities import Utilities


ex = Experiment('Experiment Configuration')
#ex.observers.append(FileStorageObserver.create('my_runs', template='/custom/template.txt'))

@ex.config
def default_config():

	general_conf = {"basedir": "",
					"dataset": "",
					}

	algorithms = {}


@ex.automain
def main(general_conf, algorithms):


	api_path = os.path.dirname(os.path.abspath(__file__)) + "/"

	interfaz = Utilities(api_path, general_conf, algorithms)
	#interfaz.runExperiment()
	#interfaz.writeReport()



