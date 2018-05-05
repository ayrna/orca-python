
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

	algorithm_parameters =	{
							}


	algorithm_hyper_parameters =	{
									}

@ex.automain
def main(general_conf, algorithm_parameters, algorithm_hyper_parameters):


	api_path = os.path.dirname(os.path.abspath(__file__)) + "/"

	interfaz = Utilities(api_path, general_conf, algorithm_parameters, algorithm_hyper_parameters)
	interfaz.runExperiment(general_conf, algorithm_parameters, algorithm_hyper_parameters)
	#interfaz.writeReport()



