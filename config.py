from sacred import Experiment

from utilities import Utilities

ex = Experiment('Experiment Configuration')

@ex.config
def default_config():

	# Giving default values
	general_conf = {"basedir": "",
					"dataset": "",
					"input_preprocessing": "",
					"hyperparam_cv_nfolds": 3,
					"jobs": 1,
					"metrics": "ccr",
					"cv_metric": "ccr",
					"output_folder": "my_runs/"
					}

	configurations = {}

@ex.automain
def main(general_conf, configurations):

	if not general_conf['basedir'] or not general_conf['datasets']:

		raise RuntimeError('A dataset has to be defined to run this program.\n' +
							'For more information about using this framework, please refer to the README.')

	if not configurations:

		raise RuntimeError('No configuration was defined.\n' + 
							'For more information about using this framework, please refer to the README.')


	interface = Utilities(general_conf, configurations)
	interface.run_experiment()
	interface.write_report()
