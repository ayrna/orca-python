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


@ex.automain
def main(general_conf, configurations):

	interface = Utilities(general_conf, configurations)
	interface.run_experiment()
	interface.write_report()
