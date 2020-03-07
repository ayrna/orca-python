#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "svm.h"
#include "svm_model_python.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

PyObject* predict(PyObject* features, struct svm_model *model, const int predict_probability)
{
	PyObject* predicted_labels = Py_BuildValue("[]"), *list_el = NULL;
	int feature_number, testing_instance_number;
	int instance_index;
	double **ptr_instance;
	struct svm_node *x;

	int i, j;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);

	feature_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(features, 0)))); /*features colums*/
	testing_instance_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(features))); /*features rows*/

	ptr_instance = Malloc(double*, testing_instance_number);
	for (i = 0; i < testing_instance_number; i++){
		ptr_instance[i] = Malloc(double, feature_number);
		for (j = 0; j < feature_number; j++){
			ptr_instance[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, i), j));
		}
	}
	
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double predict_label;
		
		for(i=0;i<feature_number;i++)
		{
			x[i].index = i+1;
			x[i].value = ptr_instance[instance_index][i];
		}
		x[feature_number].index = -1;
		
		if(svm_type == C_RNK ||
                       svm_type == SVORIM ||
		   svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR)
		{
			double res;
			svm_predict_values(model, x, &res);
			if(svm_type == ONE_CLASS)
				predict_label =  (res>0)?1:-1;
			else if (svm_type == C_RNK || svm_type == SVORIM){
				int j;
				predict_label = nr_class;
				for(j=1;j<nr_class; j++){
					if (res < model->rho[j]){
						predict_label =  j;
						j=nr_class+1;
					}
				}
			}
			else
				predict_label = res;
		}
		else
		{
			double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
			svm_predict_values(model, x, dec_values);
			
			int i;
			int *vote = (int *) malloc(sizeof(int)* nr_class);
			for(i=0;i<nr_class;i++)
				vote[i] = 0;
			int pos=0;
			for(i=0;i<nr_class;i++){
				int j;
				for(j=i+1;j<nr_class;j++)
				{
					if(dec_values[pos++] > 0)
						++vote[i];
					else
						++vote[j];
				}
			}

			int vote_max_idx = 0;
			for(i=1;i<nr_class;i++)
				if(vote[i] > vote[vote_max_idx])
					vote_max_idx = i;
			predict_label = model->label[vote_max_idx];

			free(vote);
			free(dec_values);
		}

		list_el = Py_BuildValue("d", predict_label);
		PyList_Append(predicted_labels, list_el);
		//PyList_Append increment the passed in PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
	}

	//Free memory
	free(x);
	for (i = 0; i < testing_instance_number; i++)
		free(ptr_instance[i]);
	free(ptr_instance);

	return predicted_labels;
}

void exit_with_help()
{
	printf(
		"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
	);
}

/* Interface function of Python*/
PyObject* run(PyObject* self, PyObject* args){
	int prob_estimate_flag = 0;
	struct svm_model *model;
	PyObject* predictedLabels = NULL;

	PyObject* features = NULL;
	PyObject* py_model = NULL;
	char* options = NULL;

	/*Parse arguments*/

	/*options is NULL terminated*/
	if (!PyArg_ParseTuple(args, "OOs", &features, &py_model, &options)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
		return NULL;
	}

	if(PyDict_Check(py_model))
	{
		/* parse options*/
		if(options[0] != '\0')
		{
			int i, argc = 1;
			char* argv[CMD_LEN/2];

			/* put options in argv[]*/
			if((argv[argc] = strtok(options, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				if(++i>=argc)
				{
					exit_with_help();
					PyErr_SetString(PyExc_SyntaxError, "Options syntax not correct!");
					return NULL;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					default:
						char str[128];
						snprintf(str, 128, "Unknown option: -%c", argv[i-1][1]);
						exit_with_help();
						PyErr_SetString(PyExc_ValueError, str);
						return NULL;
				}
			}
		}

		model = pythonToModel(py_model);
		if (model == NULL)
		{
			PyErr_SetString(PyExc_MemoryError, "Unable to translate python model to C");
			return NULL;
		}

		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model)==0)
			{
				svm_free_and_destroy_model(&model);
				PyErr_SetString(PyExc_ValueError, "Model does not support probabiliy estimates");
				return NULL;
			}
		}
		/*else
		{
			if(svm_check_probability_model(model)!=0)
				printf("Model supports probability estimates, but disabled in predicton.\n");
		}*/
		
		predictedLabels = predict(features, model, prob_estimate_flag);
		/* destroy model*/
		svm_free_and_destroy_model(&model);
	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "Model should be a dictionary!");
		return NULL;
	}

	return predictedLabels;
}

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svmPredictMethod[] = {
	{ "run", run, METH_VARARGS, "Predict labels" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvmpredict(void)
			{
			  Py_InitModule("svmpredict", svmPredictMethod);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svmpredictmodule = {
	    PyModuleDef_HEAD_INIT,
	    "svmpredict",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svmPredictMethod
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit_svmpredict(void){
			    return PyModule_Create(&svmpredictmodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
/******************/
