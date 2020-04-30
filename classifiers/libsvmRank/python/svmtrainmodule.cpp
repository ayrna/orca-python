#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "svm.h"
#include "svm_model_python.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');\n"
	"libsvm_options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC\n"
	"	1 -- nu-SVC\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR\n"
	"	4 -- nu-SVR\n"
	"   5 -- C-RNK\n"
	"   6 -- SVORIM\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"       0 -- linear: u'*v\n"
	"       1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"       2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"       3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"       4 -- stump: -|u-v|_1 + coef0\n"
	"       5 -- perceptron: -|u-v|_2 + coef0\n"
	"       6 -- laplacian: exp(-gamma*|u-v|_1)\n"
	"       7 -- exponential: exp(-gamma*|u-v|_2)\n"
	"		8 -- precomputed kernel (kernel values in training_instance_matrix)"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n : n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
}

int parse_command_line(char* options);
const char* read_problem(PyObject* label_vec, PyObject* instance_mat);
void do_cross_validation();

/* svm arguments*/
struct svm_parameter param;		/* set by parse_command_line*/
struct svm_problem prob;		/* set by read_problem*/
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;

/* Interface function of Python*/
PyObject* run(PyObject* self, PyObject* args){
	const char *error_msg;

	/* fix random seed to have same results for each run*/
	/* (for cross validation and probability estimation)*/
	srand(1);

	/* Transform the input Matrix to libsvm format*/
	const char *err;

	PyObject* labels = NULL;
	PyObject* features = NULL;
	char* options = NULL;

	/*Parse arguments*/

	/*options is NULL terminated*/
	if (!PyArg_ParseTuple(args, "OOs", &labels, &features, &options)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
		return NULL;
	}

	if(parse_command_line(options)){
		//exit_with_help();
		svm_destroy_param(&param);
		PyErr_SetString(PyExc_SyntaxError, "Options syntax not correct!");
		return NULL;
	}

	err = read_problem(labels, features);

	/* svmtrain's original code*/
	error_msg = svm_check_parameter(&prob, &param);

	if(err || error_msg)
	{
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		if (error_msg != NULL)
			PyErr_SetString(PyExc_ValueError, error_msg);
		else
			PyErr_SetString(PyExc_ValueError, err);
		return NULL;
	}
	PyObject*  py_model;

	model = svm_train(&prob, &param);

	if (model == NULL){
		PyErr_SetString(PyExc_ValueError, "The trained model is null");
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
		return NULL;
	}
	
	py_model = modelToPython(model);

	svm_free_and_destroy_model(&model);

	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);

	return py_model;
}

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svmTrainMethod[] = {
	{ "run", run, METH_VARARGS, "Fits a model" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvmtrain(void)
			{
			  Py_InitModule("svmtrain", svmTrainMethod);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svmtrainmodule = {
	    PyModuleDef_HEAD_INIT,
	    "svmtrain",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svmTrainMethod
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit_svmtrain(void){
			    return PyModule_Create(&svmtrainmodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
/******************/

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else if(param.svm_type == C_RNK ||
		param.svm_type == SVORIM)

	{
		int nloss = 0;
	  	for(i=0;i<prob.l;i++){
			if(target[i] == prob.y[i])
				++total_correct;
			nloss += abs((int)(target[i]) - (int)(prob.y[i]));
		}
		printf("Cross Validation Average absolute error = %g\n",(double)nloss/prob.l);

	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

/* nrhs should be 3*/
int parse_command_line(char* options)
{
	int i, argc = 1;
	char *argv[CMD_LEN/2];

	/* default values*/
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	/* 1/num_features*/
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	/* put options in argv[]*/
	if((argv[argc] = strtok(options, " ")) != NULL)
		while((argv[++argc] = strtok(NULL, " ")) != NULL)
			;

	/* parse options*/
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q')	/* since option -q has no parameter*/
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				svm_set_print_string_function(&print_null);
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					printf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				printf("Unknown option -%c\n", argv[i-1][1]);
				return 1;
		}
	}

	return 0;
}

/* read in a problem (in svmlight format)*/
const char* read_problem(PyObject* label_vec, PyObject* instance_mat)
{
	int i, j, k;
	int elements, max_index, sc, label_vector_row_num;
	double **samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	sc = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(instance_mat, 0)))); /*instance_mat colums*/

	elements = 0;
	/* the number of instance*/
	prob.l = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(instance_mat))); /*instance_mat rows*/
	label_vector_row_num = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(label_vec)));

	if(label_vector_row_num!=prob.l)
	{
		return "Length of label vector does not match # of instances";
	}

	//PyObjects are transformed to C arrays
	labels = Malloc(double, label_vector_row_num);
	for (i = 0; i < label_vector_row_num; i++){
		labels[i] = PyFloat_AsDouble(PyList_GetItem(label_vec, i));
	}

	samples = Malloc(double*, prob.l);
	for (i = 0; i < prob.l; i++){
		samples[i] = Malloc(double, sc);
		for (j = 0; j < sc; j++){
			samples[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(instance_mat, i), j));
		}
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(i = 0; i < prob.l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[i][k] != 0)
					elements++;
			/* count the '-1' element*/
			elements++;
		}
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[i][k] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[i][k];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	//Free memory
	free(labels);
	for(i = 0; i < prob.l; i++)
		free(samples[i]);
	free(samples);

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				return "Wrong input format: sample_serial_number out of range";
			}
		}

	return NULL;
}
