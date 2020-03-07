/*******************************************************************************\

	svorex_predict.c
		
	entry function for the python program predict function.

\*******************************************************************************/
#include "Python.h"

#include <stdio.h>
#include <stdlib.h>
#ifndef __MACH__
    #include <malloc.h>
#endif

#include "smo.h"
#include "smo_model_python.h"

PyObject* run(PyObject* self, PyObject* args)
{
	smo_Settings* model = NULL;
	Data_List* testdata = NULL;
	Data_Node* testnode = NULL;
	int feature_number, testing_instance_number, i, j;
	PyObject* predicted_labels = NULL, *list_el = NULL;
   	//Python parameters
   	PyObject* features = NULL;
	PyObject* py_model = NULL;

   	/*Options is NULL terminated*/
   	if (!PyArg_ParseTuple(args, "OO", &features, &py_model)){
		PyErr_SetString(PyExc_RuntimeError, "Unable to parse arguments");
   		return NULL;
   	}

	if(!PyDict_Check(py_model))
	{
		PyErr_SetString(PyExc_TypeError, "Model should be a dictionary!");
		return NULL;
	}

	//Get the test data from python
	testdata = (Data_List*) malloc(sizeof(Data_List));
	if(testdata == NULL){
		return NULL;
	}
	testdata->front = NULL;
	testdata->rear = NULL;

	feature_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(PyList_GetItem(features, 0)))); /*features colums*/
	testing_instance_number = (int) PyLong_AsLong(PyLong_FromSsize_t(PyList_Size(features))); /*features rows*/
	
	for(i=0; i<testing_instance_number; i++){
		testnode = (Data_Node*) malloc(sizeof(Data_Node));
		if(testnode == NULL){
			free(testdata);
			return NULL;
		}
		testnode->point = (double*) malloc(feature_number * sizeof(double));
			if(testnode->point == NULL){
				free(testnode);
				free(testdata);
				return NULL;
			}
		testnode->next = NULL;

		for(j=0; j<feature_number; j++)
			testnode->point[j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(features, i), j));

		Add_Data_List(testdata, testnode);
	}
	
	//Get the model from python
	model = pythonToModel(py_model);
	if (model == NULL)
	{
		PyErr_SetString(PyExc_MemoryError, "Unable to translate python model to C");
		free(testdata);
		return NULL;
	}

	//Just copy the data dimension used for train (Python must check if the data is correct before calling this process)
	testdata->dimen = model->pairs->dimen;
	
	//Predict process
	svm_predict_Python (testdata, model);

	predicted_labels = Py_BuildValue("[]");
	testnode = testdata->front ;
	while (testnode!=NULL){
		list_el = Py_BuildValue("d", testnode->guess);
		PyList_Append(predicted_labels, list_el);
		//PyList_Append increment the passed in PyObjects references so is necesary
		//to decrement them in order to let python free memory when the model 
		//is not longer needed
		Py_DECREF(list_el);
		testnode = testnode->next ;
	}
	
	// free memory then exit
	if ( NULL != model )
	{
		if (NULL != model->ard)
			free(model->ard) ;
		if (ORDINAL == model->pairs->datatype)
		{
			if (NULL != model->biasj)
				free(model->biasj) ;
		}

		free(model->alpha) ;

		free (model) ;
		model = NULL ;
	}
	free(testdata);

	return predicted_labels;
}
//end of main.c 

/*Python module init*/
#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

static PyMethodDef svorexPredictMethod[] = {
	{ "run", run, METH_VARARGS, "Predict labels" },
	{ NULL, NULL, 0, NULL }
};

#ifndef IS_PY3K /*For Python 2*/
	#ifdef __cplusplus
		extern "C" {
	#endif
			DL_EXPORT(void) initsvorex(void)
			{
			  Py_InitModule("svorexpredict", svorexPredictMethod);
			}
	#ifdef __cplusplus
		}
	#endif
#else /*For Python 3*/
	static struct PyModuleDef svorexpredictmodule = {
	    PyModuleDef_HEAD_INIT,
	    "svorexpredict",   /* name of module */
	    NULL, 		 /* module documentation, may be NULL */
	    -1,       	 /* size of per-interpreter state of the module,
	                 or -1 if the module keeps state in global variables. */
	    svorexPredictMethod
	};

	#ifdef __cplusplus
		extern "C" {
	#endif
			PyMODINIT_FUNC
			PyInit_svorexpredict(void){
			    return PyModule_Create(&svorexpredictmodule);
			}
	#ifdef __cplusplus
		}
	#endif
#endif
/******************/
