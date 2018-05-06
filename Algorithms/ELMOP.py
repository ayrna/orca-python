#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

def entrenar_rbf_total(train_set, test_set, hyper_parameters, clasification):

	X_train = train_set['inputs']
	X_test = test_set['inputs']
	y_train = train_set['outputs']
	y_test = test_set['outputs']


	#Estandarizamos los conjuntos de train y test
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	metrics = []

	sys.path.insert(0, '../') # Import modules from different directory
	import DSU


	#Probamos con cada una de las combinaciones
	for c in hyper_parameters['c']:
		for g in hyper_parameters['g']:

			# Entrenar el modelo SVM
			svm_model = svm.SVC(kernel='rbf',C=c,gamma=g)
			svm_model.fit(X_train, y_train)

			hyper_params = {"C": c, "G": g}

			train_metrics = {"MSE": 0, "CCR": (svm_model.score(X_train, y_train) * 100)}
			test_metrics = {"MSE": 0, "CCR": (svm_model.score(X_test, y_test) * 100)}

			conf_metrics = DSU.ParamMetrics(hyper_params, train_metrics, test_metrics)
			metrics.append(conf_metrics)


	return metrics


