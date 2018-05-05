#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2016

@author: pagutierrez
"""

# TODO Incluir todos los import necesarios
import click
import math
import time
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


# TODO incluir el resto de parámetros...
def entrenar_rbf_total(train_set, test_set, clasificacion, ratio_rbf, l2, eta, outputs):

	train_metrics = {}
	test_metrics = {}

	#np.random.seed(100)
	train_metrics["MSE"], test_metrics["MSE"], train_metrics["CCR"], test_metrics["CCR"] = entrenar_rbf(train_set, test_set, clasificacion, ratio_rbf, l2, eta, outputs)

	"""
	print("MSE de entrenamiento: %f" % train_mse)
	print("MSE de test: %f" % test_mse)
	if clasificacion:
		print("CCR de entrenamiento: %.2f%%" % train_ccr)
		print("CCR de test: %.2f%%" % test_ccr)
	"""

	return train_metrics, test_metrics

def entrenar_rbf(train_set, test_set, clasificacion, ratio_rbf, l2, eta, outputs):
	""" Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
		Una única ejecución.
		Recibe los siguientes parámetros:
		    - train_file: nombre del fichero de entrenamiento.
		    - test_file: nombre del fichero de test.
		    - clasificacion: True si el problema es de clasificacion.
		    - ratio_rbf: Ratio (en tanto por uno) de neuronas RBF con 
		      respecto al total de patrones.
		    - l2: True si queremos utilizar L2 para la Regresión Logística. 
		      False si queremos usar L1 (para regresión logística).
		    - eta: valor del parámetro de regularización para la Regresión 
		      Logística.
		    - outputs: número de variables que se tomarán como salidas 
		      (todas al final de la matriz).
		Devuelve:
		    - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
		      En el caso de clasificación, calcularemos el MSE de las 
		      probabilidades predichas frente a las objetivo.
		    - test_mse: Error de tipo Mean Squared Error en test. 
		      En el caso de clasificación, calcularemos el MSE de las 
		      probabilidades predichas frente a las objetivo.
		    - train_ccr: Error de clasificación en entrenamiento. 
		      En el caso de regresión, devolvemos un cero.
		    - test_ccr: Error de clasificación en test. 
		      En el caso de regresión, devolvemos un cero.
	"""

	train_inputs, train_outputs = train_set['inputs'], train_set['outputs']
	test_inputs, test_outputs = test_set['inputs'], test_set['outputs']

	#TODO: Obtener num_rbf a partir de ratio_rbf
	num_rbf = int(math.ceil(train_outputs.shape[0] * ratio_rbf))

	kmedias, distancias, centros = clustering(clasificacion, train_inputs, 
		                                      train_outputs, num_rbf)

	radios = calcular_radios(centros, num_rbf)

	matriz_r = calcular_matriz_r(distancias, radios)


	if not clasificacion:
		coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
	else:
		logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)


	"""
	TODO: Calcular las distancias de los centroides a los patrones de test
		  y la matriz R de test
	"""
	distancias_test = kmedias.transform(test_inputs)
	matriz_r_test = calcular_matriz_r(distancias_test, radios)

	if not clasificacion:
		"""
		TODO: Obtener las predicciones de entrenamiento y de test y calcular
		      el MSE
		"""

		salidas_train = np.dot(matriz_r, coeficientes)
		salidas_test = np.dot(matriz_r_test, coeficientes)
		
		train_mse = ((train_outputs - salidas_train)**2).mean()
		test_mse = ((test_outputs - salidas_test)**2).mean()
		
		train_ccr = test_ccr = 0.0
		
	else:
		"""
		TODO: Obtener las predicciones de entrenamiento y de test y calcular
		      el CCR. Calcular también el MSE, comparando las probabilidades 
		      obtenidas y las probabilidades objetivo
		"""
		#print matriz_r.shape, coeficientes.shape
		salidas_train = logreg.predict(matriz_r)
		salidas_test = logreg.predict(matriz_r_test)
		
		train_mse = ((train_outputs - salidas_train)**2).mean()
		test_mse = ((test_outputs - salidas_test)**2).mean()
		
		train_ccr = (np.count_nonzero(salidas_train == train_outputs) / float(train_outputs.shape[0]))*100
		test_ccr = (np.count_nonzero(salidas_test == test_outputs) / float(test_outputs.shape[0]))*100

	return train_mse, test_mse, train_ccr, test_ccr


def lectura_datos(fichero_train, fichero_test, outputs, clasificacion):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - outputs: número de variables que se tomarán como salidas 
              (todas al final de la matriz).
            - clasificacion: booleano que indica el tipo de problema.
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """

    #TODO: Completar el código de la función
    
    train = pd.read_csv(fichero_train, header=None)
    test = pd.read_csv(fichero_test, header=None)
    
    if clasificacion:
        
        train_inputs = train.values[:,0:(-1)]
        train_outputs = train.values[:,(-1)]
        
        test_inputs = test.values[:,0:(-1)]
        test_outputs = test.values[:,(-1)]
    
    # Existen distintas columnas para las variables de salida
    else:
        
        train_inputs = train.values[:,0:(-outputs)]
        train_outputs = train.values[:,(-outputs):]
        
        test_inputs = test.values[:,0:(-outputs)]
        test_outputs = test.values[:,(-outputs):]

    
    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir, aprox., num_rbf/num_clases
        patrones por cada clase. Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """
    
    #TODO: Completar el código de la función

    sss = StratifiedShuffleSplit(n_splits=1, test_size=num_rbf)
    train_array, centroid_array = next(iter(sss.split(X=train_inputs, y=train_outputs)))
    centroides = train_inputs[centroid_array]
    
    return centroides

def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
            - outputs: número de salidas del problema.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """
    
    # Seleccionamos los puntos desde los cuales inicializaremos el algoritmo del K-Medias
    if clasificacion:
        centroides = inicializar_centroides_clas(train_inputs, train_outputs, num_rbf)
        
    else:
        centroides = train_inputs[np.random.choice(np.arange(1, train_outputs.shape[0]), size=(num_rbf), replace=False)]

    #TODO: Completar el código de la función
    
    kmedias = KMeans(n_clusters=num_rbf, init=centroides, n_init=1, max_iter=500).fit(train_inputs)
    centros = kmedias.cluster_centers_
    distancias = kmedias.transform(train_inputs)


    return kmedias, distancias, centros


def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    #TODO: Completar el código de la función
    
    distancia_centros = squareform(pdist(centros))    
    radios = (np.sum(distancia_centros, axis=0) / (2*(num_rbf - 1)) )
    
    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """

    #TODO: Completar el código de la función
    a = np.exp( -(distancias**2) / (2*(radios**2)) )
    matriz_r = np.ones((distancias.shape[0], distancias.shape[1]+1))
    matriz_r[:,:-1] = a
    
    return matriz_r


def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """

    #TODO: Completar el código de la función
    
    if ( matriz_r.shape[0] == (matriz_r.shape[1] - 1) ):
        
        coeficientes = np.dot( np.linalg.pinv(matriz_r), train_outputs )
    
    elif (matriz_r.shape[0] < (matriz_r.shape[1] - 1) ):
        
        #Hacer cosas para reducir la dimensionalidad hasta que tengan el mismo tamaño
        pass
    
    else:
        
        #Por si hay combinaciones lineales de atributos en la base de datos, usamos pinv
        #en vez de linalg.inv, porque sino arroja malos resultados
        matriz_r_MP = np.dot( np.linalg.pinv(np.dot(matriz_r.T, matriz_r)), matriz_r.T )
        coeficientes = np.dot(matriz_r_MP, train_outputs)
        
    return coeficientes


def logreg_clasificacion(matriz_r, train_outputs, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    #TODO: Completar el código de la función
    if(l2):
        aux_str = 'l2'
    else:
        aux_str = 'l1'
    
    logreg = LogisticRegression(penalty=aux_str, C=(1/eta), solver='liblinear')
    logreg.fit(matriz_r, train_outputs)
    
    return logreg



if __name__ == "__main__":
    entrenar_rbf_total()



