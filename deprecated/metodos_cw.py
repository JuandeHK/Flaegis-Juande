import numpy as np
import scipy.special
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
#from rpy2.robjects import pandas2ri
#pandas2ri.activate()
#from FuncionesAux import get_server_id

from scipy.stats import ks_2samp
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.preprocessing import LabelEncoder
import fastcluster
from scipy.cluster.hierarchy import dendrogram, fcluster
#from FedDetector.metodos_cw import mean_without_diagonal
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os


def fourier_abs(arrays):
    fourier_array = []
    iteraciones_array = [np.nditer(a) for a in arrays]
    np_ite = np.array(iteraciones_array)
    idx = arrays[0].size
    for i in range(idx):
        values = [np_ite[j][i] for j in range(len(iteraciones_array))]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        # Frequencies
        freq = np.fft.fftfreq(values.shape[-1])

        # Index of highest frequency
        index = np.argmax(np.abs(freq))

        # Value of highest frequency in original data
        result = values[index]

        fourier_array.append(result)
    return np.array(fourier_array).reshape(arrays[0].shape)

from scipy.stats import skew, trim_mean


def trimmed_mean_abs(arrays, trim):
    trimmed_mean_array = []
    iteraciones_array = [np.nditer(a) for a in arrays]
    np_ite = np.array(iteraciones_array)
    idx = arrays[0].size
    for i in range(idx):
        values = [np_ite[j][i] for j in range(len(iteraciones_array))]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        trimmed_m = trim_mean(values, trim)
        trimmed_mean_array.append(trimmed_m)
    return np.array(trimmed_mean_array).reshape(arrays[0].shape)

import scipy.stats as stats
import numpy


def weighted_median(arrays, weights):  # el bueno
    weighted_medians = []
    for i in range(arrays[0].shape[0]):
        for j in range(arrays[0].shape[1]):
            values = [array[i][j] for array in arrays]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            weights = np.array(weights)[indices]
            cumulative_weights = np.cumsum(weights)
            median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
            weighted_medians.append(values[median_index])
    return np.array(weighted_medians).reshape(arrays[0].shape)

def weighted_median_1d(arrays, weights):
    weighted_medians = []
    for i in range(len(arrays[0])):
        values = [array[i] for array in arrays]
        indices = np.argsort(values)
        values = np.array(values)[indices]

        weights = np.array(weights)[indices]
        #cumulative_weights = np.cumsum(weights)[indices]
        cumulative_weights = np.cumsum(weights)
        median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
        weighted_medians.append(values[median_index])
    return np.array(weighted_medians)

def w_median(arrays, weights):

    try:
        return weighted_median(arrays, weights)
    except IndexError:
        return weighted_median_1d(arrays, weights)

def w_median_abs(arrays, weights):
    weighted_medians = []
    iteraciones_array = [np.nditer(a) for a in arrays]
    np_ite = np.array(iteraciones_array)
    idx = arrays[0].size
    for i in range(idx):
        values = [np_ite[j][i] for j in range(len(iteraciones_array))]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        # Frequencies
        weights = np.array(weights)[indices]
        # cumulative_weights = np.cumsum(weights)[indices]
        cumulative_weights = np.cumsum(weights)
        median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
        weighted_medians.append(values[median_index])

    return np.array(weighted_medians).reshape(arrays[0].shape)


def ks_proportion(sample):
    total = []
    for i in range(1000):
        n_random = 30
        random_indices = np.random.choice(len(sample), size=n_random, replace=False)
        random_points = sample[random_indices]
        generated_points = np.delete(sample, random_indices)



        ks_stat, p_value = ks_2samp(generated_points, random_points)
        # print(ks_stat)
        # print(p_value)

        # Set a significance level
        significance_level = 0.05

        # Identify the random points
        if p_value < significance_level:
            # print("There are random points in the data.")
            total.append(1)
        else:
            # print("There are no random points in the data.")
            total.append(0)
        rtn = sum(total)/len(total)
        return rtn


def skeweness_abs(arrays):
    means_array = []
    iteraciones_array = [np.nditer(a) for a in arrays]
    np_ite = np.array(iteraciones_array)
    idx = arrays[0].size
    for i in range(idx):
        values = [np_ite[j][i] for j in range(len(iteraciones_array))]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        values = (values - np.mean(values)) / np.std(values)
        prop = ks_proportion(values)
        means_array.append(prop)

    return np.mean(means_array)

from sklearn.metrics.pairwise import cosine_similarity
def similitud_coseno_malo(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                cosine_similarity_matrix[i, j] = \
                cosine_similarity(arrays[i].reshape(1, -1), arrays[j].reshape(1, -1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 1

    return cosine_similarity_matrix

def similitud_coseno(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                if arrays[i].ndim > 1:
                    a = arrays[i].reshape(1, -1)
                    b = arrays[j].reshape(1, -1)
                else:
                    a = arrays[i]
                    b = arrays[j]

                cosine_similarity_matrix[i, j] = \
                cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 0

    return cosine_similarity_matrix


def similitud_coseno_fft(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                #print(arrays[i])
                arrays[i]= arrays[i].reshape(1, -1)
                arrays[j]= arrays[j].reshape(1, -1)

                arrays[i] = np.fft.fftfreq(arrays[i].shape[-1])
                arrays[j] = np.fft.fftfreq(arrays[j].shape[-1])

                cosine_similarity_matrix[i, j] = \
                cosine_similarity(arrays[i].reshape(1, -1), arrays[j].reshape(1, -1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 1

    return cosine_similarity_matrix




def submatrix_from_index_pairs(matrix, lista, clientes_1, clientes_2):

    sorter = np.argsort(lista)
    row_indices = sorter[np.searchsorted(lista, clientes_1, sorter=sorter)]
    col_indices = sorter[np.searchsorted(lista, clientes_2, sorter=sorter)]
    submatrix = matrix[np.ix_(row_indices, col_indices)]
    return submatrix


def mean_without_diagonal(matrix):
    n = matrix.shape[0]
    diagonal_indices = np.arange(n)
    non_diagonal_indices = np.logical_not(np.eye(n, dtype=bool))
    non_diagonal_elements = matrix[non_diagonal_indices].flatten()
    mean = np.mean(non_diagonal_elements)
    return mean

import tensorflow as tf
def get_gradient(x_train, y_train, model):
    x_sample = x_train[:1]  # Ejemplo de entrada
    y_sample = y_train[:1]  # Ejemplo de etiqueta

    # Obtiene los gradientes sin hacer predicciones explícitas
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)  # Para rastrear las variables del modelo
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_sample, model(x_sample))

    gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

def get_gradient_norm(x_train, y_train, model):
    x_sample = x_train[:1]  # Ejemplo de entrada
    y_sample = y_train[:1]  # Ejemplo de etiqueta

    # Obtiene los gradientes sin hacer predicciones explícitas
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)  # Para rastrear las variables del modelo
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_sample, model(x_sample))

    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_direction = [grad / tf.norm(grad) for grad in gradients]
    return gradient_direction

def get_gradient_norm(x_train, y_train, model):
    x_sample = x_train[:1]  # Ejemplo de entrada
    y_sample = y_train[:1]  # Ejemplo de etiqueta

    # Obtiene los gradientes sin hacer predicciones explícitas
    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)  # Para rastrear las variables del modelo
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_sample, model(x_sample))

    gradients = tape.gradient(loss, model.trainable_variables)
    gradient_direction = [grad / tf.norm(grad) for grad in gradients]
    return gradient_direction


import pywt
def apply_dwt(matrix, wavelet='haar', level=1):
    # Perform 2D DWT
    coeffs = pywt.wavedec2(matrix, wavelet, level=level)
    return coeffs

def similitud_coseno_dwt(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                #print(arrays[i])
                arrays_1= apply_dwt(arrays[i])
                arrays_1 = arrays_1[0]
                arrays_2= apply_dwt(arrays[j])
                arrays_2 = arrays_2[0]

                cosine_similarity_matrix[i, j] = \
                cosine_similarity(arrays_1.reshape(1, -1), arrays_2.reshape(1, -1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 1

    return cosine_similarity_matrix

from sklearn import preprocessing

def custom_sax(ts, alphabet_size):
    """
    Perform Symbolic Aggregate Approximation (SAX) on a time series with equal-width bins.

    Args:
    ts (numpy.ndarray): The time series data.
    alphabet_size (int): The number of symbols in the SAX alphabet.

    Returns:
    list: A list of SAX words representing the time series.
    """
    if len(ts) < alphabet_size:
        raise ValueError("Alphabet size is too large for the time series.")

    # Calculate breakpoints for the alphabet
    breakpoints = np.linspace(np.min(ts), np.max(ts), alphabet_size + 1)

    # Convert the time series to a sequence of symbols
    symbols = []
    for val in ts:
        for i in range(1, len(breakpoints)):
            if val <= breakpoints[i]:
                symbols.append(chr(97 + i - 1))  # Convert to letters (a, b, c, ...)
                break

    return symbols

def similitud_coseno_sax(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    n_bins = 45
    #sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
    #X_sax = sax.fit_transform(prueba)
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                """
                a = arrays[i].reshape(1, -1)
                # norma = np.linalg.norm(a)
                # a = a / norma
                #arrays_1 = sax.fit_transform(a)
                arrays_1 = custom_sax(a)
                arrays_1 = LabelEncoder().fit_transform(arrays_1[0])

                a = arrays[j].reshape(1, -1)
                # norma = np.linalg.norm(a)
                # a = a / norma
                #arrays_2 = sax.fit_transform(a)
                arrays_2 = custom_sax(a)
                arrays_2 = LabelEncoder().fit_transform(arrays_2[0])
                """
                #print(arrays[i])
                if arrays[i].ndim > 1:
                    a =arrays[i].reshape(1,-1)
                    #norma = np.linalg.norm(a)
                    #a = a / norma
                    arrays_1 = custom_sax(a, n_bins)
                    arrays_1 = LabelEncoder().fit_transform(arrays_1)
                else:
                    a = arrays[i]
                    #norma = np.linalg.norm(a)
                    #a = a / norma
                    arrays_1 = custom_sax(a, n_bins)
                    arrays_1 = LabelEncoder().fit_transform(arrays_1)

                if arrays[j].ndim > 1:
                    a = arrays[j].reshape(1, -1)
                    #norma = np.linalg.norm(a)
                    #a = a / norma
                    arrays_2 = custom_sax(a, n_bins)
                    arrays_2 = LabelEncoder().fit_transform(arrays_2)
                else:
                    a = arrays[j]
                    #norma = np.linalg.norm(a)
                    #a = a / norma
                    arrays_2 = custom_sax(a, n_bins)
                    arrays_2 = LabelEncoder().fit_transform(arrays_2)

                #arrays_1 = arrays_1 / np.linalg.norm(arrays_1)
                #arrays_2 = arrays_2 / np.linalg.norm(arrays_2)

                cosine_similarity_matrix[i, j] = \
                cosine_similarity(arrays_1.reshape(1,-1), arrays_2.reshape(1,-1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 0

    return cosine_similarity_matrix

from pyts.approximation import PiecewiseAggregateApproximation

def similitud_coseno_paa(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    #n_bins = 26
    paa = PiecewiseAggregateApproximation()
    #X_sax = sax.fit_transform(prueba)
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                a = arrays[i].reshape(1, -1)
                # norma = np.linalg.norm(a)
                # a = a / norma
                arrays_1 = paa.transform(a)
                #arrays_1 = LabelEncoder().fit_transform(arrays_1[0])

                a = arrays[j].reshape(1, -1)
                # norma = np.linalg.norm(a)
                # a = a / norma
                arrays_2 = paa.transform(a)
                #arrays_2 = LabelEncoder().fit_transform(arrays_2[0])

                cosine_similarity_matrix[i, j] = \
                cosine_similarity(arrays_1.reshape(1,-1), arrays_2.reshape(1,-1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 0

    return cosine_similarity_matrix
"""
from fastdtw import fastdtw
def similitud_dwt_sax(arrays):
    num_matrices = len(arrays)
    similarity_matrix = np.zeros((num_matrices, num_matrices))
    n_bins = 26
    sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
    #X_sax = sax.fit_transform(prueba)
    for i in range(num_matrices):
        for j in range(num_matrices):
            a = arrays[i].reshape(1, -1)
            # norma = np.linalg.norm(a)
            # a = a / norma
            arrays_1 = sax.fit_transform(a)
            arrays_1 = LabelEncoder().fit_transform(arrays_1[0])

            a = arrays[j].reshape(1, -1)
            # norma = np.linalg.norm(a)
            # a = a / norma
            arrays_2 = sax.fit_transform(a)
            arrays_2 = LabelEncoder().fit_transform(arrays_2[0])

            similarity_matrix[i, j],_ = fastdtw(arrays_1, arrays_2)


    return similarity_matrix
"""
#import rpy2.robjects as robjects
"""
def beats(matriz):
    if matriz.ndim == 1:
        cadena_r = 'c(' + ', '.join(map(str, matriz)) + ')'
    else:
        lista_aplanada = [elemento for fila in matriz for elemento in fila]
        cadena_r = 'c(' + ', '.join(map(str, lista_aplanada)) + ')'
    robjects.r.source('/home/enrique/flower/FedDetector/beats (1).R')
    vector_r = robjects.r(cadena_r)
    resultado = robjects.r.BEATS(vector_r, 5, 25)
    return resultado

def similitud_coseno_beats(arrays):
    num_matrices = len(arrays)
    cosine_similarity_matrix = np.zeros((num_matrices, num_matrices))
    for i in range(num_matrices):
        for j in range(num_matrices):
            try:
                if arrays[i].ndim > 1:
                    a = arrays[i].reshape(1, -1)
                    arrays_1 = np.array(beats(a))
                else:
                    a = arrays[i]
                    arrays_1 = np.array(beats(a))

                if arrays[j].ndim > 1:
                    a = arrays[j].reshape(1, -1)
                    arrays_2 = np.array(beats(a))
                else:
                    a = arrays[j]
                    arrays_2 = np.array(beats(a))

                cosine_similarity_matrix[i, j] = \
                    cosine_similarity(arrays_1.reshape(1, -1), arrays_2.reshape(1, -1))[0, 0]
            except ValueError:
                cosine_similarity_matrix[i, j] = 0

"""
def apply_Kmeans(distances):

    # Convert distances to similarities (you can adjust the formula as needed)
    #similarities = np.exp(-distances ** 2)

    # Specify the number of clusters (K)
    K = 2  # Change this to your desired number of clusters

    # Initialize the K-means clustering algorithm
    kmeans = KMeans(n_clusters=K)#, algorithm='elkan', tol=1e-10)

    # Fit the model to your data (similarities)
    #kmeans.fit(similarities)
    kmeans.fit(distances)

    # Get cluster labels for each data point
    cluster_labels = kmeans.labels_

    return cluster_labels

def detector_clientes_maliciosos_fastcluster(matriz, clientes, threshold):
    distances = 1 - matriz
    clustering = fastcluster.linkage(distances, method='average')

    # Visualizar el dendrograma
    dendrogram_data = dendrogram(clustering)

    cluster_labels = fcluster(clustering, threshold, criterion='distance')
    valores = np.unique(cluster_labels)

    if len(valores) > 1:
        clusters = apply_Kmeans(distances)
        indexes = np.where(clusters==0)
        indx2 = np.where(clusters==1)
        matriz_1 = np.take(clientes, indexes[0])
        matriz_2 = np.take(clientes, indx2[0])
        if len(matriz_1)>len(matriz_2):
            print("clientes malos: ", sorted(matriz_2))
        else:
            print("clientes malos: ", sorted(matriz_1))

    else:
        print("No hay clientes maliciosos")
import hdbscan
def detector_clientes_maliciosos_hdbscan(matriz, clientes):
    print("HDBSCAN")
    distances = matriz
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2)
    clusterer.fit(distances)
    print("Etiquetas clustering: ",clusterer.labels_)
    print(apply_Kmeans(distances))
    valores = np.unique(clusterer.labels_)
    if len(valores) > 1:
        clusters = apply_Kmeans(distances)
        indx1 = np.where(clusters == 0)
        indx2 = np.where(clusters == 1)
        matriz_1 = np.take(clientes, indx1[0])
        matriz_2 = np.take(clientes, indx2[0])
        print(len(matriz_1))
        if len(matriz_1) > len(matriz_2):
            print("clientes malos: ", sorted(matriz_2))
            return sorted(matriz_2)
        else:
            print("clientes malos: ", sorted(matriz_1))
            return sorted(matriz_1)
    else:
        print("No hay clientes maliciosos")
        return []

def detector_clientes_maliciosos_hierarchical(matriz, clientes):
    distances = matriz
    linkage_matrix = linkage(distances, method='average')  # You can choose a different linkage method
    dendrogram(linkage_matrix)

    # plt.show()

    # Cut the dendrogram to obtain clusters
    max_d = 0.6  # Adjust the distance threshold as needed
    cluster_labels = fcluster(linkage_matrix, t=max_d, criterion='distance')
    print("Etiquetas clustering: ",cluster_labels)
    print(apply_Kmeans(distances))
    valores = np.unique(cluster_labels)
    if len(valores) > 1:
        clusters = apply_Kmeans(distances)
        indx1 = np.where(clusters == 0)
        indx2 = np.where(clusters == 1)
        matriz_1 = np.take(clientes, indx1[0])
        matriz_2 = np.take(clientes, indx2[0])
        print(len(matriz_1))
        if len(matriz_1) > len(matriz_2):
            print("clientes malos: ", sorted(matriz_2))
            return sorted(matriz_2)
        else:
            print("clientes malos: ", sorted(matriz_1))
            return sorted(matriz_1)
    else:
        print("No hay clientes maliciosos")
        return []

def clasificacion_final(clusters, clientes):
    indx1 = np.where(clusters == 0)
    indx2 = np.where(clusters == 1)
    matriz_1 = np.take(clientes, indx1[0])
    matriz_2 = np.take(clientes, indx2[0])
    print(len(matriz_1))
    if len(matriz_1) > len(matriz_2):
        print("clientes malos: ", sorted(matriz_2))
        return sorted(matriz_2)
    else:
        print("clientes malos: ", sorted(matriz_1))
        return sorted(matriz_1)


from sklearn.cluster import SpectralClustering
from scipy.linalg import eigh
from sklearn.mixture import GaussianMixture
#from sklearn_extra.cluster import KMedoids
from sklearn.cluster import OPTICS

def detector_clientes_maliciosos_spectral(similarity_matrix, threshold, clientes):
    print("Spectral")
    D = np.diag(np.sum(similarity_matrix, axis=1))
    L = D - similarity_matrix

    eigenvalues = np.linalg.eigvals(similarity_matrix)
    eigenvalues.sort()
    eigengap = np.diff(eigenvalues)
    k = eigengap.argmax() + 1  # Adding 1 to the index for 0-based indexing

    # Perform spectral clustering with the estimated number of clusters
    #spectral_clustering = SpectralClustering(n_clusters=k, affinity='precomputed')

    """
    # Compute eigenvalues and eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = eigh(L)

    # Sort eigenvalues in ascending order
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]

    significant_drop_threshold = threshold
    num_clusters = sum(eigenvalues < significant_drop_threshold)
    """
    # Perform spectral clustering with the estimated number of clusters
    spectral_clustering = SpectralClustering(n_clusters=k, affinity='precomputed')
    #similarity_matrix = np.nan_to_num(similarity_matrix)
    contains_nan = np.isinf(similarity_matrix).any()

    if contains_nan:
        print("The matrix contains NaN values.")
    else:
        print("The matrix does not contain NaN values.")

    cluster_labels = spectral_clustering.fit_predict(similarity_matrix)

    #print("Etiquetas clustering: ", cluster_labels)

    #print(apply_Kmeans(similarity_matrix))
    valores = np.unique(cluster_labels)
    with open(os.getcwd() + "/nodos_comprometidos.npy", 'rb') as f:
        malos = np.load(f)
    if True:#len(valores) > 1:

        """
        distances = 1 - similarity_matrix
        # Aplicar HDBSCAN a la matriz de distancias
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=2)
        clusters = clusterer.fit_predict(distances)
        print("HDBSCAN: ")
        clasificacion_final(clusters, clientes)
        print("------------ ")
        
        
        spectral_clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
        clusters = spectral_clustering.fit_predict(similarity_matrix)
        print("SpectralClustering: ")
        clasificacion_final(clusters, clientes)
        print("------------ ")
        """

        clusters = apply_Kmeans(similarity_matrix)
        #print("K-means: ")
        #clasificacion_final(clusters, clientes)
        print("------------ ")
        """
        _,clusters = inverse_distance_weighted_kmeans(similarity_matrix)
        print("K-means: ")
        clasificacion_final(clusters, clientes)
        print("------------ ")
        """
        graficas_clustering(similarity_matrix, clusters, clientes, threshold)
        return clasificacion_final(clusters, clientes)

    else:
        print("No hay clientes maliciosos")
        return []


def get_norms(weights):
    normas = []
    for i in range(len(weights)):
        if np.array(weights[i]).ndim == 1:
            #if np.linalg.norm(weights[i])==0:
            #    normas.append(weights[i])
            #else:
            normas.append(np.linalg.norm(weights[i]))
        else:
            #if np.linalg.norm(weights[i].reshape(1,-1))==0:
            #    normas.append(weights[i].reshape(1,-1))
            #else:
            normas.append(np.linalg.norm(weights[i].reshape(1,-1)))
    return np.mean(normas)

from sklearn.metrics import accuracy_score
import pandas as pd
def calcular_accuracy_detector(array_total, array_objetivo, sugerencia):
    """
    Calcula la precisión (accuracy) de una sugerencia con respecto a un conjunto objetivo utilizando sklearn.

    Args:
    array_total (list): El conjunto total de números.
    array_objetivo (list): El conjunto de números que se consideran como objetivo.
    sugerencia (list): La sugerencia que deseas evaluar.

    Returns:
    float: El valor de precisión (accuracy) como un número entre 0 y 1.
    """
    array_total = sorted(array_total)
    array_objetivo = sorted(array_objetivo)
    sugerencia = sorted(sugerencia)
    # Crear un conjunto de 0s y 1s para comparar con accuracy_score
    if len(array_objetivo)>0:
        y_true = [1 if numero in array_objetivo else 0 for numero in array_total]
    else:
        y_true = [0 for numero in array_total]
    y_pred = [1 if numero in sugerencia else 0 for numero in array_total]


    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy detector: ", accuracy)

    df = pd.read_csv(os.getcwd() + "/detector_accuracy.csv")
    new = pd.DataFrame([accuracy], columns=["Accuracy"])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(os.getcwd() + "/detector_accuracy.csv", index=False)


def contar_elementos(array):
    unique_values, counts = np.unique(array, return_counts=True)
    count_dict = dict(zip(unique_values, counts))

    contador_1 = count_dict.get(1, 0)
    contador_0 = count_dict.get(0, 0)
    contador_menos_1 = count_dict.get(-1, 0)

    return [contador_1, contador_0, contador_menos_1]



def sign_statistics(weights):
    sig = []
    for i in range(len(weights)):
        # Aplica 'numpy.sign' para obtener los valores -1, 0 y 1
        signed_arr = np.sign(weights[i])
        #print(signed_arr)
        if signed_arr.ndim >1:
            signed_arr = signed_arr.reshape(1,-1)
            # Usa 'numpy.bipncount' para contar la cantidad de -1, 0 y 1
            counts = contar_elementos(signed_arr)


        if signed_arr.ndim ==1:
            counts = contar_elementos(signed_arr)

        # Los índices 0, 1 y 2 en 'counts' representan -1, 0 y 1 respectivamente
        count_1 = counts[0]
        count_0 = counts[1]
        count_neg_1 = counts[2]

        sig.append([count_1,count_0,count_neg_1])

    sig_total = np.sum(sig, axis=0)
    suma = sig_total[0] + sig_total[1] + sig_total[2]
    sig_total = sig_total/suma
    return sig_total


from sklearn.cluster import MeanShift
def signGuard(gradients, party_numbers):
    normas = []
    for i in range(len(gradients)):
        normas.append(get_norms(gradients[i]))

    malos_1 = [num for num, norma in zip(party_numbers, normas) if (norma/np.median(normas) < 0.1) or (norma/np.median(normas))>3]

    print("Malos 1 = ", malos_1)

    #[similitud_coseno_sax(gradiente) for gradiente in zip(*gradients)]
    signos = []
    for i in range(len(gradients)):
        signos.append(sign_statistics(gradients[i]))

    meanshift = MeanShift()
    meanshift.fit(signos)
    labels = meanshift.labels_
    print(labels)
    counts = np.bincount(labels)
    valor_mas_repetido = np.argmax(counts)
    indices_valor_mas_repetido = np.where(labels == valor_mas_repetido)
    party_numbers = np.array(party_numbers)
    buenos_2 = party_numbers[indices_valor_mas_repetido]
    print(buenos_2)
    malos_2 = list(set(party_numbers) - set(buenos_2))
    if len(malos_2)>len(buenos_2):
        malos_2=buenos_2

    print("Malos 2 = ", malos_2)

    #malos = np.intersect1d(malos_1, malos_2)
    malos = np.union1d(malos_1, malos_2)

    print("Malos total = ", malos)
    return malos


def resta_grad_media(gradientes, media):
    resultado = []
    for fila1 in gradientes:
            resul =[np.array(sublista1)-np.array(sublista2) for sublista1, sublista2 in zip(fila1, media)]
            resultado.append(resul)
    return resultado

def calcular_top_sing_eigenvector(gradients):
    resultado = []
    for matrix in gradients:
        if matrix.ndim == 1:
            A = np.outer(matrix, matrix)

            # Find the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(A)

            # The eigenvector corresponding to the largest eigenvalue
            top_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
            resultado.append(top_eigenvector)
        else:
            U, S, VT = np.linalg.svd(matrix)

            # The top right singular vector is the last column of V
            #top_right_singular_vector = VT[-1, :]
            #resultado.append(top_right_singular_vector)
            resultado.append(VT)

    return resultado

def prod_escalar(grad, v):
    for g in grad:
        print(g.shape)
    print("-------")
    for va in v:
        print(va.shape)
    producto_escalar = [np.sum(np.array(gradi)*np.array(vi.T)) for gradi, vi in zip(grad,v)]
    final = np.mean(producto_escalar)
    return final

from  random import sample
def DnC(gradients, c, m, niters, party_numbers):
    n = len(gradients)
    buenos = []
    for i in range(niters):
        list_length = np.random.randint(5, n-5)
        r = sorted(sample(range(1, n), list_length))
        grad = np.array(gradients)[r]
        media = [np.mean(arrays, axis=0) for arrays in zip(*grad)]
        centrados = resta_grad_media(grad, media)
        v =[calcular_top_sing_eigenvector(gradi) for gradi in centrados]
        ps = [prod_escalar(gradi, vi) for gradi, vi in zip(gradients,v)]
        indice = len(gradients)- c*m
        lowest_values = sorted(ps)[:indice]
        indices_of_lowest_values = [i for i, value in enumerate(lowest_values) if value in lowest_values]
        buenos.append(party_numbers[indices_of_lowest_values])

    list_intersection = set(buenos[0]).intersection(*buenos[1:])
    print("Clientes buenos = ", list_intersection)
    clientes_malos = party_numbers - list_intersection
    print("Clientes malos = ", clientes_malos)
    return clientes_malos


import matplotlib.pyplot as plt
def graficas_clustering(similarity_matrix, clusters, clientes, ronda):
    num_clusters = 2
    for i, cluster_num in enumerate(range(num_clusters)):
        cluster_points = similarity_matrix[clusters == cluster_num]
        cluster_indices = np.argwhere(clusters == cluster_num).flatten()
        for point, index in zip(cluster_points, cluster_indices):
            plt.scatter(point[0], point[1], c=f'C{cluster_num + 1}')
            plt.text(point[0], point[1], str(clientes[index]), fontsize=8, ha='center', va='center')

    # Agregar leyenda para los clusters
    for i in range(num_clusters):
        plt.scatter([], [], c=f'C{i}', label=f'Cluster {i + 1}')  # Ajuste la numeración aquí

    plt.title('Clustering mediante K-means con 2 componentes')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.legend()
    plt.savefig(os.getcwd() + '/graficas_clustering/grafica_ronda' + str(ronda) +'.pdf', format="pdf", bbox_inches="tight")
    plt.close('all')


def inverse_distance_weighted_kmeans(X, n_clusters=2, max_iter=300, tol=1e-4):
    # Inicialización de centroides con KMeans++
    kmeans_init = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
    kmeans_init.fit(X)
    centroids = kmeans_init.cluster_centers_

    for _ in range(max_iter):
        # Calcular distancias y asignar puntos a clusters
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calcular pesos inversamente proporcionales a la distancia
        inverse_distances = 1 / (np.min(distances, axis=1) + 1e-8)**4  # Se agrega un pequeño valor para evitar divisiones por cero

        # Actualizar centroides con pesos inversamente proporcionales a la distancia
        new_centroids = np.empty_like(centroids)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.average(cluster_points, axis=0, weights=inverse_distances[labels == i])

        # Verificar convergencia
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    final_labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)
    return centroids, final_labels

def maximos_por_fila(matriz):
    maximos = []
    for fila in matriz:
        maximo_fila = max(fila)
        maximos.append(maximo_fila)
    return maximos

def resta_matriz_diagonal(matriz):
    # Convertir la matriz a un array de NumPy
    matriz_np = np.array(matriz)

    # Obtener la matriz diagonal de unos
    matriz_diagonal_unos = np.eye(matriz_np.shape[0])

    # Restar la matriz diagonal de unos de la matriz original
    resultado = matriz_np - matriz_diagonal_unos

    return resultado

def foolsgold(coseno):

    coseno_aux = resta_matriz_diagonal(coseno)
    v = maximos_por_fila(coseno_aux)
    print("v: ", v)
    alpha = []
    for i in range(len(v)):
        for j in range(i, len(v)):
            if v[j] > v[i]:
                coseno[i][j] = coseno[i][j]*v[i]/v[j]

        alpha.append(1-max(coseno[i]))
    print("alpha: ", alpha)
    alpha = [alpha[i]/max(alpha) for i in range(len(alpha))]
    print("alpha: ", alpha)
    alpha = [scipy.special.logit(alpha[i])+0.5 for i in range(len(alpha))]
    alpha = np.asarray(alpha)
    alpha[alpha == np.inf] = 10000
    print("alpha: ", alpha)
    return alpha

def n_elementos_menos_bajos_con_indices(vector, party_numbers, n):
    # Convierte el vector a un array de NumPy
    vector_np = np.array(vector)

    # Utiliza numpy.argsort para obtener los índices de los elementos ordenados
    indices_ordenados = np.argsort(vector_np)

    # Utiliza numpy.partition para obtener los N índices más bajos
    indices_n_menos_bajos = indices_ordenados[:n]

    # Obtén los N elementos más bajos y sus índices
    n_menos_bajos = vector_np[indices_n_menos_bajos]
    #party_numbers2 = party_numbers[indices_n_menos_bajos]
    party_numbers2 = [party_numbers[ind] for ind in indices_n_menos_bajos]
    print("clientes malos: ", party_numbers2)

    return party_numbers2#n_menos_bajos, indices_n_menos_bajos























