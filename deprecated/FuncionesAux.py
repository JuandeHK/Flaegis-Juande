import os
import time

import numpy
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import flwr as fl
from keras.layers import Embedding, GRU

from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, LSTM
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import Nadam
from sklearn.metrics import  f1_score, precision_score, \
    recall_score, matthews_corrcoef, cohen_kappa_score

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from random import sample
import itertools
import pickle
import multiprocessing
import psutil
import cProfile



server_id = 0
nodesMal= []


def set_nodelMal(nodesMal):
    glob = globals()
    glob['nodesMal'] = nodesMal

def set_server_id(id):
    glob = globals()
    glob['server_id'] = id

def get_server_id():
    glob = globals()
    return  glob['server_id']


def leer_cliente_veremi(client_n):
    dir_base = os.getcwd()

    dir_tam = '/home/enrique/flower/FedMedian' + "/SMOTETomek/AaTabla_tamanos.csv"
    dir_lista_tam = os.path.join(dir_base, dir_tam)
    dir_lista_tam = os.path.abspath(dir_lista_tam)

    lista_clientes = pd.read_csv(dir_lista_tam)
    lista_clientes = np.array(lista_clientes['Vehicle'])
    lc_ind = client_n - 1
    dir_datos = '/home/enrique/flower/FedMedian/'+"SMOTETomek/data_party" + str(int(lista_clientes[lc_ind])) + ".csv"

    dir_lista_datos = os.path.join(dir_base, dir_datos)
    dir_lista_datos = os.path.abspath(dir_lista_datos)
    return dir_lista_datos

def load_data(path):
    csv = leer_cliente_veremi(path)

    training_dataset = pd.read_csv(csv)

    training_dataset = preprocess(training_dataset)

    # split the data
    x_0 = training_dataset.iloc[:, :-1]
    y_0 = training_dataset.iloc[:, -1]

    x = np.array(x_0)
    y = np.array(y_0)

    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)

    return (x_train, y_tr), (x_test, y_te)


def load_data_emnist(client):
    path = os.getcwd() + "/Femnist_100/data_party" + str(client) + ".csv"
    training_dataset = pd.read_csv(path)
    x_0 = training_dataset.iloc[:, 1:]
    y_0 = training_dataset.iloc[:, 0]

    x = np.array(x_0)
    x = x /255.0
    y = np.array(y_0)

    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)

    return (x_train, y_tr), (x_test, y_te)


def load_data_femnist(client):
    training_dataset = pd.read_csv("/home/enriquemarmol/parties_femnist/Femnist_leaf/data_party" + str(client) + ".csv")

    # split the data
    x_0 = training_dataset.iloc[:, :-1]
    y_0 = training_dataset.iloc[:, -1]

    x = np.array(x_0)
    y = np.array(y_0)

    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)  

    return (x_train, y_tr), (x_test, y_te)


def load_data_mnist(client):
    training_dataset = pd.read_csv("/home/enriquemarmol/Datasets/archive/data_party" + str(client) + ".csv")

    # split the data
    x_0 = training_dataset.iloc[:, 1:]
    y_0 = training_dataset.iloc[:, 0]

    x = np.array(x_0)
    y = np.array(y_0)

    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)

    return (x_train, y_tr), (x_test, y_te)

def load_data_femnist_data_poisoning(client):
    training_dataset = pd.read_csv("/home/enriquemarmol/parties_femnist/Femnist_leaf/data_party" + str(client) + ".csv")

    # split the data
    x_0 = training_dataset.iloc[:, :-1]
    y_0 = training_dataset.iloc[:, -1]

    x = np.array(x_0)
    y = np.array(y_0)
    y= np.random.randint(60, size=len(y))

    x_train, x_test, y_tr, y_te = \
        train_test_split(x, y, test_size=0.2)
    
    return (x_train, y_tr), (x_test, y_te)





def preprocess(training_data):
    # Transform INF and NaN values to median
    pd.set_option('use_inf_as_na', True)
    training_data.fillna(training_data.median(), inplace=True)

    # Shuffle samples
    training_data = training_data.sample(frac=1).reset_index(drop=True)

    # Normalize values
    scaler = MinMaxScaler()

    features_to_normalize = training_data.columns.difference(['Label'])

    training_data[features_to_normalize] = scaler.fit_transform(training_data[features_to_normalize])

    # Return preprocessed data
    return training_data




def my_FPR(y_true, y_pred):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    TC = confusion_matrix.sum(axis=1)
    TC_s = TC.sum()
    TC = TC / TC_s

    FPR = FP / (FP + TN)
    FPR_div = FPR * TC
    FPR_w = FPR_div.sum()

    return FPR_w


def create_model_emnist():
    model = tf.keras.models.Sequential([
        # initial normalization
        tf.keras.layers.Reshape((28, 28, 1), input_shape=(784,)),

        # first convolution
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),  # applies kernels to our data
        tf.keras.layers.MaxPooling2D(2, 2),  # reduce dimension

        # second convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        # third convolution
        tf.keras.layers.Conv2D(24, (3, 3), activation='relu'),

        # feed to DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(47, activation=tf.nn.softmax)  # generalized logistic regression
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
from tensorflow.keras import layers
from keras.initializers import RandomNormal
def create_model_femnist():
    model = tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(24, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(62, activation='softmax')  # Number of classes in FEMNIST dataset
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model_femnist_random():
    model = tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(24, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(62, activation='softmax')  # Number of classes in FEMNIST dataset
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model_mnist():
    model = tf.keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(784,)),
        layers.Conv2D(8, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(24, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # Number of classes in MNIST dataset
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model




def create_model_mlp(input_shape):
    model = Sequential()
    model.add(Dense(350, input_shape=(input_shape,), activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.0))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    nadam = Nadam(learning_rate=0.005)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'],
                  run_eagerly=True)

    return model

def create_model_mlp_random(input_shape):
    model = Sequential()
    model.add(Dense(350, input_shape=(input_shape,), activation='relu', kernel_constraint=maxnorm(4),
                    kernel_initializer='random_normal',
                    bias_initializer='zeros'))
    model.add(Dropout(0.0))
    model.add(Dense(50, activation='relu',kernel_initializer='random_normal',
                    bias_initializer='zeros'))
    model.add(Dense(6, activation='sigmoid',kernel_initializer='random_normal',
                    bias_initializer='zeros'))
    opt = Nadam(learning_rate=0.005)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  run_eagerly=True)
    return model

def create_model_binary(n_inputs):
    model = Sequential()
    model.add(Dense(550, input_dim=n_inputs, activation='relu'))
    model.add(Dense(540, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def min_max_attack(all_updates, model_re, dev_type='unit_vec'):
    if dev_type == 'unit_vec':

        if np.linalg.norm((model_re)) == 0:
            deviation = model_re
        else:
            deviation = model_re / np.linalg.norm(model_re) 

    elif dev_type == 'sign':
        deviation = np.sign(model_re)
    elif dev_type == 'std':
        deviation = np.std(all_updates, 0)

    lamda = 50
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = np.linalg.norm((all_updates - update), axis=1) ** 2
        distances = distance[None, :] if not len(distances) else np.concatenate((distances, distance[None, :]), 0)

    max_distance = np.max(distances)
    while np.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = np.linalg.norm((all_updates - mal_update), axis=1) ** 2
        max_d = np.max(distance)

        if max_d <= max_distance:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

def min_sum_attack(all_updates, model_re, dev_type='unit_vec'):
    if dev_type == 'unit_vec':
        if np.linalg.norm((model_re)) == 0:
            deviation = model_re
        else:
            deviation = model_re / np.linalg.norm(model_re)  # unit vector, dir opp to good dir

    elif dev_type == 'sign':
        deviation = np.sign(model_re)
    elif dev_type == 'std':
        deviation = np.std(all_updates, 0)

    lamda = 50
    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    distances = []
    for update in all_updates:
        distance = np.linalg.norm((all_updates - update), axis=1) ** 2
        distances = distance[None, :] if not len(distances) else np.concatenate((distances, distance[None, :]), 0)

    scores = np.sum(distances, axis=1)
    min_score = np.min(scores)

    while np.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = np.linalg.norm((all_updates - mal_update), axis=1) ** 2
        score = np.sum(distance)

        if score <= min_score:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)

    return mal_update

def LIE_attack(updates):
    print("LIE")
    matrices_array = np.array(updates)
    mean = np.mean(matrices_array, axis=0)
    std = np.var(matrices_array, axis=0)**0.5
    rtn = mean - 1.5*std
    return rtn

def STATOPT(updates):
    print("STATOPT")
    matrices_array = np.array(updates)
    mean = np.mean(matrices_array, axis=0)
    signo = np.sign(mean)
    rtn = -signo
    return rtn



import json
class tfmlpClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, party_number, nodesMal, barrier):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.party_number = party_number
        self.nodesMal = nodesMal
        self.barrier = barrier

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        dir_base = os.getcwd()
        # Update local model parameters
        compr_array = 0
        self.model.set_weights(parameters)
        mean_weights = parameters

        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]
        rnd = config["round"]
        steps = config["val_steps"]
        rondas = config["rounds"]
        bd = nodesMal


        """
        if rnd == 1:
            self.model.set_weights(parameters)
        """

        print(("Ronda: " + str(rnd)))


        #self.barrier.wait()
        if rnd > 1:
            for epoch in range(epochs):
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    batch_size,
                    1,
                    validation_split=0.1,
                )
                theta = 0.98
                new_param = fedplus2(self.model.get_weights(), mean_weights, theta)
                self.model.set_weights(new_param)
        else:
            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=0.1,
            )
            new_param = self.model.get_weights()

        print("PARTY ", self.party_number, " : HA TERMINADO ENTRENAMIENDO")
        bd = nodesMal
        aux_check = 0
        path = os.getcwd() + "/check_arrays/array_comp" + str(self.party_number) + ".pickle"


        compromised_complete = True

        # ataque del paper
        # Ataque en si del paper, el necesario
        
        if self.party_number in bd:
            modelos_att = []
            pesos_finales = []
            contador = 0
            verificar = 0
            pesos = self.model.get_weights()
            capas = len(pesos)
            print("Hay ", capas, " capas")
            while verificar < len(bd):
                print("party ", self.party_number, " atrancada")
                for cliente_malo in bd:
                    modelo_malo = create_model_femnist()
                    try:
                        modelo_malo.load_weights(os.getcwd() + '/modelos_att/modelo_guardado' + str(cliente_malo) + '.h5')
                        modelos_att.append(modelo_malo.get_weights())
                        verificar = verificar+1
                    except BlockingIOError:
                        hola = 0
                    except OSError:
                        hola = 0
                    except KeyError:
                        hola = 0

            for matrices in zip(*modelos_att):
                pesos_compr = LIE_attack(updates=matrices)
                pesos_finales.append(pesos_compr)
                contador=contador+1

            self.model.set_weights(pesos_finales) #pesos finales
        

        print("Cliente ", str(self.party_number), " Terminó. ")
        # Return updated model parameters and results
        new_param = self.model.get_weights()
        parameters_prime = new_param
        del new_param
        num_examples_train = len(self.x_train)


        y_pred = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred, axis=1)

        gradient = get_gradient(self.x_train, self.y_train, self.model)
        gradient = [np.array(gradient[i]) for i in range(len(gradient))]
        cadena_matrices = json.dumps([matriz.tolist() for matriz in gradient])

        # Serializar la lista de matrices a una cadena JSON
        #cadena_matrices = json.dumps(cadena_matrices)

        results = {
            "Party number": self.party_number,
            "gradient": cadena_matrices,
            "ronda": rnd,

            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],

        }

        # Guardar en csv externo
        aux = []
        path = dir_base + '/metricas_parties/history_' + str(self.party_number) + '.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)#, 32, steps=steps)
        lista = {
            "Accuracy": accuracy,
            "Recall": recall_score(self.y_test, y_pred, average='weighted'),
            "Precision": precision_score(self.y_test, y_pred, average='weighted'),
            "F1_score": f1_score(self.y_test, y_pred, average='weighted'),
            "FPR": my_FPR(self.y_test, y_pred),
            "Matthew_Correlation_coefficient": matthews_corrcoef(self.y_test, y_pred),
            "Cohen_Kappa_Score": cohen_kappa_score(self.y_test, y_pred)
        }
        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        #del loss, accuracy, lista
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def fedplus(weights, mean, theta):
    z = numpy.asarray(mean)
    weights = numpy.asarray(weights)

    fedp = (theta * weights) + (1 - theta) * z
    return fedp

def fedplus2(weights, mean, theta):
    z = [numpy.asarray(mean[i]) for i in range(len(mean))]
    weights = [numpy.asarray(weights[i]) for i in range(len(mean))]

    fedp = [(theta * weights[i]) + (1 - theta) * z[i] for i in range(len(weights))]
    return fedp

def check_model(model_bien,path):
    model = create_model_femnist()

    check = False
    while not check:
        try:
            model.load_weights(path)
            check = True
        except BlockingIOError:
            check_save(model_bien,path)
        except OSError:
            check_save(model_bien,path)
        except KeyError:
            check_save(model_bien,path)
def check_save(model, path):
    check = False
    while not check:
        print(path, " : ATRANCADA")
        try:
            model.save_weights(path)
            check = True
        except BlockingIOError:
            nada = 0
        except OSError:
            nada = 0
        except KeyError:
            nada = 0

def check_pickle(path):
    check = False
    while not check:
        try:
            with open(path, "rb") as f:
                unpickler = pickle.Unpickler(f)
                # if file is not empty scores will be equal
                # to the value unpickled
                aux = unpickler.load()
                if aux>0:
                    check= True
        except:
            aux = 0
    return  aux


def start_client(client_n, nodesMal, barrier):
    time.sleep(10)

    # Limpiamos el csv
    carpeta_nueva = "metricas_parties"
    try:
        os.mkdir(carpeta_nueva)
    except FileExistsError:
        aux = 0

    vacio = []
    col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'FPR', 'Matthew_Correlation_coefficient',
                'Cohen_Kappa_Score']
    path = os.getcwd() + '/metricas_parties/history_' + str(client_n) + '.csv'
    df = pd.DataFrame(vacio, columns=col_name)
    df.to_csv(path, index=False)


    (x_train, y_train), (x_test, y_test) = load_data_femnist(client_n)
    input_shape = len(x_train[0])
    model = create_model_femnist()

    if client_n in nodesMal:
        model.save_weights(os.getcwd() + '/modelos_att/modelo_guardado' + str(client_n) + '.h5')

    # Start Flower client

    client = tfmlpClient(model, x_train, y_train, x_test, y_test,
                         client_n,nodesMal, barrier)
    # client = sklearnClient(model, x_train, y_train, x_test, y_test)
    print(("party " + str(client_n) + " lista"))

    # IP lorien 155.54.95.95
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)


def start_server(parties, rounds, th):

    dir_base = '/home/enrique/flower/FedMedian'
    dir_lista_datos = dir_base + "/SMOTETomek/data_party" + str(8) + ".csv"
    (x_train, y_train), _ = load_data_femnist(1)
    input_shape = len(x_train[0])
    model = create_model_femnist()
    del dir_lista_datos, input_shape

    # Create strategy
    """
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=parties,
        min_eval_clients=parties,
        min_available_clients=parties,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )
    """



    strgy = WeightedMedian(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=parties,
        min_evaluate_clients=parties,
        min_available_clients=parties,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        th= th
    )
    # IP lorien 155.54.95.95
    # Start Flower server for four rounds of federated learning
    fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=rounds), strategy=strgy)
    #fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=rounds), strategy=krumstr)



def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    dir_base = '/home/enrique/flower/FedMedian'
    dir_datos = "SMOTETomek/data_party" + str(1) + ".csv"
    dir_lista_datos = os.path.join(dir_base, dir_datos)
    dir_lista_datos = os.path.abspath(dir_lista_datos)

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _, (x_val, y_val) = load_data(1)

    # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    # The `evaluate` function will be called after every round
    def evaluate(
            weights,
    ):  # -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]: model.set_weights(weights)

        # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        """
        aux = []
        path = '/home/enrique/Flower/sklearn-logreg-mnist/tf red neuronal/metricas_parties/history_Server.csv'
        col_name = ['Accuracy', 'Recall', 'Precision', 'F1_score', 'Matthew_Correlation_coefficient',
                    'Cohen_Kappa_Score']
        lista = {
            "Accuracy": accuracy,
            "Recall": 0,
            "Precision": 0,
            "F1_score": 0,
            "Matthew_Correlation_coefficient": 0,
            "Cohen_Kappa_Score": 0
        }

        aux.append(lista)
        df1 = pd.DataFrame(aux, columns=col_name)
        df1.to_csv(path, index=None, mode="a", header=not os.path.isfile(path))
        """
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "local_epochs": 1,  # if rnd < 2 else 2,
        "round": rnd,
        "val_steps": 5,
        "rounds": 50
    }
    return config


def evaluate_config(rnd):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 5
    return {"val_steps": val_steps}

#------ MEDIANA



import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, norm
from scipy.stats import ks_2samp
def ks_proportion(sample):
    total = []
    for i in range(1000):
        n_random = 15
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


from logging import WARNING
from flwr.server.strategy import Strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Scalar, MetricsAggregationFn,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import NDArray, NDArrays
from functools import reduce


def aggregate(results: List[Tuple[NDArrays, int]], party_numbers) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    malos = [1,2,3,4]
    buenos = [4,5,6,7,8,9,10,11]
    capas = [weights for weights, _ in results]

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]

    return weights_prime


def aux_media(add,layer_updates, num):
    return reduce(add, layer_updates) / num

def profile_cpu_usage(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_cpu = process.cpu_percent()

        result = func(*args, **kwargs)

        end_cpu = process.cpu_percent()
        cpu_used = end_cpu - start_cpu
        print(f"CPU used by {func.__name__}: {end_cpu - start_cpu}%")

        return result, cpu_used

    return wrapper


from deprecated.metodos_cw import  *
def agg_median(results: List[Tuple[NDArrays, int]], party_numbers, gradients, ronda) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    print("HOLA")
    num_examples_total = sum([num_examples for _, num_examples in results])
    pesos = [num_examples/num_examples_total for _, num_examples in results]
    #pesos = [1 for _, num_examples in results]
    #capas = [weights for weights, _ in results]

    #malos = range(1, 20+1)#[1,2,3,4,5,6,7,8,9,10]
    with open(os.getcwd() + "/nodos_comprometidos.npy", 'rb') as f:
        malos = np.load(f)
    print(len(malos))
    print(malos)
    buenos = [5, 6, 7, 8, 9, 10, 11]
    capas = [weights for weights, _ in results]
    #coseno = [similitud_coseno(layers) for layers in zip(*capas)]
    coseno = [similitud_coseno_sax(layers) for layers in zip(*capas)]
    #coseno = [similitud_coseno_sax(gradiente) for gradiente in zip(*gradients)]

    #coseno_malosBuenos = [submatrix_from_index_pairs(cos, party_numbers, malos, buenos) for cos in coseno]
    #coseno_buenosBuenos = [submatrix_from_index_pairs(cos, party_numbers, buenos, buenos) for cos in coseno]

    #media_capas_malosBuenos = [np.mean(coseno_malosBuenos[i]) for i in range(len(coseno_malosBuenos))]
    #media_capas_buenosBuenos = [mean_without_diagonal(coseno_buenosBuenos[i]) for i in range(len(coseno_buenosBuenos))]

    #medias = [np.mean(coseno[i], axis=0) for i in range(len(coseno))]
    media_abs = np.mean(coseno, axis=0)
    np.save(os.getcwd() + "/matriz_simi.npy", media_abs)
    print("matriz guardada")
    #print(medias)
    #print("MATRIZ MEDIA", media_abs)

    #medias_malosBuenos = np.mean(media_capas_malosBuenos)
    #medias_buenosBuenos = np.mean(media_capas_buenosBuenos)
    """
    print("Similitud coseno capa 1 =", coseno[0])
    print("Similitud coseno capa 1 malos-buenos =", coseno_malosBuenos[0])
    print("Similitud coseno capa 1 buenos-buenos =", coseno_buenosBuenos[0])
    """
    #print("Similud malos con buenos:", medias_malosBuenos)
    #print("Similitud buenos con buenos: ", medias_buenosBuenos)

    #normas = [get_norms(gradients[i]) for i in range(len(gradients))]
    #print("NORMAS: ",normas)


    #clientes_maliciosos = detector_clientes_maliciosos_hdbscan(media_abs, party_numbers)
    clientes_maliciosos = detector_clientes_maliciosos_spectral(media_abs,ronda, party_numbers)
    
    #clientes_maliciosos = signGuard(gradients,party_numbers)
    
    #pesos_fg = foolsgold(media_abs)
    #print(pesos_fg)
    #clientes_maliciosos = n_elementos_menos_bajos_con_indices(pesos_fg, party_numbers, len(malos))
    #clientes_maliciosos = DnC(gradients,1,len(malos), 50, party_numbers)
    
    calcular_accuracy_detector(party_numbers, malos, clientes_maliciosos)
    time.sleep(10)
    if len(clientes_maliciosos) > 0:
        conjunto_grande = set(party_numbers)
        conjunto_subconjunto = set(clientes_maliciosos)

        # Verifica si el segundo conjunto es un subconjunto del primero
        if conjunto_subconjunto.issubset(conjunto_grande):
            # Elimina los elementos del subconjunto del conjunto grande
            conjunto_grande -= conjunto_subconjunto

        # Encuentra los índices de los elementos que quedan en el vector grande
        indices_buenos = [i for i, elem in enumerate(party_numbers) if elem in conjunto_grande]

        #indices_malos = np.in1d(clientes_maliciosos, party_numbers).nonzero()[0]
        #indices_buenos = list(set(party_numbers)-set(indices_malos))
        print("Se toman los clientes: ", conjunto_grande)
        #capas = [capas[i] for i in indices_buenos]
        #weights_prime: NDArrays = [w_median_abs(layers, pesos) for layers in zip(*capas)]


        results = [results[i] for i in indices_buenos]


    num_examples_total = sum([num_examples for _, num_examples in results])
    """
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
     # Compute average weights of each layer
    weights_prime: NDArrays = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
    """

    """
    #para foolsgold
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
     # Compute average weights of each layer
    weights_prime: NDArrays = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
    """


    
    capas = [weights for weights, _ in results]
    weights_prime: NDArrays = [fourier_abs(layers) for layers in zip(*capas)]
    




    # weights_prime: NDArrays = [trimmed_mean_abs(layers, th) for layers in zip(*capas)]
    # weights_prime: NDArrays = [fourier_abs(layers) for layers in zip(*capas)]
    # weights_prime: NDArrays = [w_median_abs(layers, pesos) for layers in zip(*capas)]


    # Compute average weights of each layer
    return weights_prime


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


class WeightedMedian(Strategy):
    def __init__(self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        th,
):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.th = th

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep


    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters


    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]



    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}


        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        party_numbers = [res.metrics for _, res in results]
        party_numbers = [party_numbers[i]['Party number'] for i in range(len(party_numbers))]
        print(party_numbers)
        time.sleep(2)


        gradients = [res.metrics for _, res in results]
        gradients = [gradients[i]['gradient'] for i in range(len(party_numbers))]
        matrices_recuperadas_como_listas = [json.loads(gradients[i]) for i in range(len(party_numbers))]
        matrices_recuperadas = [[np.array(matriz) for matriz in matrices_recuperadas_como_listas[i]] for i in range(len(matrices_recuperadas_como_listas))]


        #parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results, party_numbers))
        #parameters_aggregated = ndarrays_to_parameters(agg_median(weights_results,self.th))
        parameters_aggregated = ndarrays_to_parameters(agg_median(weights_results, party_numbers, matrices_recuperadas, server_round))
        """
        profiled_w_median_abs = profile_cpu_usage(aggregate)
        profiler = cProfile.Profile()
        profiler.enable()
        _, cpu_used = profiled_w_median_abs(weights_results)
        print("CPU USED: ", cpu_used)
        profiler.disable()
        time.sleep(5)
        """
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

from flwr.server.strategy import krum

from flwr.server.strategy.aggregate import aggregate_krum
WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
class my_krum(krum.Krum):
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            num_malicious_clients: int = 0,
            num_clients_to_keep: int = 0,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep

    def __repr__(self) -> str:
        rep = f"Krum(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using Krum."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate_krum(
                weights_results, self.num_malicious_clients, self.num_clients_to_keep
            )
        )

        profiled_w_median_abs = profile_cpu_usage(aggregate_krum)

        profiler = cProfile.Profile()
        profiler.enable()
        #cpu_usages = []
        _, cpu_used = profiled_w_median_abs(weights_results, self.num_malicious_clients, self.num_clients_to_keep)
        profiler.disable()
        print("CPU USED: ", cpu_used)
        profiler.print_stats(sort="cumulative")
        time.sleep(5)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
