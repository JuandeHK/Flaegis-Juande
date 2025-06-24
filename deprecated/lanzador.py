import numpy as np

from deprecated.FuncionesAux import *
import warnings
import os, shutil

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


warnings.filterwarnings("ignore")
import multiprocessing
import time
import argparse

parametros = argparse.ArgumentParser()
parametros.add_argument("-n_mal", nargs="?", help="number of malicious clients", type=int, default=0)
parametros.add_argument("-rondas", nargs="?", help="rondas", type=int, default=50)

param = parametros.parse_args()
N_mal = param.n_mal
rnds= param.rondas

print("Number of cpu : ", multiprocessing.cpu_count())

rondas = rnds
n_clientes = 50
acc_media_gb = []
clients = []

if N_mal == 0:
    L = []
    np.save(os.getcwd() + "/nodos_comprometidos.npy", L)
else:
    L=sorted(sample(range(1, n_clientes+1), N_mal))
    print("Numero de clientes maliciosos = ", len(L))
    print("Nodos comprometidos = " + str(L))
    L = np.asarray(L)
    np.save(os.getcwd() + "/nodos_comprometidos.npy", L)


dir = os.getcwd() + '/modelos_att'
for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)

dir_2 = os.getcwd() + '/check_arrays'
for files in os.listdir(dir_2):
    path = os.path.join(dir_2, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)


#limpiamos el detector
vacio = []
col_name = ['Accuracy']
path = os.getcwd() + '/detector_accuracy.csv'
df = pd.DataFrame(vacio, columns=col_name)
df.to_csv(path, index=False)

for client in L:
    aux = 0
    path_aux = os.getcwd() + "/check_arrays/array_comp" + str(client) + ".pickle"
    with open(path_aux, "wb") as f:
        pickle.dump(aux, f)

barrier = multiprocessing.Barrier(n_clientes)

set_nodelMal(L)
inicio = time.time()
th = 0
server = multiprocessing.Process(target=start_server, args=(n_clientes, rondas, th))
server.start()
time.sleep(10)

for i in range(n_clientes):
    inx = i + 1
    p = multiprocessing.Process(target=start_client, args=(inx,L, barrier))
    p.start()
    clients.append(p)


server.join()
for client in clients:
    client.join()

print("Nodos comprometidos = " + str(L))
L = np.asarray(L)
np.save(os.getcwd() + "/nodos_comprometidos.npy", L)
