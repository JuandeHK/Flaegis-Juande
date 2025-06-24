import flwr as fl
import tensorflow as tf
import logging
import multiprocessing
import time
import os

# Es crucial que este script se ejecute desde la raíz del proyecto
from src.utils.data_loader import DataLoader

# --- Parámetros de Configuración ---
CONFIG = {
    "data": {
        'base_dir': os.getcwd(),
        'dataset_type': 'pleidata',
        'target_col': 'dif_cons_real',
        'feature_cols': ['dif_cons_real', 'dif_cons_smooth', 'V2', 'V4', 'V12', 'V26', 'Hour_1', 'Hour_2', 'Hour_3', 'Season_1', 'Season_2', 'Season_3', 'Season_4', 'tmed', 'hrmed', 'radmed', 'vvmed', 'dvmed', 'prec', 'dewpt', 'dpv'],
        'window_size': 12,
        'separator': ';'
    },
    "model": {'batch_size': 32},
    "training": {"local_epochs": 5}
}

# --- Cliente de Flower Mínimo ---
class MinimalClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = None
        self.train_data = None

    def get_parameters(self, config):
        if self.model is None:
            print("CLIENTE: Creando modelo ConvLSTM de PRUEBA (súper ligero)...")
            # Modelo mínimo para ver si sobrevive al arranque
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(12, 1, 21, 1)),
                tf.keras.layers.ConvLSTM2D(filters=4, kernel_size=(1, 3), padding='same', return_sequences=False),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            self.model.compile(optimizer='adam', loss='mse')
        print("CLIENTE: Devolviendo pesos iniciales.")
        return self.model.get_weights()

    def fit(self, parameters, config):
        print("CLIENTE: Recibida orden de entrenamiento (fit).")
        if self.train_data is None:
            print("CLIENTE: Cargando datos para el cliente 0...")
            loader = DataLoader(base_dir=CONFIG["data"]["base_dir"], data_config=CONFIG["data"])
            (self.train_data, self.test_data) = loader.load_data_for_client(0)

        (x_train, y_train) = self.train_data
        print(f"CLIENTE: Empezando model.fit() con datos de forma {x_train.shape}")
        self.model.set_weights(parameters)
        history = self.model.fit(x_train, y_train, epochs=CONFIG["training"]["local_epochs"], batch_size=CONFIG["model"]["batch_size"], verbose=2)
        print("CLIENTE: Entrenamiento finalizado.")
        return self.model.get_weights(), len(x_train), {"loss": history.history['loss'][0]}

    def evaluate(self, parameters, config):
        return 0.0, 0, {"mae": 0.0}

# --- Funciones para arrancar Servidor y Cliente ---
def start_server():
    print("SERVIDOR: Iniciando servidor...")
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=2), strategy=fl.server.strategy.FedAvg())
    print("SERVIDOR: Terminado.")

def start_client():
    print("CLIENTE: Conectando...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=MinimalClient())
    print("CLIENTE: Terminado.")

# --- Bloque de Ejecución Principal ---
if __name__ == "__main__":
    print("MAIN: Lanzando servidor...")
    server_proc = multiprocessing.Process(target=start_server)
    server_proc.start()

    print("MAIN: Pausa de 5s...")
    time.sleep(5)

    print("MAIN: Lanzando cliente...")
    start_client()

    server_proc.join()
    print("MAIN: Finalizado.")