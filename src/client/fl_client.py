import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.metrics import r2_score

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data, is_malicious=False, client_id=0):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.is_malicious = is_malicious
        self.client_id = client_id
        self.logger = logging.getLogger(f"Client-{client_id}")

    def get_parameters(self, config):
        """Obtiene los parámetros del modelo"""
        try:
            return self.model.get_weights()
        except Exception as e:
            self.logger.error(f"Error obteniendo parámetros: {e}")
            raise

    # En src/client/fl_client.py, reemplaza la función fit() entera

    def fit(self, parameters, config):
        try:
            self.logger.info(f"CLIENT {self.client_id}: === FIT_START ===")
            self.logger.info(f"Recibida orden de entrenamiento para la ronda {config.get('server_round', 'desconocida')}")

            self.logger.info(f"CLIENT {self.client_id}: Estableciendo pesos del modelo global...")
            self.model.set_weights(parameters)
            self.logger.info(f"CLIENT {self.client_id}: Pesos del modelo establecidos.")
            
            x_train, y_train = self.train_data
            self.logger.info(f"CLIENT {self.client_id}: Forma de los datos de entrenamiento: X={x_train.shape}, y={y_train.shape}")
            
            if len(x_train) == 0:
                self.logger.warning(f"CLIENT {self.client_id}: No hay datos de entrenamiento. Devolviendo resultado vacío.")
                return [], 0, {}
            
            batch_size = config.get("batch_size", 32)
            epochs = config.get("local_epochs", 10)
            self.logger.info(f"CLIENT {self.client_id}: Preparado para llamar a model.fit() con epochs={epochs} y batch_size={batch_size}")
            
            # Este es el punto más probable de fallo silencioso
            history = self.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=2  # Usamos verbose=2 para que Keras nos dé más información
            )
            
            self.logger.info(f"CLIENT {self.client_id}: ¡model.fit() completado con éxito!")
            
            loss = history.history["loss"][0]
            self.logger.info(f"CLIENT {self.client_id}: Entrenamiento finalizado. Loss: {loss:.4f}")
            
            return self.model.get_weights(), len(x_train), {"loss": loss}

        except Exception as e:
            # Esto debería atrapar cualquier error de Python que ocurra
            self.logger.error(f"CLIENT {self.client_id}: !!! ERROR CRÍTICO DURANTE FIT !!!", exc_info=True)
            raise

    def evaluate(self, parameters, config):
        """Evalúa el modelo localmente"""
        try:
            self.logger.info("Iniciando evaluación")
            
            # Establecer parámetros del modelo
            self.model.set_weights(parameters)
            
            # Obtener datos de prueba
            x_test, y_test = self.test_data
            self.logger.info(f"Datos de prueba: X={x_test.shape}, y={y_test.shape}")
            
            if len(x_test) == 0:
                self.logger.warning("No hay datos de prueba, usando datos de entrenamiento")
                x_test, y_test = self.train_data
            
            if len(x_test) == 0:
                raise ValueError("No hay datos disponibles para evaluación")
            
            # Configuración de evaluación
            batch_size = config.get("batch_size", 32)
            
            # Evaluación
            loss, mae = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            
            self.logger.info(f"Evaluación completada. Loss: {loss:.4f}, MAE: {mae:.4f}")
            
            # Devolver loss, número de ejemplos y métricas
            return loss, len(x_test), {"mae": mae}
            
        except Exception as e:
            self.logger.error(f"Error durante evaluación: {e}", exc_info=True)
            raise

