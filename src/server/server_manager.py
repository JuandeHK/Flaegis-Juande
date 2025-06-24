from typing import Dict, Any
import flwr as fl
from .strategy_configurators.fedavg_strategy_configuration import FedAvgStrategyConfigurator
from .strategy_configurators.weighted_median_strategy_configuration import WeightedMedianStrategyConfigurator
from .strategy_configurators.fourier_strategy_configuration import FourierStrategyConfigurator
from .strategy_configurators.trimmed_mean_strategy_configuration import TrimmedMeanStrategyConfigurator
import logging
import numpy as np
import os
from flwr.common import parameters_to_ndarrays
from pathlib import Path

print("=== ARCHIVO SERVER_MANAGER.PY CARGADO ===")
logging.info("=== ARCHIVO SERVER_MANAGER.PY CARGADO ===")

def fit_config_simple(server_round: int) -> Dict[str, Any]:
    """
    Función de configuración simple que no es un método de instancia.
    Devuelve siempre la misma configuración para la prueba.
    """
    logging.info(f"Generando configuración para ronda {server_round} desde la función simple.")
    return {
        "batch_size": 32,
        "local_epochs": 10,
    }


class FederatedServer:
    def __init__(self, n_clients: int, n_rounds: int, config: Dict[str, Any]):
        """
        Inicializa el servidor federado con los parámetros proporcionados.

        Args:
            n_clients (int): Número total de clientes conectados al servidor.
            n_rounds (int): Número total de rondas de entrenamiento.
            config (Dict[str, Any]): Configuración del servidor, incluyendo parámetros
                de entrenamiento, evaluación y estrategias.
        """
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.config = config
         # --- LÍNEA CLAVE QUE FALTA ---
        self.logger = logging.getLogger(__name__)

        self._strategies = {
            "fedavg": FedAvgStrategyConfigurator(),
            "weighted_median": WeightedMedianStrategyConfigurator(),
            "fourier": FourierStrategyConfigurator(),
            "trimmed_mean": TrimmedMeanStrategyConfigurator()
        }

    def _get_address(self) -> str:
        """Obtiene la dirección del servidor desde la configuración."""
        host = self.config.get("server", {}).get("host", "0.0.0.0")
        port = self.config.get("server", {}).get("port", 8080)
        logging.info(f"Server address configured to: {host}:{port}")
        return f"{host}:{port}"
    
    def _get_strategy(self) -> fl.server.strategy.Strategy:
        """
        Obtiene la estrategia de agregación basada en la configuración.

        Returns:
            La estrategia de agregación seleccionada.
        """
        strategy_name = self.config.get("aggregation", {}).get("strategy", "fedavg")
        
        strategy_creator = self._strategies.get(strategy_name)
        if not strategy_creator:
            raise ValueError(f"Estrategia de agregación no soportada: {strategy_name}")
            
        server_config = {
            "n_clients": self.n_clients,
            "config": self.config,
            "fit_config": self._fit_config,
            "evaluate_config": self._evaluate_config
        }
        
        return strategy_creator.create_strategy(server_config)


    def start(self):
        """
        Inicia el servidor de Flower con la estrategia definida en el config.yaml.
        """
        server_address = self._get_address()
        
        # Esta línea lee la configuración de tu config.yaml (ej: "fedavg" o "weighted_median")
        # y usa el configurador correspondiente para crear la estrategia.
        strategy = self._get_strategy()
        
        logging.info(f"Servidor iniciando con la estrategia: '{self.config.get('aggregation', {}).get('strategy')}'")

        fl.server.start_server(
            server_address=server_address,
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=self.n_rounds),
        )

    # def start(self):
    #     """
    #     Inicia el servidor, ejecuta las rondas y GUARDA el modelo final.
    #     """
    #     server_address = self._get_address()
    #     strategy = self._get_strategy()

    #     logging.info(f"Servidor iniciando con la estrategia: '{self.config.get('aggregation', {}).get('strategy')}'")

    #     # Esta llamada se ejecuta y devuelve el historial cuando termina
    #     history = fl.server.start_server(
    #         server_address=server_address,
    #         strategy=strategy,
    #         config=fl.server.ServerConfig(num_rounds=self.n_rounds),
    #     )

    #     self.logger.info("--- [SERVER] El entrenamiento federado (start_server) ha finalizado. ---")
    #     self.logger.info("--- [SERVER] Intentando guardar el modelo global final... ---")

    #     # --- DEBUG COMPLETO ---
    #     self.logger.info(f"--- [SERVER-DEBUG] Atributos de strategy: {[attr for attr in dir(strategy) if 'param' in attr.lower()]}")
        
    #     # Verificar initial_parameters
    #     if hasattr(strategy, 'initial_parameters'):
    #         self.logger.info(f"--- [SERVER-DEBUG] initial_parameters existe: {strategy.initial_parameters is not None}")
    #         self.logger.info(f"--- [SERVER-DEBUG] tipo: {type(strategy.initial_parameters)}")
    #         if strategy.initial_parameters:
    #             self.logger.info(f"--- [SERVER-DEBUG] contenido: {len(strategy.initial_parameters) if hasattr(strategy.initial_parameters, '__len__') else 'N/A'}")

    #     # --- OBTENER PARÁMETROS ---
    #     try:
    #         final_params = None
            
    #         # Opción 1: Desde history
    #         if hasattr(history, 'parameters') and history.parameters:
    #             self.logger.info("--- [SERVER] Obteniendo parámetros desde history ---")
    #             final_params = parameters_to_ndarrays(history.parameters)
            
    #         # Opción 2: Desde initial_parameters (pero verificando que no esté vacío)
    #         elif hasattr(strategy, 'initial_parameters') and strategy.initial_parameters is not None:
    #             self.logger.info("--- [SERVER] Obteniendo parámetros desde strategy.initial_parameters ---")
    #             final_params = parameters_to_ndarrays(strategy.initial_parameters)
            
    #         else:
    #             raise ValueError("No se encontraron parámetros válidos en ninguna ubicación")

    #         # Guardar los parámetros
    #         output_path = os.path.join(
    #             "/home/juand/TFG/FLAegis-Federated-Learning-Approach-for-Enhanced-Guarding-against-Intrusion-and-Security-threats",
    #             "final_global_model.npz"
    #         )
            
    #         np.savez(output_path, *final_params)
    #         self.logger.info(f"--- [SERVER] Parámetros del modelo final guardados en '{output_path}' ---")

    #     except Exception as e:
    #         self.logger.error(f"--- [SERVER-ERROR] Error al guardar el modelo: {e} ---", exc_info=True)

    #     return history
    
    # def _get_strategy(self) -> fl.server.strategy.Strategy:
    #     """
    #     Obtiene la estrategia de agregación basada en la configuración.
    #     """
    #     strategy_name = self.config.get("aggregation", {}).get("strategy", "fedavg")
        
    #     strategy_creator = self._strategies.get(strategy_name)
    #     if not strategy_creator:
    #         raise ValueError(f"Estrategia de agregación no soportada: {strategy_name}")
        
    #     # --- CREAR PARÁMETROS INICIALES ---
    #     initial_parameters = self._create_initial_parameters()
        
    #     server_config = {
    #         "n_clients": self.n_clients,
    #         "config": self.config,
    #         "fit_config": self._fit_config,
    #         "evaluate_config": self._evaluate_config,
    #         "initial_parameters": initial_parameters  # ← AÑADIR ESTO
    #     }
        
    #     return strategy_creator.create_strategy(server_config)

    def _create_initial_parameters(self):
        """Crea los parámetros iniciales del modelo para la estrategia."""
        self.logger.info("--- [SERVER] Creando parámetros iniciales del modelo ---")
        
        try:
            # Importar el builder apropiado según el tipo de modelo
            model_type = self.config.get("model", {}).get("type", "transformer")
            
            if model_type == "transformer":
                from src.models.transformer_model_builder import TransformerModelBuilder
                # Necesitas definir input_shape basado en tu configuración
                # Por ejemplo, para datos de secuencia temporal:
                window_size = self.config.get("data", {}).get("window_size", 10)
                n_features = len(self.config.get("data", {}).get("feature_cols", []))
                input_shape = (window_size, n_features)
                builder = TransformerModelBuilder(input_shape=input_shape)
            else:
                from src.models.convlstm_model_builder import ConvLSTMModelBuilder
                # Similar para ConvLSTM
                window_size = self.config.get("data", {}).get("window_size", 10)
                n_features = len(self.config.get("data", {}).get("feature_cols", []))
                input_shape = (window_size, n_features, 1)  # ConvLSTM necesita dimensión extra
                builder = ConvLSTMModelBuilder(input_shape=input_shape)
            
            # Crear y compilar el modelo
            model = builder.build()
            
            # Convertir los pesos a parámetros de Flower
            from flwr.common import ndarrays_to_parameters
            initial_parameters = ndarrays_to_parameters(model.get_weights())
            
            self.logger.info(f"--- [SERVER] Parámetros iniciales creados para modelo {model_type} ---")
            return initial_parameters
            
        except Exception as e:
            self.logger.error(f"--- [SERVER-ERROR] Error creando parámetros iniciales: {e} ---", exc_info=True)
            raise
    

    def _fit_config(self, server_round: int) -> Dict[str, Any]:
        """
        Configura los parámetros para el entrenamiento en una ronda específica.
        Args:
            server_round (int): Número de la ronda actual del servidor.

        Returns:
            Dict[str, Any]: Diccionario con los parámetros de configuración para la ronda,
                incluyendo el tamaño de batch y las épocas locales.
        """
        try: 
            batch_size = self.config["model"]["batch_size"] # Busca bajo "model"
            local_epochs = self.config["training"]["local_epochs"] # Asumo que local_epochs SÍ está bajo "training"

            config_dict = {
                "batch_size": batch_size,
                "local_epochs": local_epochs,
                "server_round": server_round, # Puede ser útil pasar la ronda actual
            }
            # Loguear la configuración que se envía a los clientes
            # logging.info(f"Round {server_round}: Fit config sent to clients: {config_dict}")
            return config_dict

        except KeyError as e:
            logging.error(f"Missing configuration key in _fit_config: {e}", exc_info=True)
            # Decide qué hacer si falta una clave, ¿enviar config vacía o parar?
            # Podrías retornar un diccionario vacío o relanzar el error:
            # return {}
            raise ValueError(f"Configuration error in _fit_config for key: {e}") from e

    def _evaluate_config(self, server_round: int) -> Dict[str, Any]:
        """
        Configura los parámetros para la evaluación en una ronda específica.

        Args:
            server_round (int): Número de la ronda actual del servidor.

        Returns:
            Dict[str, Any]: Diccionario con los parámetros de configuración para la evaluación,
                incluyendo el tamaño de batch.
        """
        return {
            "batch_size": self.config["evaluation"]["batch_size"],
        }
