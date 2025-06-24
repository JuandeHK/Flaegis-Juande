import flwr as fl
import logging
from src.models.model_builder import ModelBuilder
from src.client.client_config import ClientConfig
from src.models.convlstm_model_builder import ConvLSTMModelBuilder
from src.models.transformer_model_builder import TransformerModelBuilder
from src.models.LSTMModelBuilder import LSTMModelBuilder

from src.utils.data_loader import DataLoader
from .fl_client import FLClient

class FLClientManager:
    """Gestiona la configuración y ejecución del cliente de aprendizaje federado (FL)."""

    def __init__(self, base_dir: str, clientConfig: ClientConfig, config: dict):
        self.client_config = clientConfig
        self.global_config = config
        self.logger = logging.getLogger(f"ClientManager-{self.client_config.numeric_id}")
        
        try:
            # Obtenemos las configuraciones
            data_config = self.global_config.get('data', {}).copy()
            model_config = self.global_config.get("model", {})
            self.model_type_str = model_config.get("type", "convlstm")

            # **Añadimos el tipo de modelo a la config de datos** para que el DataLoader sepa qué hacer
            data_config['type'] = self.model_type_str
            
            self.logger.info(f"Inicializando DataLoader con config: {data_config}")
            self.data_loader = DataLoader(base_dir, data_config=data_config)
            
            self.logger.info(f"Preparando constructor de modelo tipo: {self.model_type_str}")
            self.model_builder = self._buildModel(self.model_type_str, model_config)
            
        except Exception as e:
            self.logger.error(f"Error en inicialización: {e}", exc_info=True)
            raise
            
    def start(self):
        """Inicia el cliente FL."""
        self.logger.info("=== INICIANDO CLIENTE ===")
        try:
            self.logger.info("Cargando datos...")
            # La llamada a load_data_for_client ahora es más simple aquí
            train_data, test_data = self.data_loader.load_data_for_client(self.client_config.numeric_id)
            
            self.logger.info(f"Datos cargados - Train: {train_data[0].shape}, Test: {test_data[0].shape}")
            if train_data[0].size == 0: raise ValueError("No hay datos de entrenamiento disponibles")
            
            self.logger.info("Construyendo modelo...")
            model = self.model_builder.build()
            model.summary()
            
            client = FLClient(model, train_data, test_data, self.client_config.is_malicious, self.client_config.numeric_id)
            server_address = self.global_config.get("server_address", "127.0.0.1:8080")
            fl.client.start_numpy_client(server_address=server_address, client=client)
                
        except Exception as e:
            self.logger.error(f"Error general en cliente: {e}", exc_info=True)
            raise
        finally:
            self.logger.info("Cliente finalizó su ciclo de ejecución.")
        
    def _buildModel(self, model_type: str, model_config: dict) -> ModelBuilder:
        """Construye el constructor de modelo correcto según el tipo."""
        self.logger.info(f"_buildModel: tipo='{model_type}', config={model_config}")
        data_cfg = self.global_config.get('data', {})
        num_features = len(data_cfg.get('feature_cols', []))
        window_size = data_cfg.get('window_size', 12)

        if model_type == "convlstm":
            input_shape = (window_size, 1, num_features, 1)
            return ConvLSTMModelBuilder(input_shape=input_shape)
        elif model_type == "transformer":
            input_shape = (window_size, num_features)
            return TransformerModelBuilder(input_shape=input_shape)
        elif model_type == "lstm":
            input_shape = (window_size, num_features)
            return LSTMModelBuilder(input_shape=input_shape)
        else:
            raise ValueError(f"Tipo de modelo '{model_type}' no soportado.")
