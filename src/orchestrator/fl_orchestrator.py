import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple
import threading

import numpy as np
import tensorflow as tf

from models.femnist_model_builder import FemnistModelBuilder
from src.utils.data_loader import DataLoader
from src.detectors.sign_guard_detector import SignGuardDetector
from src.utils.evaluation import DetectorEvaluator
from src.client.client_manager import FLClientManager
from src.client.client_config import ClientConfig
# from src.utils.plotting import ResultPlotter

class FederatedLearningOrchestrator:
    """
    Orquestador principal del proceso de aprendizaje federado.

    Esta clase gestiona la configuración, inicialización y ejecución del 
    aprendizaje federado, incluyendo la selección de clientes maliciosos 
    y la sincronización entre clientes y servidor.
    """

    def __init__(
        self,
        n_clients: int = 50,
        n_rounds: int = 50,
        n_malicious: int = 0,
        base_dir: str = None,
        config: Optional[dict] = None
    ):
        """
        Inicializa el orquestador del aprendizaje federado.

        Args:
            n_clients (int): Número total de clientes en el sistema.
            n_rounds (int): Número de rondas de entrenamiento.
            n_malicious (int): Número de clientes maliciosos.
            base_dir (str): Directorio base para almacenar logs y archivos temporales.
            config (Optional[dict]): Configuración del sistema cargada desde un archivo YAML.
        """
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.n_malicious = n_malicious
        self.base_dir = Path(base_dir if base_dir is not None else ".").resolve()
        self.config = config

        self.setup_logging()
        self.setup_gpu()

        # Configurar el modelo
        self.configure_model()

    @staticmethod 
    def setup_process_logging(log_filename: str):
        """Sets up logging specifically for a child process."""
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)
            h.close() 
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, mode='w') # Use 'w' to overwrite each run
            ]
        )

    @staticmethod
    def setup_gpu():
        """
        Configura el uso de la GPU para TensorFlow, permitiendo el crecimiento 
        dinámico de la memoria en lugar de asignarla por completo.
        """
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def setup_logging(self):
        """
        Configura el sistema de logging, incluyendo un archivo de log y salida en consola.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / 'federated_learning.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def select_malicious_clients(self) -> List[int]:
        """
        Selecciona aleatoriamente los clientes maliciosos y guarda sus identificadores.

        Returns:
            List[int]: Lista de identificadores de los clientes maliciosos.
        """
        if self.n_malicious == 0:
            self.logger.info("No malicious clients selected")
            return []
            
        malicious_clients = sorted(np.random.choice(
            range(self.n_clients), 
            size=self.n_malicious, 
            replace=False
        ))
        
        self.logger.info(f"Selected malicious clients: {malicious_clients}")
        np.save(self.base_dir / "nodos_comprometidos.npy", malicious_clients)
        return malicious_clients

    def clean_previous_run(self):
        """
        Limpia los archivos y directorios de ejecuciones anteriores, como modelos y datos temporales.
        """
        dirs_to_clean = ['modelos_att', 'check_arrays']
        for dir_name in dirs_to_clean:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                for file in dir_path.iterdir():
                    if file.is_dir():
                        file.rmdir()
                    else:
                        file.unlink()

        # Inicializa el archivo de precisión del detector
        detector_file = self.base_dir / 'detector_accuracy.csv'
        detector_file.write_text('Accuracy\n')

    def configure_model(self):
        """
        Configura el modelo utilizando los valores definidos en el archivo YAML.
        
        Los parámetros incluyen:
        - Tipo de modelo.
        - Tasa de aprendizaje.
        - Tamaño de batch.
        """
        try:
            model_config = self.config.get("model", {})
            
            # Validar las claves necesarias
            required_keys = ["type", "learning_rate", "batch_size"]
            for key in required_keys:
                if key not in model_config:
                    raise ValueError(f"Missing required model configuration key: {key}")
            
            # Configurar los parámetros del modelo
            self.model_type = model_config["type"]
            self.learning_rate = model_config["learning_rate"]
            self.batch_size = model_config["batch_size"]

            self.logger.info(f"Model configured: {self.model_type} "
                            f"(learning_rate={self.learning_rate}, batch_size={self.batch_size})")

        except Exception as e:
            self.logger.error(f"Failed to configure model: {e}")
            raise

    def initialize_clients(self, malicious_clients: List[int]) -> Tuple[multiprocessing.Process, List[multiprocessing.Process]]:
        """
        Inicializa los procesos de los clientes y del servidor.

        Args:
            malicious_clients (List[int]): Lista de identificadores de los clientes maliciosos.

        Returns:
            Tuple[multiprocessing.Process, List[multiprocessing.Process]]:
                Proceso del servidor y lista de procesos de clientes.
        """
        clients = []
        server_ready_event = multiprocessing.Event()
        
        try:
            # Inicia el servidor
            server = multiprocessing.Process(
                target=self.start_server,
                args=(self.n_clients, self.n_rounds, self.config, server_ready_event)
            )
            server.start()
            self.logger.info("Server process started")
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            raise

        # Espera a que el servidor indique que está listo
        # En la función initialize_clients de FLOrchestrator
        # ...
        # Espera a que el servidor indique que está listo
        server_ready_event.wait()
        self.logger.info("Server is ready, starting clients")

        # Inicia los clientes
        for i in range(self.n_clients):
            is_malicious = (i in malicious_clients)
            client_id_str = f"client_process_{i}"
            
            client_config = ClientConfig(
                client_id=client_id_str,
                numeric_id=i,
                is_malicious=is_malicious,
            )
            
            client_proc = multiprocessing.Process(
                target=FederatedLearningOrchestrator.start_client,
                args=((client_config, self.config),),
                name=client_id_str
            )
            client_proc.start()
            clients.append(client_proc)
            self.logger.info(f"Started client {i}")

        return server, clients

    @staticmethod
    def start_server(n_clients: int, n_rounds: int, config: dict, server_ready_event: multiprocessing.Event):
        """Inicia el proceso del servidor con logging detallado."""
        log_dir = Path(config.get("data", {}).get("base_dir", "data/")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        server_log_file = log_dir / 'server_process.log'
        FederatedLearningOrchestrator.setup_process_logging(str(server_log_file))

        process_name = multiprocessing.current_process().name
        logging.info(f"SERVER_PROC ({process_name}): Starting execution.")

        try:
            logging.info(f"SERVER_PROC ({process_name}): Importing FederatedServer...")
            from src.server.server_manager import FederatedServer
            logging.info(f"SERVER_PROC ({process_name}): Import successful.")
            
            logging.info(f"SERVER_PROC ({process_name}): Creating FederatedServer instance...")
            server = FederatedServer(
                n_clients=n_clients,
                n_rounds=n_rounds,
                config=config
            )
            logging.info(f"SERVER_PROC ({process_name}): Instance created successfully.")

            logging.info(f"SERVER_PROC ({process_name}): Signaling server_ready_event...")
            server_ready_event.set()
            logging.info(f"SERVER_PROC ({process_name}): Event set.")

            logging.info(f"SERVER_PROC ({process_name}): Calling server.start() -> This is a blocking call...")
            server.start()
            logging.info(f"SERVER_PROC ({process_name}): server.start() has finished (all rounds completed).")

        except Exception as e:
            logging.error(f"SERVER_PROC ({process_name}): !!! CRITICAL ERROR !!!", exc_info=True)
            raise
        finally:
            logging.info(f"SERVER_PROC ({process_name}): Terminating.")

    @staticmethod
    def start_client(client_config_tuple):
        """
        Inicia un proceso de cliente FL.

        Args:
            client_config_tuple: Tupla que contiene (client_config, global_config)
        """
        try:
            client_config, global_config = client_config_tuple
            
            # Configurar logging para el cliente
            log_dir = Path(global_config.get("data", {}).get("base_dir", "data/")).resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            client_log_file = log_dir / f'client_{client_config.numeric_id}.log'
            FederatedLearningOrchestrator.setup_process_logging(str(client_log_file))
            
            logging.info(f"Cliente {client_config.numeric_id}: Iniciando proceso")
            
            # Inicializar y arrancar el cliente
            client_manager = FLClientManager(
                base_dir=global_config.get('data', {}).get('base_dir', ''),
                clientConfig=client_config,
                config=global_config
            )
            client_manager.start()
            
        except Exception as e:
            client_id = client_config.numeric_id if 'client_config' in locals() else 'unknown'
            logging.error(f"Error in client {client_id}: {e}", exc_info=True)
            raise

    def verify_environment(self):
        """
        Verifica que el entorno esté configurado correctamente antes de la ejecución.

        - Crea directorios requeridos si no existen.
        - Verifica la configuración de GPU (opcional).
        - Asegura que la configuración YAML sea válida.
        """
        # Verificar directorios requeridos
        required_dirs = ['modelos_att', 'check_arrays']
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                self.logger.warning(f"Directory {dir_path} not found. Creating it.")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar configuración YAML
        if not self.config:
            raise ValueError("Configuration not loaded. Ensure the YAML file is valid.")
        
        # (Opcional) Verificar la configuración de GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            self.logger.warning("No GPU devices found. The program will run on CPU, which may impact performance.")
        else:
            self.logger.info(f"Found {len(physical_devices)} GPU(s).")

        self.logger.info("Environment verification completed successfully.")

    def run(self) -> List[int]:
        """
        Ejecuta el proceso de aprendizaje federado completo.

        Returns:
            List[int]: Lista de clientes maliciosos seleccionados.
        """
        start_time = time.perf_counter()
        self.logger.info(f"Starting federated learning with {self.n_clients} clients")
        
        # Verificar el entorno antes de comenzar
        self.verify_environment()
        
        # Limpiar ejecuciones anteriores
        self.clean_previous_run()
        
        # Seleccionar clientes maliciosos
        malicious_clients = self.select_malicious_clients()
        
        # Inicializar clientes y servidor
        server, clients = self.initialize_clients(malicious_clients)
        
        # Esperar a que terminen todos los procesos
        server.join()
        for client in clients:
            client.join()
        
        end_time = time.perf_counter()
        self.logger.info(f"Federated learning process completed in {end_time - start_time:.2f} seconds")
        return malicious_clients

    # def run(self) -> List[int]:
    #     """Ejecuta el proceso de aprendizaje federado y genera los plots al final."""
    #     start_time = time.perf_counter()
    #     self.logger.info(f"Iniciando aprendizaje federado con {self.n_clients} clientes")
        
    #     self.verify_environment()
        
    #     malicious_clients = self.select_malicious_clients()
    #     server, clients = self.initialize_clients(malicious_clients)
        
    #     # Esperamos a que el servidor y los clientes terminen su ejecución
    #     server.join()
    #     for client in clients:
    #         client.join()
        
    #     self.logger.info("El entrenamiento federado ha finalizado.")

    #     # --- SECCIÓN NUEVA PARA PLOTEAR ---
    #     try:
    #         self.logger.info("Iniciando fase de evaluación y ploteo final...")
    #         plotter = ResultPlotter(config=self.config, base_dir=str(self.base_dir))
    #         plotter.generate_plots()
    #     except Exception as e:
    #         self.logger.error(f"Falló la generación de plots post-entrenamiento: {e}", exc_info=True)
    #     # --- FIN DE LA SECCIÓN ---
        
    #     end_time = time.perf_counter()
    #     self.logger.info(f"Proceso completo finalizado en {end_time - start_time:.2f} segundos")
    #     return malicious_clients