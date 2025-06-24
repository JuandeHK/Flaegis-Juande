import argparse
import logging
import yaml
from pathlib import Path
import multiprocessing

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Carga la configuración desde un archivo YAML y valida que contenga las claves necesarias.

    Args:
        config_path (str): Ruta del archivo de configuración YAML. Por defecto, "config/config.yaml".

    Returns:
        dict: Configuración cargada como un diccionario.

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe en la ruta proporcionada.
        yaml.YAMLError: Si ocurre un error al analizar el archivo YAML.
        ValueError: Si faltan claves necesarias en el archivo de configuración.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validar que todas las claves necesarias estén presentes
        required_keys = ["training", "model", "detection"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in YAML configuration: {key}")

        # Validar subclaves dentro de 'training', 'model' y 'detection'
        training_keys = ["rounds", "clients", "min_fit_clients", "min_eval_clients"]
        for key in training_keys:
            if key not in config["training"]:
                raise ValueError(f"Missing key in 'training' section: {key}")

        model_keys = ["type", "learning_rate", "batch_size"]
        for key in model_keys:
            if key not in config["model"]:
                raise ValueError(f"Missing key in 'model' section: {key}")

        detection_keys = ["threshold", "min_clients_per_cluster"]
        for key in detection_keys:
            if key not in config["detection"]:
                raise ValueError(f"Missing key in 'detection' section: {key}")

        return config

    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise



def validate_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """
    Valida los argumentos de línea de comandos para garantizar que sean consistentes y válidos.

    Args:
        args (argparse.Namespace): Argumentos parseados de la línea de comandos.
        parser (argparse.ArgumentParser): Parser de argumentos para mostrar errores personalizados.

    Raises:
        ArgumentError: Si alguno de los argumentos no cumple las restricciones.
    """
    # Validar que m no sea mayor que el número total de clientes
    if args.m >= args.c:
        parser.error(f"Number of malicious clients (-m={args.m}) must be less than total clients ({args.c})")
    
    # Validar que m no sea negativo
    if args.m < 0:
        parser.error(f"Number of malicious clients (-m={args.m}) cannot be negative")
    
    # Validar que c sea positivo
    if args.c <= 0:
        parser.error(f"Number of clients (-c={args.c}) must be positive")


def parse_arguments(config: dict):
    """
    Parsea los argumentos de línea de comandos y aplica validaciones.

    Args:
        config (dict): Configuración cargada desde un archivo YAML, usada para valores predeterminados.

    Returns:
        argparse.Namespace: Argumentos parseados y validados.
    """
    parser = argparse.ArgumentParser(description='Federated Learning Orchestrator')
    parser.add_argument(
        "-m",
        help="Number of malicious clients (must be less than total clients). Default is 0.",
        type=int,
        default=0
    )
    parser.add_argument(
        "-r",
        help=f"Number of rounds. Default: {config['training']['rounds']}.",
        type=int,
        default=config['training']['rounds']
    )
    parser.add_argument(
        "-c",
        help=f"Total number of clients. Default: {config['training']['clients']}.",
        type=int,
        default=config['training']['clients']
    )
    args = parser.parse_args()
    
    # Validar argumentos
    validate_arguments(args, parser)
    
    return args


def main():
    """
    Punto de entrada principal del programa.

    Maneja la carga de la configuración, el parseo de los argumentos de línea de comandos y la inicialización del orquestador.
    También captura y maneja errores comunes durante la ejecución.

    Raises:
        SystemExit: Si ocurre un error fatal durante la ejecución.
    """
    multiprocessing.set_start_method('spawn', force=True)

    try:
        
        # Cargar configuración
        config = load_config()
        
        # Parsear argumentos usando valores de configuración como defaults
        args = parse_arguments(config)
        
        from src.orchestrator.fl_orchestrator import FederatedLearningOrchestrator
        
        orchestrator = FederatedLearningOrchestrator(
        n_clients=args.c,
        n_rounds=args.r,
        n_malicious=args.m,
        base_dir=config.get("data", {}).get("base_dir", "data/"),
        config=config
    )

        
        malicious_clients = orchestrator.run()

        print("\n" + "="*60)
        print(f"  - Modelo utilizado:         {config['model']['type']}")
        print("="*60 + "\n")

        #print(f"Completed successfully. Malicious clients: {malicious_clients}")
        
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise SystemExit(1)
    except RuntimeError as e:
        logging.error(f"GPU initialization error: {e}")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"Unexpected error during execution: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
