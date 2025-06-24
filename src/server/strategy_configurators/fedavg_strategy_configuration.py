# Fichero: src/server/strategy_configurators/fedavg_strategy_configuration.py

from typing import Dict, Any, List, Tuple
import flwr as fl
from flwr.common import Metrics
from .base_strategy_configurator import StrategyConfigurator

# --- 1. Definimos nuestra propia función de agregación "inteligente" ---
def aggregate_weighted_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calcula la media ponderada para cada métrica que envían los clientes.
    Esta función SÍ sabe cómo manejar diccionarios como {'loss': 0.1, 'mae': 0.2}.
    """
    if not metrics:
        return {}

    # Extraer todas las claves de las métricas de todos los clientes (ej: 'loss', 'mae')
    all_keys = {key for _, m in metrics for key in m.keys()}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {}

    # Calcular la media ponderada para cada métrica
    aggregated_metrics = {}
    for key in all_keys:
        weighted_sum = sum(m.get(key, 0.0) * num_examples for num_examples, m in metrics)
        aggregated_metrics[key] = weighted_sum / total_examples
    
    return aggregated_metrics

class FedAvgStrategyConfigurator(StrategyConfigurator):
    """Configurador para FedAvg que usa nuestra función de agregación personalizada."""
    
    def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
        """Crea una instancia de FedAvg, pasándole nuestra función de agregación."""
        
        return fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=server_config["n_clients"],
            min_evaluate_clients=server_config["n_clients"],
            min_available_clients=server_config["n_clients"],
            on_fit_config_fn=server_config["fit_config"],
            on_evaluate_config_fn=server_config["evaluate_config"],
            
            # --- 2. Usamos nuestra nueva función "inteligente" para ambas tareas ---
            fit_metrics_aggregation_fn=aggregate_weighted_metrics,
            evaluate_metrics_aggregation_fn=aggregate_weighted_metrics,
        )

