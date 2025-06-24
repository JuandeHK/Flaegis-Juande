"""
Implementación del configurador de la estrategia de media recortada.

Este módulo configura una estrategia basada en la media recortada, que elimina
valores extremos antes de la agregación para mayor robustez.
"""

from typing import Dict, Any
import flwr as fl
from .base_strategy_configurator import StrategyConfigurator
from src.custom_strategies.trimmed_mean_strategy import TrimmedMeanStrategy

class TrimmedMeanStrategyConfigurator(StrategyConfigurator):
    """
    Configurador de la estrategia de media recortada.

    Configura una estrategia que elimina un porcentaje de valores extremos
    antes de calcular la media de los parámetros de los clientes.
    """
    def __init__(self, trim_ratio: float = 0.1):
        """
        Inicializa el configurador de la estrategia de media recortada.

        Args:
            trim_ratio (float): Proporción de valores a recortar de cada extremo.
                                Por defecto es 0.1 (10% de cada lado).
        """
        self.trim_ratio = trim_ratio

    def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
        """
        Crea una instancia de la estrategia de media recortada.

        Args:
            server_config (Dict[str, Any]): Configuración del servidor, incluyendo:
                - n_clients: Número de clientes.
                - fit_config: Configuración para entrenamiento.
                - evaluate_config: Configuración para evaluación.

        Returns:
            fl.server.strategy.Strategy: Estrategia configurada para usar media recortada.
        """
        strategy_params = server_config["config"].get("aggregation", {}).get("params", {})
        return TrimmedMeanStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=server_config["n_clients"],
            min_evaluate_clients=server_config["n_clients"],
            min_available_clients=server_config["n_clients"],
            on_fit_config_fn=server_config.get("fit_config"),
            on_evaluate_config_fn=server_config.get("evaluate_config"),
            trim_ratio=self.trim_ratio,
            **strategy_params
        )

