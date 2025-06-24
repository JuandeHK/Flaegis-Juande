"""
Implementación del configurador de la estrategia de mediana ponderada.

Este módulo implementa un configurador para la estrategia de agregación robusta basada en la
mediana ponderada, que es más resistente a valores atípicos y ataques
que la media simple.
"""

from typing import Dict, Any
import flwr as fl
from .base_strategy_configurator import StrategyConfigurator
from src.custom_strategies.weighted_median_strategy import WeightedMedianStrategy

class WeightedMedianStrategyConfigurator(StrategyConfigurator):
    """
    Configurador de la estrategia de mediana ponderada.
    
    Configura una estrategia de agregación que utiliza la mediana ponderada
    para combinar los modelos de los clientes, proporcionando mayor robustez
    contra clientes maliciosos.
    """

    def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
        """Crea una instancia de la estrategia de mediana ponderada."""
        strategy_params = server_config["config"].get("aggregation", {}).get("params", {})
        return WeightedMedianStrategy(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=server_config["n_clients"],
            min_evaluate_clients=server_config["n_clients"],
            min_available_clients=server_config["n_clients"],
            on_fit_config_fn=server_config["fit_config"],
            on_evaluate_config_fn=server_config["evaluate_config"],
            **strategy_params
        )

    # def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
    #     """
    #     Crea una instancia de la estrategia de mediana ponderada.

    #     Args:
    #         server_config (Dict[str, Any]): Configuración del servidor incluyendo:
    #             - n_clients: Número de clientes
    #             - config: Configuración general con parámetros específicos
    #             - fit_config: Función de configuración para entrenamiento
    #             - evaluate_config: Función de configuración para evaluación

    #     Returns:
    #         fl.server.strategy.Strategy: Estrategia de mediana ponderada configurada
    #     """
    #     strategy_params = server_config["config"].get("aggregation", {}).get("params", {})
    #     return WeightedMedianStrategy(
    #         fraction_fit=1.0,
    #         fraction_evaluate=1.0,
    #         min_fit_clients=server_config["n_clients"],
    #         min_evaluate_clients=server_config["n_clients"],
    #         min_available_clients=server_config["n_clients"],
    #         on_fit_config_fn=server_config["fit_config"],
    #         on_evaluate_config_fn=server_config["evaluate_config"],
    #         **strategy_params
    #     ) 