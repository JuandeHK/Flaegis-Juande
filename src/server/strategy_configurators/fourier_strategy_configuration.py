"""
Implementación del configurador de la estrategia basada en transformada de Fourier.

Este módulo implementa un configurador para la estrategia de agregación que utiliza 
análisis de Fourier para combinar los modelos de los clientes, lo que puede ayudar 
a identificar y filtrar componentes anómalos en el espacio de frecuencias.
"""

from typing import Dict, Any
import flwr as fl
from .base_strategy_configurator import StrategyConfigurator
from src.custom_strategies.fourier_strategy import FourierStrategy

class FourierStrategyConfigurator(StrategyConfigurator):
    """
    Configurador de la estrategia basada en transformada de Fourier.
    
    Configura una estrategia que utiliza análisis de Fourier para combinar
    los modelos de los clientes, permitiendo un filtrado más sofisticado
    de las contribuciones de los clientes en el dominio de la frecuencia.
    """
    
    def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
        """
        Crea una instancia de la estrategia basada en Fourier.

        Args:
            server_config (Dict[str, Any]): Configuración del servidor incluyendo:
                - n_clients: Número de clientes
                - config: Configuración general con parámetros específicos
                - fit_config: Función de configuración para entrenamiento
                - evaluate_config: Función de configuración para evaluación

        Returns:
            fl.server.strategy.Strategy: Estrategia Fourier configurada
        """
        strategy_params = server_config["config"].get("aggregation", {}).get("params", {})
        return FourierStrategy(
            min_fit_clients=server_config["n_clients"],
            min_evaluate_clients=server_config["n_clients"],
            **strategy_params
        ) 