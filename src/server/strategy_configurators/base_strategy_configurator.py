"""
Módulo que define la interfaz base para los configuradores de estrategias.

Este módulo contiene la clase abstracta que todos los configuradores de estrategias
deben implementar, asegurando una interfaz consistente para la configuración de
estrategias en el sistema de aprendizaje federado.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import flwr as fl

class StrategyConfigurator(ABC):
    """
    Interfaz base para configuradores de estrategias.
    
    Esta clase abstracta define el contrato que todos los configuradores de estrategias
    deben seguir, proporcionando un método común para crear estrategias específicas
    de Flower.
    """
    
    @abstractmethod
    def create_strategy(self, server_config: Dict[str, Any]) -> fl.server.strategy.Strategy:
        """
        Crea una instancia de estrategia de Flower basada en la configuración proporcionada.
        """
        pass 