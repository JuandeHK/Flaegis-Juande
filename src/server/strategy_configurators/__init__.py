"""
Módulo de configuradores de estrategias para el aprendizaje federado.

Este módulo proporciona diferentes implementadores de configuración de estrategias
que pueden ser utilizadas en el proceso de aprendizaje federado.
"""

from .base_strategy_configurator import StrategyConfigurator
from .fedavg_strategy_configuration import FedAvgStrategyConfigurator
from .weighted_median_strategy_configuration import WeightedMedianStrategyConfigurator
from .fourier_strategy_configuration import FourierStrategyConfigurator
from .trimmed_mean_strategy_configuration import TrimmedMeanStrategyConfigurator

__all__ = [
    'StrategyConfigurator',
    'FedAvgStrategyConfigurator',
    'WeightedMedianStrategyConfigurator',
    'FourierStrategyConfigurator',
    'TrimmedMeanStrategyConfigurator'
] 