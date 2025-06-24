"""
Módulo de estrategias personalizadas para aprendizaje federado.

Este módulo proporciona implementaciones completas de estrategias de agregación
que extienden las capacidades básicas de Flower, añadiendo funcionalidades como
detección de nodos maliciosos y métodos robustos de agregación.

Available Strategies:
    - MedianStrategy: Estrategia básica basada en mediana simple
    - WeightedMedianStrategy: Estrategia robusta basada en mediana ponderada
    - FourierStrategy: Estrategia basada en análisis de Fourier
    - TrimmedMeanStrategy: Estrategia basada en media recortada
"""
from .base_strategy import BaseStrategy
#from ..custom_strategies.weighted_median_strategy import MedianStrategy
from .weighted_median_strategy import WeightedMedianStrategy
from .fourier_strategy import FourierStrategy
from .trimmed_mean_strategy import TrimmedMeanStrategy

__all__ = [
    'BaseStrategy',
    #'CustomStrategy',
    #'MedianStrategy',
    'WeightedMedianStrategy',
    'FourierStrategy',
    'TrimmedMeanStrategy'
]

# Version del módulo
__version__ = '1.0.0'

# Información adicional
__author__ = 'Enrique Mármol and Francisco José Cortés Delgado'
__email__ = 'franciscojose.cortesd@um.es'
__description__ = 'Estrategias de agregación para aprendizaje federado'