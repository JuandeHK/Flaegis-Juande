import numpy as np
from .clustering_strategy import ClusteringStrategy
from typing import List

class ClusteringContext:
    """Contexto que usa una estrategia de clustering para detectar clientes maliciosos."""
    
    def __init__(self, strategy: ClusteringStrategy):
        self._strategy = strategy
    
    @property
    def strategy(self) -> ClusteringStrategy:
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: ClusteringStrategy):
        self._strategy = strategy
    
    def detect_malicious_clients(self, similarity_matrix: np.ndarray, clients: List[int]) -> List[int]:
        """Usa la estrategia actual para detectar clientes maliciosos."""
        return self._strategy.detect(similarity_matrix, clients)