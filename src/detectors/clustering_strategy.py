from abc import ABC, abstractmethod
import numpy as np
from typing import List

class ClusteringStrategy(ABC):
    """Interfaz para estrategias de detección de clustering."""
    
    @abstractmethod
    def detect(self, similarity_matrix: np.ndarray, clients: List[int]) -> List[int]:
        """Método abstracto para detectar clientes maliciosos usando clustering."""
        pass