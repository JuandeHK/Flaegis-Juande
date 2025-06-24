import numpy as np
from typing import List
from .clustering_strategy import ClusteringStrategy

class BaseClusteringDetector(ClusteringStrategy):
    """Clase base para detectores basados en clustering que implementa la lógica común."""
    
    def _classify_clusters(self, clusters: np.ndarray, clients: List[int]) -> List[int]:
        """Clasifica los clusters en buenos y malos basado en su tamaño."""
        indx1 = np.where(clusters == 0)[0]
        indx2 = np.where(clusters == 1)[0]
        cluster1 = np.take(clients, indx1)
        cluster2 = np.take(clients, indx2)
        
        return sorted(cluster2) if len(cluster1) > len(cluster2) else sorted(cluster1)