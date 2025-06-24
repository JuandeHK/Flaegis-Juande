import numpy as np
from sklearn.cluster import SpectralClustering
from .base_clustering_detector import BaseClusteringDetector
from typing import List

class SpectralDetector(BaseClusteringDetector):
    """
    Detector de clientes maliciosos basado en clustering espectral.

    Este detector utiliza un enfoque de clustering espectral para identificar 
    clientes maliciosos en un entorno de aprendizaje federado. La técnica de 
    clustering espectral utiliza una matriz de similitud precomputada para separar 
    los clientes en dos grupos, clasificándolos como "buenos" o "malos" según 
    el tamaño de los clusters.

    Hereda:
        BaseClusteringDetector: Clase base que define la interfaz común para 
        detectores basados en clustering.
    """

    def detect(self, similarity_matrix: np.ndarray, clients: List[int]) -> List[int]:
        """
        Detecta clientes maliciosos utilizando clustering espectral.

        Args:
            similarity_matrix (np.ndarray): Matriz de similitud precomputada entre clientes.
            clients (List[int]): Lista de identificadores de los clientes.

        Returns:
            List[int]: Lista de identificadores de clientes clasificados como maliciosos.
        """
        clustering = SpectralClustering(n_clusters=2, affinity='precomputed')
        clusters = clustering.fit_predict(similarity_matrix)
        return self._classify_clusters(clusters, clients)
