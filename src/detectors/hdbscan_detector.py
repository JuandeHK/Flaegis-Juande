import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from .base_clustering_detector import BaseClusteringDetector
from typing import List

class HierarchicalDetector(BaseClusteringDetector):
    """
    Detector de clientes maliciosos basado en clustering jerárquico.

    Este detector utiliza un enfoque de clustering jerárquico para identificar 
    patrones entre los clientes en un entorno de aprendizaje federado. 
    El método `linkage` genera una matriz de enlaces utilizando la similitud entre clientes.
    Si se detectan múltiples clusters, se aplica K-Means para separar los clientes 
    en dos grupos y clasifica a los clientes como "buenos" o "malos" según el tamaño 
    de los clusters.

    Hereda:
        BaseClusteringDetector: Clase base que define la interfaz común para 
        detectores basados en clustering.
    """

    def __init__(self, threshold: float = 0.6):
        """
        Inicializa el detector jerárquico con un umbral para cortar el dendrograma.

        Args:
            threshold (float): Umbral de distancia para determinar los clusters. 
                Los clientes se agrupan en clusters según este criterio.
        """
        self.threshold = threshold
        
    def detect(self, similarity_matrix: np.ndarray, clients: List[int]) -> List[int]:
        """
        Detecta clientes maliciosos utilizando clustering jerárquico y K-Means.

        Args:
            similarity_matrix (np.ndarray): Matriz de similitud precomputada entre clientes.
            clients (List[int]): Lista de identificadores de los clientes.

        Returns:
            List[int]: Lista de identificadores de clientes clasificados como maliciosos.
        """
        linkage_matrix = linkage(similarity_matrix, method='average')
        clusters = fcluster(linkage_matrix, t=self.threshold, criterion='distance')
        
        if len(np.unique(clusters)) > 1:
            kmeans = KMeans(n_clusters=2)
            clusters = kmeans.fit_predict(similarity_matrix)
            return self._classify_clusters(clusters, clients)
        return []
