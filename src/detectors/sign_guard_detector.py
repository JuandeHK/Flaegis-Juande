import numpy as np
from sklearn.cluster import MeanShift
from typing import List, Tuple, Union

class SignGuardDetector:
    """
    Detector de clientes maliciosos basado en el enfoque SignGuard.

    Este detector utiliza un enfoque de doble filtro para identificar clientes maliciosos 
    en sistemas de aprendizaje federado. Los filtros incluyen:
    
    1. **Normas de los Gradientes**:
        - Filtra clientes cuyos gradientes tienen normas inusualmente altas o bajas.
    2. **Estadísticas de Signos**:
        - Agrupa clientes según los signos de sus gradientes usando MeanShift clustering.
        - Identifica el menor grupo como el conjunto de clientes maliciosos.

    Este enfoque garantiza que tanto los gradientes aberrantes como aquellos con 
    direcciones opuestas sean detectados y eliminados, protegiendo así el modelo global.

    """

    def __init__(self):
        """
        Inicializa el detector SignGuard.

        Crea una instancia de `MeanShift` para realizar agrupaciones en las estadísticas
        de signos de los gradientes.
        """
        self.meanshift = MeanShift()
    
    def detect(self, gradients: List[np.ndarray], party_numbers: List[int]) -> List[int]:
        """
        Detecta clientes maliciosos usando el algoritmo SignGuard.
        
        Args:
            gradients: Lista de gradientes de cada cliente
            party_numbers: Lista de identificadores de clientes
            
        Returns:
            List[int]: Lista de clientes identificados como maliciosos
        """
        normas = [self._get_norm(grad) for grad in gradients]
        malos_1 = self._detect_norm_outliers(normas, party_numbers)
        
        signos = [self._get_sign_statistics(grad) for grad in gradients]
        malos_2 = self._detect_sign_outliers(signos, party_numbers)
        
        return list(np.union1d(malos_1, malos_2))
    def _get_norm(self, weights: List[np.ndarray]) -> float:
        """
        Calcula la norma media de los pesos proporcionados.

        Este método calcula la norma euclidiana (L2) promedio de todos los tensores
        de pesos proporcionados. Se utiliza como parte del primer filtro del detector
        SignGuard para identificar gradientes con magnitudes anómalas.

        Args:
            weights (List[np.ndarray]): Lista de arrays NumPy que representan los
                pesos/gradientes del modelo.

        Returns:
            float: Norma media calculada de todos los tensores de pesos.
        """
        normas = []
        for w in weights:
            if w.ndim == 1:
                normas.append(np.linalg.norm(w))
            else:
                normas.append(np.linalg.norm(w.reshape(1, -1)))
        return np.mean(normas)
    
    def _detect_norm_outliers(self, norms: List[float], party_numbers: List[int]) -> List[int]:
        """
        Identifica clientes maliciosos según normas aberrantes.

        Este método detecta clientes cuyas normas son valores atípicos 
        (muy altos o muy bajos) en comparación con el resto.

        Args:
            norms (List[float]): Lista de normas de gradientes, una por cliente.
            party_numbers (List[int]): Identificadores únicos de los clientes.

        Returns:
            List[int]: Lista de identificadores de clientes con normas aberrantes.
        """
        malos_1 = [num for num, norma in zip(party_numbers, norms) if (norma/np.median(norms) < 0.1) or (norma/np.median(norms))>3]
        return malos_1

    
    def _get_sign_statistics(self, weights: List[np.ndarray]) -> Tuple[float, float]:
        """
        Calcula estadísticas de signos de los gradientes.

        Este método analiza los signos de los gradientes enviados por un cliente 
        para extraer patrones que podrían indicar comportamiento malicioso.

        Args:
            weights (List[np.ndarray]): Gradientes del cliente.

        Returns:
            Tuple[float, float]: Estadísticas de signos:
                - Proporción de valores positivos.
                - Proporción de valores negativos.
        """
        sig = []
        for i in range(len(weights)):
            # Aplica 'numpy.sign' para obtener los valores -1, 0 y 1
            signed_arr = np.sign(weights[i])
        #print(signed_arr)
        if signed_arr.ndim >1:
            signed_arr = signed_arr.reshape(1,-1)
            # Usa 'numpy.bipncount' para contar la cantidad de -1, 0 y 1
            counts = contar_elementos(signed_arr)


        if signed_arr.ndim ==1:
            counts = contar_elementos(signed_arr)

        # Los índices 0, 1 y 2 en 'counts' representan -1, 0 y 1 respectivamente
        count_1 = counts[0]
        count_0 = counts[1]
        count_neg_1 = counts[2]

        sig.append([count_1,count_0,count_neg_1])
        sig_total = np.sum(sig, axis=0)
        suma = sig_total[0] + sig_total[1] + sig_total[2]
        sig_total = sig_total/suma
        return sig_total

    
    def _detect_sign_outliers(self, sign_stats: List[Tuple[float, float]], party_numbers: List[int]) -> List[int]:
        """
        Identifica clientes maliciosos usando agrupamiento en estadísticas de signos.

        Este método agrupa clientes en base a las estadísticas de sus signos (positivos/negativos).
        Los clientes que pertenezcan al grupo más pequeño son considerados maliciosos.

        Args:
            sign_stats (List[Tuple[float, float]]): Estadísticas de signos de cada cliente.
            party_numbers (List[int]): Identificadores únicos de los clientes.

        Returns:
            List[int]: Identificadores de clientes maliciosos según las estadísticas de signos.
        """
        
        meanshift = MeanShift()
        meanshift.fit(sign_stats)
        labels = meanshift.labels_

        counts = np.bincount(labels)
        valor_mas_repetido = np.argmax(counts)
        indices_valor_mas_repetido = np.where(labels == valor_mas_repetido)
        party_numbers = np.array(party_numbers)
        buenos_2 = party_numbers[indices_valor_mas_repetido]

        malos_2 = list(set(party_numbers) - set(buenos_2))
        if len(malos_2)>len(buenos_2):
            malos_2=buenos_2

      
        return malos_2