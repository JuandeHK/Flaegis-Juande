"""
Estrategia personalizada basada en media recortada.

Esta estrategia elimina valores extremos de los parámetros de los clientes
antes de calcular la media, proporcionando mayor robustez contra ataques o anomalías.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
from scipy.stats import trim_mean
from flwr.common import Parameters, Scalar, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from .base_strategy import BaseStrategy

class TrimmedMeanStrategy(BaseStrategy):
    """Estrategia de agregación basada en media recortada."""

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        """
        Inicializa la estrategia de media recortada.

        Args:
            trim_ratio (float): Proporción de valores a recortar de cada extremo.
            kwargs: Argumentos adicionales para la estrategia base.
        """
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """Agrega resultados usando media recortada."""
        if not results or (not self.accept_failures and failures):
            return None, {}

        # Extraer pesos y ejemplos
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aplicar media recortada
        stacked_weights = np.stack([weights for weights, _ in weights_results], axis=0)
        trimmed_mean_weights = np.apply_along_axis(
            lambda x: trim_mean(x, proportiontocut=self.trim_ratio), axis=0, arr=stacked_weights
        )
        parameters_aggregated = ndarrays_to_parameters(trimmed_mean_weights)

        # Agregar métricas personalizadas
        metrics_aggregated = self.aggregate_metrics(results)

        return parameters_aggregated, metrics_aggregated
