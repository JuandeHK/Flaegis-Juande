"""
Estrategia personalizada basada en transformada de Fourier.

Esta estrategia utiliza análisis de Fourier para la agregación de modelos,
permitiendo identificar y filtrar componentes anómalos en el dominio de la frecuencia.
"""
from typing import List, Tuple, Union, Optional
import numpy as np
from flwr.common import Parameters, Scalar, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from .base_strategy import BaseStrategy

class FourierStrategy(BaseStrategy):
    """Estrategia de agregación basada en transformada de Fourier."""

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """Agrega resultados usando transformada de Fourier."""
        if not results or (not self.accept_failures and failures):
            return None, {}

        # Extraer pesos
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        stacked_weights = np.stack([weights for weights, _ in weights_results], axis=0)

        # Aplicar transformada de Fourier
        transformed_weights = np.fft.fft(stacked_weights, axis=0)
        aggregated_transformed = np.median(transformed_weights, axis=0)
        aggregated_weights = np.real(np.fft.ifft(aggregated_transformed, axis=0))
        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        # Agregar métricas personalizadas
        metrics_aggregated = self.aggregate_metrics(results)

        return parameters_aggregated, metrics_aggregated
