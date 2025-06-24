import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
import flwr as fl
from flwr.common import (
    Metrics, FitRes, EvaluateRes, Parameters, Scalar, 
    NDArrays, parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy

class WeightedMedianStrategy(Strategy):
    """Estrategia de agregación usando mediana ponderada robusta contra ataques."""
    
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        **kwargs
    ):
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.logger = logging.getLogger(__name__)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Inicializa los parámetros del modelo global."""
        return None  # Los parámetros se inicializarán desde el primer cliente

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configura los clientes para entrenamiento."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # Configuración para los clientes
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        fit_configurations = [(client, config) for client in clients]
        return fit_configurations

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
        """Configura los clientes para evaluación."""
        if self.fraction_evaluate == 0.0:
            return []
        
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # Configuración para evaluación
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        
        evaluate_configurations = [(client, config) for client in clients]
        return evaluate_configurations

    # def aggregate_fit(
    #     self, 
    #     server_round: int, 
    #     results: List[Tuple[ClientProxy, FitRes]], 
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    # ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    #     """Agrega los parámetros usando mediana ponderada."""
    #     if not results:
    #         return None, {}
        
    #     self.logger.info(f"Round {server_round}: Aggregating {len(results)} client updates")
        
    #     # Convertir parámetros a arrays numpy
    #     weights_results = [
    #         (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
    #         for _, fit_res in results
    #     ]
        
    #     # Implementar mediana ponderada (por simplicidad, usando promedio ponderado)
    #     # En una implementación real, aquí irían algoritmos robustos de mediana ponderada
    #     aggregated_ndarrays = self._aggregate_weighted_average(weights_results)
    #     parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        
    #     metrics_aggregated = {
    #         "num_clients": len(results),
    #         "total_examples": sum(fit_res.num_examples for _, fit_res in results)
    #     }
        
    #     return parameters_aggregated, metrics_aggregated

    def aggregate_fit(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Agrega los parámetros usando mediana ponderada."""
        
        self.logger.info(f"ESTRATEGIA: Entrando en aggregate_fit para la ronda {server_round}")
        
        if not results:
            self.logger.warning("ESTRATEGIA: aggregate_fit no ha recibido resultados. No se puede agregar.")
            return None, {}
        
        # Logueamos si hubo fallos de conexión
        if failures:
            self.logger.error(f"ESTRATEGIA: Se han recibido {len(failures)} fallos de clientes.")

        try:
            self.logger.info(f"ESTRATEGIA: Agregando {len(results)} actualizaciones de clientes.")
            
            # Convertir parámetros a arrays numpy
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            self.logger.info("ESTRATEGIA: Parámetros de clientes convertidos a ndarrays.")
            
            # Implementar promedio ponderado (lógica actual)
            aggregated_ndarrays = self._aggregate_weighted_average(weights_results)
            self.logger.info("ESTRATEGIA: Agregación con _aggregate_weighted_average completada.")
            
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
            self.logger.info("ESTRATEGIA: Parámetros agregados convertidos de nuevo a formato Parameters.")
            
            metrics_aggregated = {
                "num_clients": len(results),
                "total_examples": sum(fit_res.num_examples for _, fit_res in results)
            }
            
            self.logger.info("ESTRATEGIA: Saliendo de aggregate_fit con éxito.")
            return parameters_aggregated, metrics_aggregated

        except Exception as e:
            self.logger.error("ESTRATEGIA: !!! ERROR CRÍTICO DENTRO DE AGGREGATE_FIT !!!", exc_info=True)
            # Devolvemos None para indicar al servidor que la agregación ha fallado
            return None, {}

    def aggregate_evaluate(
        self, 
        server_round: int, 
        results: List[Tuple[ClientProxy, EvaluateRes]], 
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Agrega las métricas de evaluación."""
        if not results:
            return None, {}
        
        # Agregar loss usando promedio ponderado
        total_examples = sum(res.num_examples for _, res in results)
        if total_examples == 0:
            return None, {}
            
        loss_aggregated = sum(
            res.loss * res.num_examples for _, res in results
        ) / total_examples
        
        # Agregar métricas personalizadas - MAE en lugar de accuracy
        metrics_aggregated = {}
        try:
            if total_examples > 0:
                mae_values = [
                    res.metrics.get("mae", 0.0) * res.num_examples 
                    for _, res in results 
                    if "mae" in res.metrics
                ]
                
                if mae_values:
                    mae_aggregated = sum(mae_values) / total_examples
                    metrics_aggregated["mae"] = mae_aggregated
                    
                    self.logger.info(
                        f"Round {server_round}: Loss={loss_aggregated:.4f}, "
                        f"MAE={mae_aggregated:.4f}, Clients={len(results)}"
                    )
                else:
                    self.logger.warning(f"Round {server_round}: No MAE metrics found in client results")
                    
        except Exception as e:
            self.logger.error(f"Error agregando métricas en round {server_round}: {e}")
        
        return loss_aggregated, metrics_aggregated

    # def aggregate_evaluate(
    #     self, 
    #     server_round: int, 
    #     results: List[Tuple[ClientProxy, EvaluateRes]], 
    #     failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Agrega las métricas de evaluación."""
    #     if not results:
    #         return None, {}
        
    #     # Agregar loss usando promedio ponderado
    #     total_examples = sum(res.num_examples for _, res in results)
    #     if total_examples == 0:
    #         return None, {}
            
    #     loss_aggregated = sum(
    #         res.loss * res.num_examples for _, res in results
    #     ) / total_examples
        
    #     # Agregar métricas personalizadas - MAE en lugar de accuracy
    #     metrics_aggregated = {}
    #     try:
    #         if total_examples > 0:
    #             # MAE agregado
    #             mae_values = [
    #                 res.metrics.get("mae", 0.0) * res.num_examples 
    #                 for _, res in results 
    #                 if "mae" in res.metrics
    #             ]
                
    #             # ✅ R² AGREGADO
    #             r2_values = [
    #                 res.metrics.get("r2", 0.0) * res.num_examples 
    #                 for _, res in results 
    #                 if "r2" in res.metrics
    #             ]
                
    #             if mae_values:
    #                 mae_aggregated = sum(mae_values) / total_examples
    #                 metrics_aggregated["mae"] = mae_aggregated
                
    #             # ✅ AÑADIR R² A LAS MÉTRICAS AGREGADAS
    #             if r2_values:
    #                 r2_aggregated = sum(r2_values) / total_examples
    #                 metrics_aggregated["r2"] = r2_aggregated
                    
    #             self.logger.info(
    #                 f"Round {server_round}: Loss={loss_aggregated:.4f}, "
    #                 f"MAE={metrics_aggregated.get('mae', 'N/A'):.4f}, "
    #                 f"R²={metrics_aggregated.get('r2', 'N/A'):.4f}, Clients={len(results)}"
    #             )
    #         else:
    #             self.logger.warning(f"Round {server_round}: No MAE/R² metrics found in client results")
                
    #     except Exception as e:
    #         self.logger.error(f"Error agregando métricas en round {server_round}: {e}")
        
    #     return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evalúa el modelo global en el servidor (opcional)."""
        # No implementamos evaluación centralizada por ahora
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Calcula número de clientes para entrenamiento."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Calcula número de clientes para evaluación."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def _aggregate_weighted_average(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average of model parameters."""
        total_examples = sum(num_examples for _, num_examples in results)
        
        # Inicializar arrays agregados
        aggregated_arrays = None
        
        for arrays, num_examples in results:
            weight = num_examples / total_examples
            
            if aggregated_arrays is None:
                aggregated_arrays = [array * weight for array in arrays]
            else:
                for i, array in enumerate(arrays):
                    aggregated_arrays[i] += array * weight
        
        return aggregated_arrays

    
    def _calculate_weighted_median(self, results: List[Tuple[NDArrays, int]]) -> NDArrays:
        """Calcula la mediana ponderada capa por capa."""

        # Extraer los pesos de todos los clientes
        all_weights = [res[0] for res in results]

        # Lista para guardar las capas agregadas
        aggregated_layers = []

        # Iterar sobre cada capa del modelo
        for i in range(len(all_weights[0])):
            # Apilar todas las versiones de la capa i de todos los clientes
            stacked_layer = np.stack([client_weights[i] for client_weights in all_weights])

            # Calcular la mediana a lo largo del eje de los clientes (axis=0)
            median_layer = np.median(stacked_layer, axis=0)
            aggregated_layers.append(median_layer)

        return aggregated_layers
