"""
Módulo base para estrategias personalizadas de agregación.

Define la interfaz base que todas las estrategias personalizadas deben implementar,
proporcionando funcionalidad adicional a las estrategias estándar de Flower.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class BaseStrategy(Strategy, ABC):
    """
    Clase base para estrategias personalizadas.

    Esta clase proporciona métodos y lógica comunes para configurar estrategias
    personalizadas en un entorno de aprendizaje federado. Actúa como una plantilla
    para las estrategias concretas, que deben implementar métodos específicos.

    Attributes:
        fraction_fit (float): Fracción de clientes usados durante el entrenamiento.
        fraction_evaluate (float): Fracción de clientes usados durante la evaluación.
        min_fit_clients (int): Número mínimo de clientes requeridos para entrenamiento.
        min_evaluate_clients (int): Número mínimo de clientes requeridos para evaluación.
        min_available_clients (int): Número mínimo total de clientes disponibles en el sistema.
        evaluate_fn (Optional[Callable]): Función opcional para evaluar los parámetros globales.
        on_fit_config_fn (Optional[Callable]): Función para configurar parámetros de entrenamiento.
        on_evaluate_config_fn (Optional[Callable]): Función para configurar parámetros de evaluación.
        accept_failures (bool): Indica si se aceptan fallos durante las rondas de entrenamiento/evaluación.
        initial_parameters (Optional[Parameters]): Parámetros iniciales del modelo global.
        fit_metrics_aggregation_fn (Optional[MetricsAggregationFn]): Función de agregación de métricas de ajuste.
        evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn]): Función de agregación de métricas de evaluación.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, dict[str, Scalar]], Optional[Tuple[float, dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Devuelve el número de clientes necesarios para entrenamiento.

        Args:
            num_available_clients (int): Número total de clientes actualmente disponibles.

        Returns:
            Tuple[int, int]: Número de clientes seleccionados para entrenamiento y número mínimo requerido.
        """
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Devuelve el número de clientes necesarios para evaluación.

        Args:
            num_available_clients (int): Número total de clientes actualmente disponibles.

        Returns:
            Tuple[int, int]: Número de clientes seleccionados para evaluación y número mínimo requerido.
        """
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """
        Inicializa los parámetros del modelo global.

        Args:
            client_manager (ClientManager): Gestor de clientes que contiene los clientes conectados.

        Returns:
            Optional[Parameters]: Parámetros iniciales del modelo global.
        """
        return self.initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, dict[str, Scalar]]]:
        """
        Evalúa los parámetros globales.

        Args:
            server_round (int): Número actual de la ronda de entrenamiento.
            parameters (Parameters): Parámetros del modelo global a evaluar.

        Returns:
            Optional[Tuple[float, dict[str, Scalar]]]: Pérdida y métricas de evaluación, o None si no se evalúa.
        """
        if self.evaluate_fn is None:
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura la siguiente ronda de ajuste.

        Args:
            server_round (int): Número actual de la ronda de entrenamiento.
            parameters (Parameters): Parámetros actuales del modelo global.
            client_manager (ClientManager): Gestor de clientes conectados.

        Returns:
            List[Tuple[ClientProxy, FitIns]]: Lista de pares (cliente, instrucciones de ajuste).
        """
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configura la siguiente ronda de evaluación.

        Args:
            server_round (int): Número actual de la ronda de evaluación.
            parameters (Parameters): Parámetros actuales del modelo global.
            client_manager (ClientManager): Gestor de clientes conectados.

        Returns:
            List[Tuple[ClientProxy, EvaluateIns]]: Lista de pares (cliente, instrucciones de evaluación).
        """
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        return [(client, evaluate_ins) for client in clients]

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Agrega resultados de entrenamiento.

        Este método debe ser implementado por estrategias específicas para definir
        cómo se agregan los resultados de los clientes.

        Args:
            server_round (int): Número actual de la ronda de entrenamiento.
            results (List[Tuple[ClientProxy, FitRes]]): Resultados exitosos del entrenamiento.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): Fallos durante el entrenamiento.

        Returns:
            Tuple[Optional[Parameters], dict[str, Scalar]]: Parámetros agregados y métricas adicionales.
        """
        pass
