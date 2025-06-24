<a id="src.custom\_strategies.base\_strategy"></a>

# Module src.custom\_strategies.base\_strategy [(ver código)](../src/custom_strategies/base_strategy.py)

Módulo base para estrategias personalizadas de agregación.

Define la interfaz base que todas las estrategias personalizadas deben implementar,
proporcionando funcionalidad adicional a las estrategias estándar de Flower.

<a id="src.custom_strategies.base_strategy.BaseStrategy"></a>

## BaseStrategy Objects

```python
class BaseStrategy(Strategy, ABC)
```

Clase base para estrategias personalizadas.

Esta clase proporciona métodos y lógica comunes para configurar estrategias
personalizadas en un entorno de aprendizaje federado. Actúa como una plantilla
para las estrategias concretas, que deben implementar métodos específicos.

**Attributes**:

- `fraction_fit` _float_ - Fracción de clientes usados durante el entrenamiento.
- `fraction_evaluate` _float_ - Fracción de clientes usados durante la evaluación.
- `min_fit_clients` _int_ - Número mínimo de clientes requeridos para entrenamiento.
- `min_evaluate_clients` _int_ - Número mínimo de clientes requeridos para evaluación.
- `min_available_clients` _int_ - Número mínimo total de clientes disponibles en el sistema.
- `evaluate_fn` _Optional[Callable]_ - Función opcional para evaluar los parámetros globales.
- `on_fit_config_fn` _Optional[Callable]_ - Función para configurar parámetros de entrenamiento.
- `on_evaluate_config_fn` _Optional[Callable]_ - Función para configurar parámetros de evaluación.
- `accept_failures` _bool_ - Indica si se aceptan fallos durante las rondas de entrenamiento/evaluación.
- `initial_parameters` _Optional[Parameters]_ - Parámetros iniciales del modelo global.
- `fit_metrics_aggregation_fn` _Optional[MetricsAggregationFn]_ - Función de agregación de métricas de ajuste.
- `evaluate_metrics_aggregation_fn` _Optional[MetricsAggregationFn]_ - Función de agregación de métricas de evaluación.

<a id="src.custom_strategies.base_strategy.BaseStrategy.num_fit_clients"></a>

#### BaseStrategy.num\_fit\_clients

```python
def num_fit_clients(num_available_clients: int) -> Tuple[int, int]
```

Devuelve el número de clientes necesarios para entrenamiento.

**Arguments**:

- `num_available_clients` _int_ - Número total de clientes actualmente disponibles.
  

**Returns**:

  Tuple[int, int]: Número de clientes seleccionados para entrenamiento y número mínimo requerido.

<a id="src.custom_strategies.base_strategy.BaseStrategy.num_evaluation_clients"></a>

#### BaseStrategy.num\_evaluation\_clients

```python
def num_evaluation_clients(num_available_clients: int) -> Tuple[int, int]
```

Devuelve el número de clientes necesarios para evaluación.

**Arguments**:

- `num_available_clients` _int_ - Número total de clientes actualmente disponibles.
  

**Returns**:

  Tuple[int, int]: Número de clientes seleccionados para evaluación y número mínimo requerido.

<a id="src.custom_strategies.base_strategy.BaseStrategy.initialize_parameters"></a>

#### BaseStrategy.initialize\_parameters

```python
def initialize_parameters(
        client_manager: ClientManager) -> Optional[Parameters]
```

Inicializa los parámetros del modelo global.

**Arguments**:

- `client_manager` _ClientManager_ - Gestor de clientes que contiene los clientes conectados.
  

**Returns**:

- `Optional[Parameters]` - Parámetros iniciales del modelo global.

<a id="src.custom_strategies.base_strategy.BaseStrategy.evaluate"></a>

#### BaseStrategy.evaluate

```python
def evaluate(
        server_round: int,
        parameters: Parameters) -> Optional[Tuple[float, dict[str, Scalar]]]
```

Evalúa los parámetros globales.

**Arguments**:

- `server_round` _int_ - Número actual de la ronda de entrenamiento.
- `parameters` _Parameters_ - Parámetros del modelo global a evaluar.
  

**Returns**:

  Optional[Tuple[float, dict[str, Scalar]]]: Pérdida y métricas de evaluación, o None si no se evalúa.

<a id="src.custom_strategies.base_strategy.BaseStrategy.configure_fit"></a>

#### BaseStrategy.configure\_fit

```python
def configure_fit(
        server_round: int, parameters: Parameters,
        client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]
```

Configura la siguiente ronda de ajuste.

**Arguments**:

- `server_round` _int_ - Número actual de la ronda de entrenamiento.
- `parameters` _Parameters_ - Parámetros actuales del modelo global.
- `client_manager` _ClientManager_ - Gestor de clientes conectados.
  

**Returns**:

  List[Tuple[ClientProxy, FitIns]]: Lista de pares (cliente, instrucciones de ajuste).

<a id="src.custom_strategies.base_strategy.BaseStrategy.configure_evaluate"></a>

#### BaseStrategy.configure\_evaluate

```python
def configure_evaluate(
        server_round: int, parameters: Parameters,
        client_manager: ClientManager
) -> List[Tuple[ClientProxy, EvaluateIns]]
```

Configura la siguiente ronda de evaluación.

**Arguments**:

- `server_round` _int_ - Número actual de la ronda de evaluación.
- `parameters` _Parameters_ - Parámetros actuales del modelo global.
- `client_manager` _ClientManager_ - Gestor de clientes conectados.
  

**Returns**:

  List[Tuple[ClientProxy, EvaluateIns]]: Lista de pares (cliente, instrucciones de evaluación).

<a id="src.custom_strategies.base_strategy.BaseStrategy.aggregate_fit"></a>

#### BaseStrategy.aggregate\_fit

```python
@abstractmethod
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], dict[str, Scalar]]
```

Agrega resultados de entrenamiento.

Este método debe ser implementado por estrategias específicas para definir
cómo se agregan los resultados de los clientes.

**Arguments**:

- `server_round` _int_ - Número actual de la ronda de entrenamiento.
- `results` _List[Tuple[ClientProxy, FitRes]]_ - Resultados exitosos del entrenamiento.
- `failures` _List[Union[Tuple[ClientProxy, FitRes], BaseException]]_ - Fallos durante el entrenamiento.
  

**Returns**:

  Tuple[Optional[Parameters], dict[str, Scalar]]: Parámetros agregados y métricas adicionales.

<a id="src.custom\_strategies.weighted\_median\_strategy"></a>

# Module src.custom\_strategies.weighted\_median\_strategy [(ver código)](../src/custom_strategies/weighted_median_strategy.py)

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy"></a>

## WeightedMedianStrategy Objects

```python
class WeightedMedianStrategy(Strategy)
```

Estrategia de agregación usando mediana ponderada robusta contra ataques.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.initialize_parameters"></a>

#### WeightedMedianStrategy.initialize\_parameters

```python
def initialize_parameters(client_manager) -> Optional[Parameters]
```

Inicializa los parámetros del modelo global.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.configure_fit"></a>

#### WeightedMedianStrategy.configure\_fit

```python
def configure_fit(
        server_round: int, parameters: Parameters,
        client_manager) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]
```

Configura los clientes para entrenamiento.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.configure_evaluate"></a>

#### WeightedMedianStrategy.configure\_evaluate

```python
def configure_evaluate(
        server_round: int, parameters: Parameters,
        client_manager) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]
```

Configura los clientes para evaluación.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.aggregate_fit"></a>

#### WeightedMedianStrategy.aggregate\_fit

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]
```

Agrega los parámetros usando mediana ponderada.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.aggregate_evaluate"></a>

#### WeightedMedianStrategy.aggregate\_evaluate

```python
def aggregate_evaluate(
    server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
) -> Tuple[Optional[float], Dict[str, Scalar]]
```

Agrega las métricas de evaluación.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.evaluate"></a>

#### WeightedMedianStrategy.evaluate

```python
def evaluate(
        server_round: int,
        parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]
```

Evalúa el modelo global en el servidor (opcional).

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.num_fit_clients"></a>

#### WeightedMedianStrategy.num\_fit\_clients

```python
def num_fit_clients(num_available_clients: int) -> Tuple[int, int]
```

Calcula número de clientes para entrenamiento.

<a id="src.custom_strategies.weighted_median_strategy.WeightedMedianStrategy.num_evaluation_clients"></a>

#### WeightedMedianStrategy.num\_evaluation\_clients

```python
def num_evaluation_clients(num_available_clients: int) -> Tuple[int, int]
```

Calcula número de clientes para evaluación.

<a id="src.custom\_strategies.fourier\_strategy"></a>

# Module src.custom\_strategies.fourier\_strategy [(ver código)](../src/custom_strategies/fourier_strategy.py)

Estrategia personalizada basada en transformada de Fourier.

Esta estrategia utiliza análisis de Fourier para la agregación de modelos,
permitiendo identificar y filtrar componentes anómalos en el dominio de la frecuencia.

<a id="src.custom_strategies.fourier_strategy.FourierStrategy"></a>

## FourierStrategy Objects

```python
class FourierStrategy(BaseStrategy)
```

Estrategia de agregación basada en transformada de Fourier.

<a id="src.custom_strategies.fourier_strategy.FourierStrategy.aggregate_fit"></a>

#### FourierStrategy.aggregate\_fit

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], dict[str, Scalar]]
```

Agrega resultados usando transformada de Fourier.

<a id="src.custom\_strategies.trimmed\_mean\_strategy"></a>

# Module src.custom\_strategies.trimmed\_mean\_strategy [(ver código)](../src/custom_strategies/trimmed_mean_strategy.py)

Estrategia personalizada basada en media recortada.

Esta estrategia elimina valores extremos de los parámetros de los clientes
antes de calcular la media, proporcionando mayor robustez contra ataques o anomalías.

<a id="src.custom_strategies.trimmed_mean_strategy.TrimmedMeanStrategy"></a>

## TrimmedMeanStrategy Objects

```python
class TrimmedMeanStrategy(BaseStrategy)
```

Estrategia de agregación basada en media recortada.

<a id="src.custom_strategies.trimmed_mean_strategy.TrimmedMeanStrategy.__init__"></a>

#### TrimmedMeanStrategy.\_\_init\_\_

```python
def __init__(trim_ratio: float = 0.1, **kwargs)
```

Inicializa la estrategia de media recortada.

**Arguments**:

- `trim_ratio` _float_ - Proporción de valores a recortar de cada extremo.
- `kwargs` - Argumentos adicionales para la estrategia base.

<a id="src.custom_strategies.trimmed_mean_strategy.TrimmedMeanStrategy.aggregate_fit"></a>

#### TrimmedMeanStrategy.aggregate\_fit

```python
def aggregate_fit(
    server_round: int, results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
) -> Tuple[Optional[Parameters], dict[str, Scalar]]
```

Agrega resultados usando media recortada.

