<a id="src.server.strategy\_configurators.base\_strategy\_configurator"></a>

# Module src.server.strategy\_configurators.base\_strategy\_configurator [(ver código)](../src/server/strategy_configurators/base_strategy_configurator.py)

Módulo que define la interfaz base para los configuradores de estrategias.

Este módulo contiene la clase abstracta que todos los configuradores de estrategias
deben implementar, asegurando una interfaz consistente para la configuración de
estrategias en el sistema de aprendizaje federado.

<a id="src.server.strategy_configurators.base_strategy_configurator.StrategyConfigurator"></a>

## StrategyConfigurator Objects

```python
class StrategyConfigurator(ABC)
```

Interfaz base para configuradores de estrategias.

Esta clase abstracta define el contrato que todos los configuradores de estrategias
deben seguir, proporcionando un método común para crear estrategias específicas
de Flower.

<a id="src.server.strategy_configurators.base_strategy_configurator.StrategyConfigurator.create_strategy"></a>

#### StrategyConfigurator.create\_strategy

```python
@abstractmethod
def create_strategy(
        server_config: Dict[str, Any]) -> fl.server.strategy.Strategy
```

Crea una instancia de estrategia de Flower basada en la configuración proporcionada.

<a id="src.server.strategy\_configurators.fedavg\_strategy\_configuration"></a>

# Module src.server.strategy\_configurators.fedavg\_strategy\_configuration [(ver código)](../src/server/strategy_configurators/fedavg_strategy_configuration.py)

<a id="src.server.strategy_configurators.fedavg_strategy_configuration.aggregate_weighted_metrics"></a>

#### aggregate\_weighted\_metrics

```python
def aggregate_weighted_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics
```

Calcula la media ponderada para cada métrica que envían los clientes.
Esta función SÍ sabe cómo manejar diccionarios como {'loss': 0.1, 'mae': 0.2}.

<a id="src.server.strategy_configurators.fedavg_strategy_configuration.FedAvgStrategyConfigurator"></a>

## FedAvgStrategyConfigurator Objects

```python
class FedAvgStrategyConfigurator(StrategyConfigurator)
```

Configurador para FedAvg que usa nuestra función de agregación personalizada.

<a id="src.server.strategy_configurators.fedavg_strategy_configuration.FedAvgStrategyConfigurator.create_strategy"></a>

#### FedAvgStrategyConfigurator.create\_strategy

```python
def create_strategy(
        server_config: Dict[str, Any]) -> fl.server.strategy.Strategy
```

Crea una instancia de FedAvg, pasándole nuestra función de agregación.

<a id="src.server.strategy\_configurators.weighted\_median\_strategy\_configuration"></a>

# Module src.server.strategy\_configurators.weighted\_median\_strategy\_configuration [(ver código)](../src/server/strategy_configurators/weighted_median_strategy_configuration.py)

Implementación del configurador de la estrategia de mediana ponderada.

Este módulo implementa un configurador para la estrategia de agregación robusta basada en la
mediana ponderada, que es más resistente a valores atípicos y ataques
que la media simple.

<a id="src.server.strategy_configurators.weighted_median_strategy_configuration.WeightedMedianStrategyConfigurator"></a>

## WeightedMedianStrategyConfigurator Objects

```python
class WeightedMedianStrategyConfigurator(StrategyConfigurator)
```

Configurador de la estrategia de mediana ponderada.

Configura una estrategia de agregación que utiliza la mediana ponderada
para combinar los modelos de los clientes, proporcionando mayor robustez
contra clientes maliciosos.

<a id="src.server.strategy_configurators.weighted_median_strategy_configuration.WeightedMedianStrategyConfigurator.create_strategy"></a>

#### WeightedMedianStrategyConfigurator.create\_strategy

```python
def create_strategy(
        server_config: Dict[str, Any]) -> fl.server.strategy.Strategy
```

Crea una instancia de la estrategia de mediana ponderada.

<a id="src.server.strategy\_configurators.fourier\_strategy\_configuration"></a>

# Module src.server.strategy\_configurators.fourier\_strategy\_configuration [(ver código)](../src/server/strategy_configurators/fourier_strategy_configuration.py)

Implementación del configurador de la estrategia basada en transformada de Fourier.

Este módulo implementa un configurador para la estrategia de agregación que utiliza 
análisis de Fourier para combinar los modelos de los clientes, lo que puede ayudar 
a identificar y filtrar componentes anómalos en el espacio de frecuencias.

<a id="src.server.strategy_configurators.fourier_strategy_configuration.FourierStrategyConfigurator"></a>

## FourierStrategyConfigurator Objects

```python
class FourierStrategyConfigurator(StrategyConfigurator)
```

Configurador de la estrategia basada en transformada de Fourier.

Configura una estrategia que utiliza análisis de Fourier para combinar
los modelos de los clientes, permitiendo un filtrado más sofisticado
de las contribuciones de los clientes en el dominio de la frecuencia.

<a id="src.server.strategy_configurators.fourier_strategy_configuration.FourierStrategyConfigurator.create_strategy"></a>

#### FourierStrategyConfigurator.create\_strategy

```python
def create_strategy(
        server_config: Dict[str, Any]) -> fl.server.strategy.Strategy
```

Crea una instancia de la estrategia basada en Fourier.

**Arguments**:

- `server_config` _Dict[str, Any]_ - Configuración del servidor incluyendo:
  - n_clients: Número de clientes
  - config: Configuración general con parámetros específicos
  - fit_config: Función de configuración para entrenamiento
  - evaluate_config: Función de configuración para evaluación
  

**Returns**:

- `fl.server.strategy.Strategy` - Estrategia Fourier configurada

<a id="src.server.strategy\_configurators.trimmed\_mean\_strategy\_configuration"></a>

# Module src.server.strategy\_configurators.trimmed\_mean\_strategy\_configuration [(ver código)](../src/server/strategy_configurators/trimmed_mean_strategy_configuration.py)

Implementación del configurador de la estrategia de media recortada.

Este módulo configura una estrategia basada en la media recortada, que elimina
valores extremos antes de la agregación para mayor robustez.

<a id="src.server.strategy_configurators.trimmed_mean_strategy_configuration.TrimmedMeanStrategyConfigurator"></a>

## TrimmedMeanStrategyConfigurator Objects

```python
class TrimmedMeanStrategyConfigurator(StrategyConfigurator)
```

Configurador de la estrategia de media recortada.

Configura una estrategia que elimina un porcentaje de valores extremos
antes de calcular la media de los parámetros de los clientes.

<a id="src.server.strategy_configurators.trimmed_mean_strategy_configuration.TrimmedMeanStrategyConfigurator.__init__"></a>

#### TrimmedMeanStrategyConfigurator.\_\_init\_\_

```python
def __init__(trim_ratio: float = 0.1)
```

Inicializa el configurador de la estrategia de media recortada.

**Arguments**:

- `trim_ratio` _float_ - Proporción de valores a recortar de cada extremo.
  Por defecto es 0.1 (10% de cada lado).

<a id="src.server.strategy_configurators.trimmed_mean_strategy_configuration.TrimmedMeanStrategyConfigurator.create_strategy"></a>

#### TrimmedMeanStrategyConfigurator.create\_strategy

```python
def create_strategy(
        server_config: Dict[str, Any]) -> fl.server.strategy.Strategy
```

Crea una instancia de la estrategia de media recortada.

**Arguments**:

- `server_config` _Dict[str, Any]_ - Configuración del servidor, incluyendo:
  - n_clients: Número de clientes.
  - fit_config: Configuración para entrenamiento.
  - evaluate_config: Configuración para evaluación.
  

**Returns**:

- `fl.server.strategy.Strategy` - Estrategia configurada para usar media recortada.

