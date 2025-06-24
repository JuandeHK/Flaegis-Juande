<a id="server.server\_manager"></a>

# Module server.server\_manager [(ver código)](../src/server/server_manager.py)

<a id="server.server_manager.fit_config_simple"></a>

#### fit\_config\_simple

```python
def fit_config_simple(server_round: int) -> Dict[str, Any]
```

Función de configuración simple que no es un método de instancia.
Devuelve siempre la misma configuración para la prueba.

<a id="server.server_manager.FederatedServer"></a>

## FederatedServer Objects

```python
class FederatedServer()
```

<a id="server.server_manager.FederatedServer.__init__"></a>

#### FederatedServer.\_\_init\_\_

```python
def __init__(n_clients: int, n_rounds: int, config: Dict[str, Any])
```

Inicializa el servidor federado con los parámetros proporcionados.

**Arguments**:

- `n_clients` _int_ - Número total de clientes conectados al servidor.
- `n_rounds` _int_ - Número total de rondas de entrenamiento.
- `config` _Dict[str, Any]_ - Configuración del servidor, incluyendo parámetros
  de entrenamiento, evaluación y estrategias.

<a id="server.server_manager.FederatedServer.start"></a>

#### FederatedServer.start

```python
def start()
```

Inicia el servidor de Flower con la estrategia definida en el config.yaml.

