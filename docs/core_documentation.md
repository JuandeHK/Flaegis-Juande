<a id="main"></a>

# Module main [(ver código)](../main.py)

<a id="main.load_config"></a>

#### load\_config

```python
def load_config(config_path: str = "config/config.yaml") -> dict
```

Carga la configuración desde un archivo YAML y valida que contenga las claves necesarias.

**Arguments**:

- `config_path` _str_ - Ruta del archivo de configuración YAML. Por defecto, "config/config.yaml".
  

**Returns**:

- `dict` - Configuración cargada como un diccionario.
  

**Raises**:

- `FileNotFoundError` - Si el archivo de configuración no existe en la ruta proporcionada.
- `yaml.YAMLError` - Si ocurre un error al analizar el archivo YAML.
- `ValueError` - Si faltan claves necesarias en el archivo de configuración.

<a id="main.validate_arguments"></a>

#### validate\_arguments

```python
def validate_arguments(args: argparse.Namespace,
                       parser: argparse.ArgumentParser)
```

Valida los argumentos de línea de comandos para garantizar que sean consistentes y válidos.

**Arguments**:

- `args` _argparse.Namespace_ - Argumentos parseados de la línea de comandos.
- `parser` _argparse.ArgumentParser_ - Parser de argumentos para mostrar errores personalizados.
  

**Raises**:

- `ArgumentError` - Si alguno de los argumentos no cumple las restricciones.

<a id="main.parse_arguments"></a>

#### parse\_arguments

```python
def parse_arguments(config: dict)
```

Parsea los argumentos de línea de comandos y aplica validaciones.

**Arguments**:

- `config` _dict_ - Configuración cargada desde un archivo YAML, usada para valores predeterminados.
  

**Returns**:

- `argparse.Namespace` - Argumentos parseados y validados.

<a id="main.main"></a>

#### main

```python
def main()
```

Punto de entrada principal del programa.

Maneja la carga de la configuración, el parseo de los argumentos de línea de comandos y la inicialización del orquestador.
También captura y maneja errores comunes durante la ejecución.

**Raises**:

- `SystemExit` - Si ocurre un error fatal durante la ejecución.

<a id="src.orchestrator.fl\_orchestrator"></a>

# Module src.orchestrator.fl\_orchestrator [(ver código)](../src/orchestrator/fl_orchestrator.py)

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator"></a>

## FederatedLearningOrchestrator Objects

```python
class FederatedLearningOrchestrator()
```

Orquestador principal del proceso de aprendizaje federado.

Esta clase gestiona la configuración, inicialización y ejecución del 
aprendizaje federado, incluyendo la selección de clientes maliciosos 
y la sincronización entre clientes y servidor.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.__init__"></a>

#### FederatedLearningOrchestrator.\_\_init\_\_

```python
def __init__(n_clients: int = 50,
             n_rounds: int = 50,
             n_malicious: int = 0,
             base_dir: str = None,
             config: Optional[dict] = None)
```

Inicializa el orquestador del aprendizaje federado.

**Arguments**:

- `n_clients` _int_ - Número total de clientes en el sistema.
- `n_rounds` _int_ - Número de rondas de entrenamiento.
- `n_malicious` _int_ - Número de clientes maliciosos.
- `base_dir` _str_ - Directorio base para almacenar logs y archivos temporales.
- `config` _Optional[dict]_ - Configuración del sistema cargada desde un archivo YAML.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.setup_process_logging"></a>

#### FederatedLearningOrchestrator.setup\_process\_logging

```python
@staticmethod
def setup_process_logging(log_filename: str)
```

Sets up logging specifically for a child process.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.setup_gpu"></a>

#### FederatedLearningOrchestrator.setup\_gpu

```python
@staticmethod
def setup_gpu()
```

Configura el uso de la GPU para TensorFlow, permitiendo el crecimiento 
dinámico de la memoria en lugar de asignarla por completo.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.setup_logging"></a>

#### FederatedLearningOrchestrator.setup\_logging

```python
def setup_logging()
```

Configura el sistema de logging, incluyendo un archivo de log y salida en consola.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.select_malicious_clients"></a>

#### FederatedLearningOrchestrator.select\_malicious\_clients

```python
def select_malicious_clients() -> List[int]
```

Selecciona aleatoriamente los clientes maliciosos y guarda sus identificadores.

**Returns**:

- `List[int]` - Lista de identificadores de los clientes maliciosos.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.clean_previous_run"></a>

#### FederatedLearningOrchestrator.clean\_previous\_run

```python
def clean_previous_run()
```

Limpia los archivos y directorios de ejecuciones anteriores, como modelos y datos temporales.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.configure_model"></a>

#### FederatedLearningOrchestrator.configure\_model

```python
def configure_model()
```

Configura el modelo utilizando los valores definidos en el archivo YAML.

Los parámetros incluyen:
- Tipo de modelo.
- Tasa de aprendizaje.
- Tamaño de batch.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.initialize_clients"></a>

#### FederatedLearningOrchestrator.initialize\_clients

```python
def initialize_clients(
    malicious_clients: List[int]
) -> Tuple[multiprocessing.Process, List[multiprocessing.Process]]
```

Inicializa los procesos de los clientes y del servidor.

**Arguments**:

- `malicious_clients` _List[int]_ - Lista de identificadores de los clientes maliciosos.
  

**Returns**:

  Tuple[multiprocessing.Process, List[multiprocessing.Process]]:
  Proceso del servidor y lista de procesos de clientes.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.start_server"></a>

#### FederatedLearningOrchestrator.start\_server

```python
@staticmethod
def start_server(n_clients: int, n_rounds: int, config: dict,
                 server_ready_event: multiprocessing.Event)
```

Inicia el proceso del servidor con logging detallado.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.start_client"></a>

#### FederatedLearningOrchestrator.start\_client

```python
@staticmethod
def start_client(client_config_tuple)
```

Inicia un proceso de cliente FL.

**Arguments**:

- `client_config_tuple` - Tupla que contiene (client_config, global_config)

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.verify_environment"></a>

#### FederatedLearningOrchestrator.verify\_environment

```python
def verify_environment()
```

Verifica que el entorno esté configurado correctamente antes de la ejecución.

- Crea directorios requeridos si no existen.
- Verifica la configuración de GPU (opcional).
- Asegura que la configuración YAML sea válida.

<a id="src.orchestrator.fl_orchestrator.FederatedLearningOrchestrator.run"></a>

#### FederatedLearningOrchestrator.run

```python
def run() -> List[int]
```

Ejecuta el proceso de aprendizaje federado completo.

**Returns**:

- `List[int]` - Lista de clientes maliciosos seleccionados.

