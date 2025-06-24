<a id="src.detectors.base\_clustering\_detector"></a>

# Module src.detectors.base\_clustering\_detector [(ver código)](../src/detectors/base_clustering_detector.py)

<a id="src.detectors.base_clustering_detector.BaseClusteringDetector"></a>

## BaseClusteringDetector Objects

```python
class BaseClusteringDetector(ClusteringStrategy)
```

Clase base para detectores basados en clustering que implementa la lógica común.

<a id="src.detectors.clustering\_strategy"></a>

# Module src.detectors.clustering\_strategy [(ver código)](../src/detectors/clustering_strategy.py)

<a id="src.detectors.clustering_strategy.ClusteringStrategy"></a>

## ClusteringStrategy Objects

```python
class ClusteringStrategy(ABC)
```

Interfaz para estrategias de detección de clustering.

<a id="src.detectors.clustering_strategy.ClusteringStrategy.detect"></a>

#### ClusteringStrategy.detect

```python
@abstractmethod
def detect(similarity_matrix: np.ndarray, clients: List[int]) -> List[int]
```

Método abstracto para detectar clientes maliciosos usando clustering.

<a id="src.detectors.clustering\_context"></a>

# Module src.detectors.clustering\_context [(ver código)](../src/detectors/clustering_context.py)

<a id="src.detectors.clustering_context.ClusteringContext"></a>

## ClusteringContext Objects

```python
class ClusteringContext()
```

Contexto que usa una estrategia de clustering para detectar clientes maliciosos.

<a id="src.detectors.clustering_context.ClusteringContext.detect_malicious_clients"></a>

#### ClusteringContext.detect\_malicious\_clients

```python
def detect_malicious_clients(similarity_matrix: np.ndarray,
                             clients: List[int]) -> List[int]
```

Usa la estrategia actual para detectar clientes maliciosos.

<a id="src.detectors.spectral\_detector"></a>

# Module src.detectors.spectral\_detector [(ver código)](../src/detectors/spectral_detector.py)

<a id="src.detectors.spectral_detector.SpectralDetector"></a>

## SpectralDetector Objects

```python
class SpectralDetector(BaseClusteringDetector)
```

Detector de clientes maliciosos basado en clustering espectral.

Este detector utiliza un enfoque de clustering espectral para identificar 
clientes maliciosos en un entorno de aprendizaje federado. La técnica de 
clustering espectral utiliza una matriz de similitud precomputada para separar 
los clientes en dos grupos, clasificándolos como "buenos" o "malos" según 
el tamaño de los clusters.

Hereda:
    BaseClusteringDetector: Clase base que define la interfaz común para 
    detectores basados en clustering.

<a id="src.detectors.spectral_detector.SpectralDetector.detect"></a>

#### SpectralDetector.detect

```python
def detect(similarity_matrix: np.ndarray, clients: List[int]) -> List[int]
```

Detecta clientes maliciosos utilizando clustering espectral.

**Arguments**:

- `similarity_matrix` _np.ndarray_ - Matriz de similitud precomputada entre clientes.
- `clients` _List[int]_ - Lista de identificadores de los clientes.
  

**Returns**:

- `List[int]` - Lista de identificadores de clientes clasificados como maliciosos.

<a id="src.detectors.hierarchical\_detector"></a>

# Module src.detectors.hierarchical\_detector [(ver código)](../src/detectors/hierarchical_detector.py)

<a id="src.detectors.hierarchical_detector.HierarchicalDetector"></a>

## HierarchicalDetector Objects

```python
class HierarchicalDetector(BaseClusteringDetector)
```

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

<a id="src.detectors.hierarchical_detector.HierarchicalDetector.__init__"></a>

#### HierarchicalDetector.\_\_init\_\_

```python
def __init__(threshold: float = 0.6)
```

Inicializa el detector jerárquico con un umbral para cortar el dendrograma.

**Arguments**:

- `threshold` _float_ - Umbral de distancia para determinar los clusters.
  Los clientes se agrupan en clusters según este criterio.

<a id="src.detectors.hierarchical_detector.HierarchicalDetector.detect"></a>

#### HierarchicalDetector.detect

```python
def detect(similarity_matrix: np.ndarray, clients: List[int]) -> List[int]
```

Detecta clientes maliciosos utilizando clustering jerárquico y K-Means.

**Arguments**:

- `similarity_matrix` _np.ndarray_ - Matriz de similitud precomputada entre clientes.
- `clients` _List[int]_ - Lista de identificadores de los clientes.
  

**Returns**:

- `List[int]` - Lista de identificadores de clientes clasificados como maliciosos.

<a id="src.detectors.sign\_guard\_detector"></a>

# Module src.detectors.sign\_guard\_detector [(ver código)](../src/detectors/sign_guard_detector.py)

<a id="src.detectors.sign_guard_detector.SignGuardDetector"></a>

## SignGuardDetector Objects

```python
class SignGuardDetector()
```

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

<a id="src.detectors.sign_guard_detector.SignGuardDetector.__init__"></a>

#### SignGuardDetector.\_\_init\_\_

```python
def __init__()
```

Inicializa el detector SignGuard.

Crea una instancia de `MeanShift` para realizar agrupaciones en las estadísticas
de signos de los gradientes.

<a id="src.detectors.sign_guard_detector.SignGuardDetector.detect"></a>

#### SignGuardDetector.detect

```python
def detect(gradients: List[np.ndarray], party_numbers: List[int]) -> List[int]
```

Detecta clientes maliciosos usando el algoritmo SignGuard.

**Arguments**:

- `gradients` - Lista de gradientes de cada cliente
- `party_numbers` - Lista de identificadores de clientes
  

**Returns**:

- `List[int]` - Lista de clientes identificados como maliciosos

