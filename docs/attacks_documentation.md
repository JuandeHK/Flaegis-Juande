<a id="src.attacks.base\_attack"></a>

# Module src.attacks.base\_attack [(ver código)](../src/attacks/base_attack.py)

<a id="src.attacks.base_attack.BaseAttack"></a>

## BaseAttack Objects

```python
class BaseAttack(ABC)
```

Interfaz base para estrategias de ataque.

Define la interfaz común que deben implementar todos los ataques,
ya sean de envenenamiento de datos o de modelo.

<a id="src.attacks.base_attack.BaseAttack.attack"></a>

#### BaseAttack.attack

```python
@abstractmethod
def attack(data: Any) -> Any
```

Ejecuta el ataque sobre los datos o modelo proporcionado.

**Arguments**:

- `data` - Datos o modelo a atacar
  

**Returns**:

  Datos o modelo modificado tras el ataque

<a id="src.attacks.attack\_factory"></a>

# Module src.attacks.attack\_factory [(ver código)](../src/attacks/attack_factory.py)

<a id="src.attacks.attack_factory.AttackFactory"></a>

## AttackFactory Objects

```python
class AttackFactory()
```

Factory para crear instancias de ataques según la configuración.

<a id="src.attacks.attack_factory.AttackFactory.create_attack"></a>

#### AttackFactory.create\_attack

```python
@staticmethod
def create_attack(config: Dict[str, Any]) -> BaseAttack
```

Crea una instancia del ataque especificado.

**Arguments**:

- `config` - Configuración del ataque del archivo YAML
  

**Returns**:

- `BaseAttack` - Instancia del ataque configurado
  

**Raises**:

- `ValueError` - Si el tipo de ataque o estrategia no es válido

<a id="src.attacks.model\_poisoning.min\_max\_attack"></a>

# Module src.attacks.model\_poisoning.min\_max\_attack [(ver código)](../src/attacks/model_poisoning/min_max_attack.py)

<a id="src.attacks.model_poisoning.min_max_attack.MinMaxAttack"></a>

## MinMaxAttack Objects

```python
class MinMaxAttack(BaseAttack)
```

Implementa el ataque Min-Max sobre los parámetros del modelo.

Este ataque busca maximizar la distancia entre los parámetros benignos
y maliciosos mientras mantiene las actualizaciones maliciosas dentro
de un rango aceptable para evitar la detección.

<a id="src.attacks.model_poisoning.min_max_attack.MinMaxAttack.attack"></a>

#### MinMaxAttack.attack

```python
def attack(data: tuple) -> np.ndarray
```

Ejecuta el ataque Min-Max.

**Arguments**:

- `data` _tuple_ - Tupla conteniendo (all_updates, model_re)
  

**Returns**:

- `np.ndarray` - Actualización maliciosa generada

<a id="src.attacks.model\_poisoning.min\_sum\_attack"></a>

# Module src.attacks.model\_poisoning.min\_sum\_attack [(ver código)](../src/attacks/model_poisoning/min_sum_attack.py)

<a id="src.attacks.model_poisoning.min_sum_attack.MinSumAttack"></a>

## MinSumAttack Objects

```python
class MinSumAttack(BaseAttack)
```

Implementa el ataque Min-Sum sobre los parámetros del modelo.

Este ataque busca minimizar la suma de los parámetros del modelo global
para degradar su rendimiento mientras evita la detección.

<a id="src.attacks.model_poisoning.min_sum_attack.MinSumAttack.__init__"></a>

#### MinSumAttack.\_\_init\_\_

```python
def __init__(scale_factor: float = 1.0, clip_threshold: float = 100.0)
```

Inicializa el ataque Min-Sum.

**Arguments**:

- `scale_factor` _float_ - Factor de escala para las actualizaciones maliciosas
- `clip_threshold` _float_ - Umbral para recortar valores extremos

<a id="src.attacks.model_poisoning.min_sum_attack.MinSumAttack.attack"></a>

#### MinSumAttack.attack

```python
def attack(data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray
```

Ejecuta el ataque Min-Sum sobre los parámetros del modelo.

**Arguments**:

- `data` - Tupla (actualizaciones_benignas, modelo_referencia)
  - actualizaciones_benignas: Actualizaciones de clientes benignos
  - modelo_referencia: Parámetros del modelo de referencia
  

**Returns**:

- `np.ndarray` - Actualización maliciosa generada

<a id="src.attacks.model\_poisoning.lie\_attack"></a>

# Module src.attacks.model\_poisoning.lie\_attack [(ver código)](../src/attacks/model_poisoning/lie_attack.py)

<a id="src.attacks.model_poisoning.lie_attack.LieAttack"></a>

## LieAttack Objects

```python
class LieAttack(BaseAttack)
```

Implementa el ataque de mentira (Lie Attack)(Little Is Enough) sobre los parámetros del modelo.

Este ataque envía actualizaciones falsas pero plausibles al servidor,
escalando las actualizaciones benignas y añadiendo ruido controlado.

<a id="src.attacks.model_poisoning.lie_attack.LieAttack.__init__"></a>

#### LieAttack.\_\_init\_\_

```python
def __init__(scale_factor: float = 1.5,
             noise_range: float = 0.1,
             clip_threshold: float = 100.0)
```

Inicializa el ataque de mentira.

**Arguments**:

- `scale_factor` - Factor de escala para las actualizaciones benignas
- `noise_range` - Rango del ruido añadido [-noise_range, noise_range]
- `clip_threshold` - Umbral para recortar valores extremos

<a id="src.attacks.model_poisoning.lie_attack.LieAttack.attack"></a>

#### LieAttack.attack

```python
def attack(data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray
```

Ejecuta el ataque de mentira.

**Arguments**:

- `data` - Tupla (actualizaciones_benignas, modelo_referencia)
  

**Returns**:

- `np.ndarray` - Actualización maliciosa generada

<a id="src.attacks.model\_poisoning.statopt\_attack"></a>

# Module src.attacks.model\_poisoning.statopt\_attack [(ver código)](../src/attacks/model_poisoning/statopt_attack.py)

<a id="src.attacks.model_poisoning.statopt_attack.StatOptAttack"></a>

## StatOptAttack Objects

```python
class StatOptAttack(BaseAttack)
```

Implementa el ataque Statistical Optimization (StatOpt).

Utiliza optimización estadística para generar actualizaciones maliciosas
que maximizan el daño mientras evitan la detección.

<a id="src.attacks.model_poisoning.statopt_attack.StatOptAttack.attack"></a>

#### StatOptAttack.attack

```python
def attack(data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray
```

Ejecuta el ataque StatOpt.

**Arguments**:

- `data` - Tupla (actualizaciones_benignas, modelo_referencia)
  

**Returns**:

- `np.ndarray` - Actualización maliciosa optimizada

<a id="src.attacks.data\_poisoning.label\_flipping"></a>

# Module src.attacks.data\_poisoning.label\_flipping [(ver código)](../src/attacks/data_poisoning/label_flipping.py)

<a id="src.attacks.data_poisoning.label_flipping.LabelFlippingAttack"></a>

## LabelFlippingAttack Objects

```python
class LabelFlippingAttack(BaseAttack)
```

Implementa el ataque de inversión de etiquetas.

Este ataque modifica las etiquetas del conjunto de datos de entrenamiento
para degradar el rendimiento del modelo global.

<a id="src.attacks.data_poisoning.label_flipping.LabelFlippingAttack.attack"></a>

#### LabelFlippingAttack.attack

```python
def attack(data: tuple) -> tuple
```

Ejecuta el ataque de inversión de etiquetas.

**Arguments**:

- `data` _tuple_ - Tupla (x_train, y_train) con datos de entrenamiento
  

**Returns**:

- `tuple` - Datos modificados (x_train, y_train_poisoned)

