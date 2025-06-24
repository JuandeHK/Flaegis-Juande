# Documentación del archivo YAML

Este archivo describe los parámetros de configuración del sistema.

## Secciones principales

### **Data**
- **base_dir**: Directorio base donde se almacenan los datos. Valor predeterminado: `"data/"`. 

### **Training**
- **rounds**: Número de rondas de entrenamiento. Valor predeterminado: `50`.
- **clients**: Número total de clientes en el sistema. Valor predeterminado: `50`.
- **min_fit_clients**: Número mínimo de clientes necesarios para ajustar el modelo. Valor predeterminado: `45`.
- **min_eval_clients**: Número mínimo de clientes necesarios para evaluar el modelo. Valor predeterminado: `45`.

### **Model**
- **type**: Tipo de modelo utilizado en el aprendizaje federado (`"femnist"`, por ejemplo). Valor predeterminado: `"femnist"`.
- **learning_rate**: Tasa de aprendizaje del modelo. Valor predeterminado: `0.001`.
- **batch_size**: Tamaño del lote para el entrenamiento. Valor predeterminado: `32`.

### **Detection**
- **threshold**: Umbral para clasificar clientes como maliciosos. Valor predeterminado: `0.6`.
- **min_clients_per_cluster**: Mínimo de clientes por clúster en el análisis de detección. Valor predeterminado: `5`.

### **Evaluation**
- **batch_size**: Tamaño del lote utilizado durante la evaluación del modelo. Valor predeterminado: `64`.

### **Aggregation**
- **strategy**: Estrategia de agregación utilizada en el aprendizaje federado. Valores posibles:
  - `"fedavg"`: Agregación basada en promedio simple.
  - `"weighted_median"`: Agregación basada en la mediana ponderada.
  - `"fourier"`: Agregación basada en transformadas de Fourier.
- **params**: Diccionario con parámetros específicos para la estrategia de agregación seleccionada. Ejemplo:
  - **accept_failures**: Indica si se deben aceptar fallos en los clientes (`true` o `false`). Aplicable a estrategias como `weighted_median`.
  - **threshold**: Umbral utilizado en estrategias específicas (e.g., `weighted_median`).
  - **other_params**: Parámetros adicionales dependiendo de la estrategia seleccionada.

## Ejemplo de configuración completa

```yaml
data:
  base_dir: "data/"

training:
  rounds: 50
  clients: 50
  min_fit_clients: 45
  min_eval_clients: 45

model:
  type: "femnist"
  learning_rate: 0.001
  batch_size: 32

detection:
  threshold: 0.6
  min_clients_per_cluster: 5

evaluation:
  batch_size: 64

aggregation:
  strategy: "weighted_median"
  params:
    accept_failures: true
    threshold: 0.5
```