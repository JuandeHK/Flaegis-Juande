# Documentación FLAegis

## Estructura de la Documentación

La documentación está dividida en las siguientes secciones:

### [Core Documentation](core_documentation.md)
- Documentación del punto de entrada principal (`main.py`)
- Documentación del orquestador del sistema (`orchestrator/fl_orchestrator.py`)

### [Client Documentation](client_documentation.md)
- Documentación del cliente federado
- Implementación del cliente Flower

### [Server Documentation](server_documentation.md)
- Documentación del servidor federado
- Gestión de la comunicación con clientes

### [Strategies Documentation](custom_strategies_documentation.md)
- Documentación de las implementaciones completas de estrategias de agregación que extienden las capacidades básicas de Flower
- Implementaciones de FedAvg, Weighted Median, Trimmed Median y Fourier

### [Strategies Configuration Documentation](strategies_configuration_documentation.md)
- Documentación de la configuracion de la implementación de las estrategias de agregación
- Implementaciones de FedAvg, Weighted Median y Fourier

### [Client Models Documentation](client_models_documentation.md)
- Documentacion de los modelos de entrenamiento de los clientes
- Implementacion de FeMnist y MLP

### [Detectors Documentation](detectors_documentation.md)
- Documentación de los detectores de clientes maliciosos
- Implementaciones de detectores basados en clustering (Spectral, Hierarchical)
- Implementación del detector SignGuard

### [Attacks Documentation](attacks_documentation.md)
- Documentación de los ataques de los clientes maliciosos 
- **Tipos de Ataques:**
  - **Model Poisoning**: 
    - **MinMaxAttack**
    - **MinSumAttack**
    - **LieAttack**
    - **StatOptAttack**
  
  - **Data Poisoning**: 
    - **LabelFlippingAttack**

## Generación de Documentación

La documentación se genera automáticamente usando `pydoc-markdown` a partir de los docstrings en el código.

Para generar/actualizar la documentación:
```bash
make docs
```

O manualmente:
```bash
pydoc-markdown pydoc-markdown-core.yml
pydoc-markdown pydoc-markdown-client.yml
pydoc-markdown pydoc-markdown-server.yml
pydoc-markdown pydoc-markdown-strategies.yml
pydoc-markdown pydoc-markdown-client-models.yml
```

## Añadir nueva documentación
Para añadir nueva documentacion tienes que:
- Crear un archivo pydoc-markdown-[lo_que_sea].yml

- Modificar el makefile, fijate en cualquiera que ya esté hecho y mira los pasos que tienes que copiar y pegar

- Modificar el archivo, scripts/post_process_docs.py para añadir el enlace al codigo del módulo nuevo que quieres añadir.