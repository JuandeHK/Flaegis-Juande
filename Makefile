# Variables
PYTHON = python
DOC_GENERATOR = pydoc-markdown
POST_PROCESS_SCRIPT = ./scripts/post_process_docs.py

CORE_CONFIG = pydoc-markdown-core.yml
CLIENT_CONFIG = pydoc-markdown-client.yml
CLIENT_MODELS = pydoc-markdown-client-models.yml
SERVER_CONFIG = pydoc-markdown-server.yml
STRATEGIES_CONFIG = pydoc-markdown-strategies_configuration.yml
CUSTOM_STRATEGIES_CONFIG = pydoc-markdown-custom_strategies.yml
DETECTORS_CONFIG = pydoc-markdown-detectors.yml
ATTACKS_CONFIG = pydoc-markdown-attacks.yml

CORE_DOC = docs/core_documentation.md
CLIENT_DOC = docs/client_documentation.md
CLIENT_MODELS_DOC = docs/client_models_documentation.md
SERVER_DOC = docs/server_documentation.md
STRATEGIES_CONFIG_DOC = docs/strategies_configuration_documentation.md
CUSTOM_STRATEGIES_DOC = docs/custom_strategies_documentation.md
DETECTORS_DOC = docs/detectors_documentation.md
ATTACKS_DOC = docs/attacks_documentation.md

# # Generar documentación
# docs: clean_docs $(CORE_DOC) $(CLIENT_DOC) $(SERVER_DOC) $(STRATEGIES_CONFIG_DOC) $(CUSTOM_STRATEGIES_DOC) $(CLIENT_MODELS_DOC) $(DETECTORS_DOC) $(ATTACKS_DOC)
# 	@echo "Ejecutando script de post-procesamiento..."
# 	$(PYTHON) $(POST_PROCESS_SCRIPT)
# 	@echo "Documentación generada y procesada correctamente."

# $(CORE_DOC): $(CORE_CONFIG)
# 	PYTHONPATH=src $(DOC_GENERATOR) $(CORE_CONFIG)

# $(CLIENT_DOC): $(CLIENT_CONFIG)
# 	$(DOC_GENERATOR) $(CLIENT_CONFIG)

# $(CLIENT_MODELS_DOC): $(CLIENT_MODELS)
#     PYTHONPATH=src $(DOC_GENERATOR) $(CLIENT_MODELS)

# $(SERVER_DOC): $(SERVER_CONFIG)
# 	PYTHONPATH=src $(DOC_GENERATOR) $(SERVER_CONFIG)

# $(STRATEGIES_CONFIG_DOC): $(STRATEGIES_CONFIG)
# 	$(DOC_GENERATOR) $(STRATEGIES_CONFIG)

# $(CUSTOM_STRATEGIES_DOC): $(CUSTOM_STRATEGIES_CONFIG)
# 	PYTHONPATH=src $(DOC_GENERATOR) $(CUSTOM_STRATEGIES_CONFIG)

# $(DETECTORS_DOC): $(DETECTORS_CONFIG)
# 	PYTHONPATH=src $(DOC_GENERATOR) $(DETECTORS_CONFIG)

# $(ATTACKS_DOC): $(ATTACKS_CONFIG)
# 	PYTHONPATH=src $(DOC_GENERATOR) $(ATTACKS_CONFIG)

# # Limpiar documentación generada
# clean_docs:
# 	@echo "Eliminando documentación generada previamente..."
# 	rm -f $(CORE_DOC) $(CLIENT_DOC) $(SERVER_DOC) $(STRATEGIES_CONFIG_DOC) $(CUSTOM_STRATEGIES_DOC) $(CLIENT_MODELS_DOC) $(DETECTORS_DOC) $(ATTACKS_DOC)

# Ejecutar el programa (incluye generación de docs)
run: docs
	PYTHONPATH=src $(PYTHON) main.py $(filter-out $@,$(MAKECMDGOALS))

# # Limpiar todos los archivos generados
# clean: clean_docs
# 	@echo "Limpieza completa."

# Verificar dependencias
check:
	@command -v $(DOC_GENERATOR) >/dev/null 2>&1 || { echo >&2 "Error: $(DOC_GENERATOR) no está instalado."; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo >&2 "Error: $(PYTHON) no está instalado."; exit 1; }

# Ayuda
help:
	@echo "Tareas disponibles:"
	@echo "  make docs       - Limpiar, generar documentación y ejecutar el post-procesamiento."
	@echo "  make clean_docs - Limpiar únicamente los archivos de documentación."
	@echo "  make run        - Generar docs, ejecutar el post-procesamiento y ejecutar el programa principal."
	@echo "  make clean      - Limpiar todos los archivos generados."
	@echo "  make check      - Verificar que las herramientas necesarias están instaladas."
