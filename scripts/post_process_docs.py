import re
import os

def normalize_module_name(module_name: str) -> str:
    # Escapar únicamente los underscores para que sean \_ en el patrón de búsqueda
    return module_name.replace("_", "\\_")

def add_source_links(file_path: str, module_mappings: dict):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for module_name, source_path in module_mappings.items():
        # Normalizar el módulo para que coincida tanto con \_ como con _
        normalized_module = normalize_module_name(module_name)

        # Patrón para aceptar ambas formas (_ y \_) en los encabezados
        pattern = (
            rf'<a id="{module_name}"></a>'
            rf'\s*'
            rf'# Module (?:{normalized_module}|{module_name})'
        )
        replacement = (
            f'<a id="{module_name}"></a>\n\n'
            f'# Module {normalized_module} [(ver código)]({source_path})'
        )

        new_content, num_subs = re.subn(
            pattern, replacement, content, flags=re.MULTILINE | re.DOTALL
        )

        if num_subs > 0:
            print(f"Enlace añadido para: {module_name}")
            content = new_content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    mappings = {
        "main": "../main.py",
        "src.orchestrator.fl\_orchestrator": "../src/orchestrator/fl_orchestrator.py",
        "client.client\_manager": "../src/client/client_manager.py",
        "server.server\_manager": "../src/server/server_manager.py",

        "src.server.strategy\_configurators.base\_strategy\_configurator": "../src/server/strategy_configurators/base_strategy_configurator.py",
        "src.server.strategy\_configurators.fedavg\_strategy\_configuration": "../src/server/strategy_configurators/fedavg_strategy_configuration.py",
        "src.server.strategy\_configurators.weighted\_median\_strategy\_configuration": "../src/server/strategy_configurators/weighted_median_strategy_configuration.py",
        "src.server.strategy\_configurators.fourier\_strategy\_configuration": "../src/server/strategy_configurators/fourier_strategy_configuration.py",
        "src.server.strategy\_configurators.trimmed\_mean\_strategy\_configuration": "../src/server/strategy_configurators/trimmed_mean_strategy_configuration.py",

        "src.custom\_strategies.base\_strategy": "../src/custom_strategies/base_strategy.py",
        "src.custom\_strategies.fedavg\_strategy": "../src/custom_strategies/fedavg_strategy.py",
        "src.custom\_strategies.weighted\_median\_strategy": "../src/custom_strategies/weighted_median_strategy.py",
        "src.custom\_strategies.fourier\_strategy": "../src/custom_strategies/fourier_strategy.py",
        "src.custom\_strategies.trimmed\_mean\_strategy": "../src/custom_strategies/trimmed_mean_strategy.py",

        "src.detectors.base\_clustering\_detector": "../src/detectors/base_clustering_detector.py",
        "src.detectors.clustering\_strategy": "../src/detectors/clustering_strategy.py", 
        "src.detectors.clustering\_context": "../src/detectors/clustering_context.py",
        "src.detectors.spectral\_detector": "../src/detectors/spectral_detector.py",
        "src.detectors.hierarchical\_detector": "../src/detectors/hierarchical_detector.py",
        "src.detectors.sign\_guard\_detector": "../src/detectors/sign_guard_detector.py",

        "models.client\_config": "../src/models/client_config.py",
        "models.femnist\_model_builder": "../src/models/femnist_model_builder.py",
        "models.mlp\_model\_builder": "../src/models/mlp_model_builder.py",
        "models.model\_builder": "../src/models/model_builder.py",

        "src.attacks.base\_attack": "../src/attacks/base_attack.py",
        "src.attacks.attack\_factory": "../src/attacks/attack_factory.py",
        "src.attacks.model\_poisoning.min\_max\_attack": "../src/attacks/model_poisoning/min_max_attack.py",
        "src.attacks.model\_poisoning.min\_sum\_attack": "../src/attacks/model_poisoning/min_sum_attack.py",
        "src.attacks.model\_poisoning.lie\_attack": "../src/attacks/model_poisoning/lie_attack.py",
        "src.attacks.model\_poisoning.statopt\_attack": "../src/attacks/model_poisoning/statopt_attack.py",
        "src.attacks.data\_poisoning.label\_flipping": "../src/attacks/data_poisoning/label_flipping.py",
    }
    # Archivos donde quieres que busque los patrones
    docs_files = [
        "docs/core_documentation.md",
        "docs/client_documentation.md",
        "docs/server_documentation.md",
        "docs/strategies_configuration_documentation.md",
        "docs/custom_strategies_documentation.md",
        "docs/client_models_documentation.md",
        "docs/detectors_documentation.md",
        "docs/attacks_documentation.md"
    ]

    for doc_file in docs_files:
        if os.path.exists(doc_file):
            print(f"\nProcesando archivo: {doc_file}")
            add_source_links(doc_file, mappings)

if __name__ == "__main__":
    main()

