import tensorflow as tf
import logging
import os

# Es crucial que este script se ejecute desde la raíz del proyecto
from src.utils.data_loader import DataLoader
from src.models.convlstm_model_builder import ConvLSTMModelBuilder

# --- Configuración Mínima (de tu config.yml) ---
CLIENT_ID = 0
CONFIG = {
    "data": {
        'base_dir': '.', # Usamos el directorio actual
        'dataset_type': 'pleidata',
        'target_col': 'dif_cons_real',
        'feature_cols': ['dif_cons_real', 'dif_cons_smooth', 'V2', 'V4', 'V12', 'V26', 'Hour_1', 'Hour_2', 'Hour_3', 'Season_1', 'Season_2', 'Season_3', 'Season_4', 'tmed', 'hrmed', 'radmed', 'vvmed', 'dvmed', 'prec', 'dewpt', 'dpv'],
        'window_size': 12,
        'separator': ';'
    },
    "model": {
        'type': 'convlstm',
        'learning_rate': 0.0003,
        'batch_size': 32
    },
    "training": {
        "local_epochs": 5 # Usamos 5 epochs para la prueba
    }
}

if __name__ == "__main__":
    # Ignoramos los warnings de TensorFlow GPU, ya sabemos que usarás CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.basicConfig(level=logging.INFO)

    print("--- INICIANDO PRUEBA DE ENTRENAMIENTO LOCAL ---")

    # 1. Cargar datos para un cliente
    print(f"\n[PASO 1] Cargando datos para el cliente {CLIENT_ID}...")
    loader = DataLoader(base_dir=CONFIG["data"]["base_dir"], data_config=CONFIG["data"])
    (x_train, y_train), (x_test, y_test) = loader.load_data_for_client(CLIENT_ID)
    print(f"Datos cargados. Forma de X_train: {x_train.shape}, Forma de y_train: {y_train.shape}")

    # 2. Construir el modelo
    print("\n[PASO 2] Construyendo el modelo ConvLSTM...")
    builder = ConvLSTMModelBuilder(input_shape=(12, 1, 21, 1))
    model = builder.build()
    model.summary()

    # 3. Entrenar el modelo
    print(f"\n[PASO 3] LLAMANDO A model.fit()... (Esto puede tardar unos minutos)")
    print(f"Epochs: {CONFIG['training']['local_epochs']}, Batch Size: {CONFIG['model']['batch_size']}")
    
    history = model.fit(
        x_train,
        y_train,
        epochs=CONFIG['training']['local_epochs'],
        batch_size=CONFIG['model']['batch_size'],
        validation_data=(x_test, y_test),
        verbose=1 # verbose=1 nos mostrará una barra de progreso por época
    )

    print("\n--- ¡ENTRENAMIENTO LOCAL COMPLETADO! ---")
    print(f"Loss final: {history.history['loss'][-1]}")