import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

class DataLoader:
    def __init__(self, base_dir: str, data_config: dict = None):
        self.base_dir = Path(base_dir)
        self.data_config = data_config if data_config else {}
        self.logger = logging.getLogger(__name__)

    def _create_sequences(self, features, target, window_size):
        """Función base unificada para crear secuencias."""
        X_list, Y_list = [], []
        for i in range(len(features) - window_size):
            seq_x = features[i: i + window_size]
            seq_y = target[i + window_size]
            X_list.append(seq_x)
            Y_list.append(seq_y)
        return np.array(X_list), np.array(Y_list)

    def load_data_for_client(self, client_id: int):
        """
        Punto de entrada principal para cargar datos para un cliente.
        Maneja la lógica de despacho según el tipo de dataset.
        """
        dataset_type = self.data_config.get('dataset_type', 'pleidata')
        self.logger.info(f"Iniciando carga para cliente {client_id} con dataset_type: '{dataset_type}'")

        if dataset_type == 'pleidata':
            return self._load_pleidata_for_client(client_id)
        elif dataset_type == 'femnist':
            raise NotImplementedError("La carga de datos para 'femnist' no está implementada.")
        else:
            raise ValueError(f"Dataset type '{dataset_type}' no soportado.")

    def _load_pleidata_for_client(self, client_id: int):
        """Carga y preprocesa datos de consumo eléctrico para un cliente."""
        # 1. Obtener Configuración
        config = self.data_config
        feature_cols = config.get('feature_cols', [])
        target_col = config.get('target_col', '')
        window_size = config.get('window_size', 12)
        model_type = self.data_config.get('type', 'convlstm')

        if not feature_cols or not target_col:
            raise ValueError("Faltan 'feature_cols' o 'target_col' en la configuración.")

        # 2. Cargar Datos
        file_path = self.base_dir / 'data' / 'pleidata' / f"data_party{client_id}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Fichero de datos no encontrado para cliente {client_id} en {file_path}")
        
        df = pd.read_csv(file_path, sep=config.get('separator', ';'), index_col=0)

        # 3. Limpieza (sin escalar aquí)
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=feature_cols, inplace=True)

        if len(df) <= window_size:
            raise ValueError(f"Datos insuficientes tras limpieza ({len(df)} filas) para window_size ({window_size})")

        # 4. Escalado (UNA SOLA VEZ)
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        # 5. Crear Secuencias
        features = df[feature_cols].values.astype(np.float32)
        target = df[target_col].values.astype(np.float32)
        
        X_seq, y_seq = self._create_sequences(features, target, window_size)

        if X_seq.size == 0:
            self.logger.warning(f"Cliente {client_id}: No se generaron secuencias.")
            # Devolvemos tuplas vacías con la forma correcta
            shape_x = (0, window_size, len(feature_cols)) if model_type == 'transformer' else (0, window_size, 1, len(feature_cols), 1)
            return (np.empty(shape_x), np.empty((0,))), (np.empty(shape_x), np.empty((0,)))

        # 6. Reshape específico para el modelo
        if model_type == 'convlstm':
            n_samples, timesteps, n_features = X_seq.shape
            X_seq = X_seq.reshape((n_samples, timesteps, 1, n_features, 1))

        # 7. División Train/Test (CORREGIDO)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )

        self.logger.info(f"Cliente {client_id}: Carga y preprocesamiento completados exitosamente.")
        self.logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        self.logger.info(f"Test shapes:  X={X_test.shape}, y={y_test.shape}")

        return (X_train, y_train), (X_test, y_test)

#CODIGO QUE FUNCIONA PERO SIN REFACTORIZAR:
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from pathlib import Path
# import logging

# class DataLoader:
#     def __init__(self, base_dir: str, data_config: dict = None):
#         self.base_dir = Path(base_dir)         
#         self.data_config = data_config if data_config else {}
#         self.logger = logging.getLogger(__name__)

#     def _create_transformer_sequences(self, features, target, window_size):
#         """Crea secuencias 3D (samples, timesteps, features) para un Transformer."""
#         X_list, Y_list = [], []
#         for i in range(len(features) - window_size):
#             seq_x = features[i: i + window_size]
#             seq_y = target[i + window_size]
#             X_list.append(seq_x)
#             Y_list.append(seq_y)

#         if not X_list:
#             # Devuelve arrays 3D vacíos
#             return np.empty((0, window_size, features.shape[1])), np.empty((0,))

#         return np.array(X_list), np.array(Y_list)

#     def _create_convlstm_sequences(self, features, target, window_size):
#         X_list, Y_list = [], []
#         for i in range(len(features) - window_size):
#             seq_x = features[i: i + window_size]
#             seq_y = target[i + window_size]
#             X_list.append(seq_x)
#             Y_list.append(seq_y)
        
#         if not X_list:
#             return np.empty((0, window_size, 1, features.shape[1], 1)), np.empty((0,))

#         X_arr = np.array(X_list)
#         Y_arr = np.array(Y_list)
        
#         # Reshape para ConvLSTM2D: (samples, timesteps, rows, cols, channels)
#         n_samples, timesteps, n_features = X_arr.shape
#         X_arr = X_arr.reshape((n_samples, timesteps, 1, n_features, 1))
#         return X_arr, Y_arr

#     # En src/utils/data_loader.py
#     # En src/utils/data_loader.py

#     def _preprocess_dataframe(self, df, feature_cols):
#         """Preprocesa el DataFrame de forma robusta antes de crear secuencias."""
#         self.logger.info(f"DataFrame original shape: {df.shape}")
#         self.logger.info(f"Columnas disponibles: {df.columns.tolist()}")

#         # Seleccionar solo las columnas de features que existen en el DF
#         existing_feature_cols = [col for col in feature_cols if col in df.columns]
        
#         # --- Forzar todas las columnas a ser numéricas ---
#         # Esto convertirá cualquier valor no numérico en NaN (Not a Number)
#         for col in existing_feature_cols:
#             df[col] = pd.to_numeric(df[col], errors='coerce')

#         # --- Eliminar filas que contengan CUALQUIER NaN ---
#         # Esto limpia las filas donde la conversión a número falló.
#         df.dropna(inplace=True)
        
#         self.logger.info(f"Datos después de limpiar NaN y valores no numéricos: {df.shape}")
        
#         # Escalar las columnas
#         if not df.empty:
#             scaler = MinMaxScaler()
#             df[existing_feature_cols] = scaler.fit_transform(df[existing_feature_cols])
            
#         return df[existing_feature_cols]

#     def load_consumption_data_for_client(self, client_id: int):
#         """
#         Carga datos de consumo eléctrico fragmentados para un cliente específico.
#         Cliente 0 -> data_party0.csv, Cliente 1 -> data_party1.csv, etc.
#         """
#         # Construir la ruta del archivo fragmentado específico para este cliente
#         fragment_filename = f"data_party{client_id}.csv"
#         fragment_path = Path(self.base_dir) / "data" / "pleidata" / fragment_filename
        
#         self.logger.info(f"Cliente {client_id}: Cargando datos desde {fragment_path}")
        
#         # Verificar que el archivo existe
#         if not fragment_path.exists():
#             raise FileNotFoundError(f"Cliente {client_id}: Archivo fragmentado no encontrado: {fragment_path}")

#         # Obtener configuración
#         target_col = self.data_config.get('target_col', 'dif_cons_real')
#         feature_cols = self.data_config.get('feature_cols', [])
#         window_size = self.data_config.get('window_size', 12)
#         separator = self.data_config.get('separator', ';')

#         self.logger.info(f"Cliente {client_id}: target_col='{target_col}', features={len(feature_cols)}")
#         self.logger.info(f"Cliente {client_id}: window_size={window_size}, separator='{separator}'")

#         # Cargar el archivo CSV específico del cliente
#         try:
#             df = pd.read_csv(fragment_path, sep=separator)
#             self.logger.info(f"Cliente {client_id}: Archivo cargado exitosamente. Shape: {df.shape}")
#         except Exception as e:
#             self.logger.error(f"Cliente {client_id}: Error leyendo CSV: {e}")
#             raise
        
#         # Preprocesar DataFrame
#         try:
#             df_clean = self._preprocess_dataframe(df, feature_cols)
#         except Exception as e:
#             self.logger.error(f"Cliente {client_id}: Error en preprocesamiento: {e}")
#             raise
        
#         # Verificar que hay suficientes datos
#         if len(df_clean) <= window_size:
#             raise ValueError(f"Cliente {client_id}: Datos insuficientes - {len(df_clean)} filas <= window_size {window_size}")

#         # Verificar que target_col está en feature_cols
#         if target_col not in feature_cols:
#             raise ValueError(f"Cliente {client_id}: target_col '{target_col}' debe estar en feature_cols")

#         try:
#             # Escalado
#             self.logger.info(f"Cliente {client_id}: Aplicando escalado...")
#             scaler = MinMaxScaler()
#             df_scaled = pd.DataFrame(
#                 scaler.fit_transform(df_clean), 
#                 columns=df_clean.columns
#             )
            
#             X_all = df_scaled[feature_cols].values
#             y_all = df_scaled[target_col].values
            
#             self.logger.info(f"Cliente {client_id}: X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
#             self.logger.info(f"Cliente {client_id}: X_all range: [{X_all.min():.4f}, {X_all.max():.4f}]")
#             self.logger.info(f"Cliente {client_id}: y_all range: [{y_all.min():.4f}, {y_all.max():.4f}]")
            
#         except Exception as e:
#             self.logger.error(f"Cliente {client_id}: Error en escalado: {e}")
#             raise
        
#         # Crear secuencias
#         try:
#             self.logger.info(f"Cliente {client_id}: Creando secuencias...")
#             X_seq, y_seq = self._create_convlstm_sequences(X_all, y_all, window_size)
#             self.logger.info(f"Cliente {client_id}: Secuencias creadas - X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
#         except Exception as e:
#             self.logger.error(f"Cliente {client_id}: Error creando secuencias: {e}")
#             raise
        
#         # Verificar que hay suficientes secuencias
#         if len(X_seq) < 2:
#             raise ValueError(f"Cliente {client_id}: Secuencias insuficientes para dividir - {len(X_seq)}")
        
#         # División train/test
#         try:
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_seq, y_seq, test_size=0.2, shuffle=False
#             )
            
#             self.logger.info(f"Cliente {client_id}: División exitosa")
#             self.logger.info(f"Cliente {client_id}: Train X: {X_train.shape}, y: {y_train.shape}")
#             self.logger.info(f"Cliente {client_id}: Test X: {X_test.shape}, y: {y_test.shape}")
            
#             return (X_train, y_train), (X_test, y_test)
            
#         except Exception as e:
#             self.logger.error(f"Cliente {client_id}: Error en división train/test: {e}")
#             raise

#     def load_femnist(self, client_id: int):
#         """Carga datos de FEMNIST para un cliente específico"""
#         # Implementación básica para FEMNIST
#         self.logger.warning(f"Cliente {client_id}: FEMNIST no implementado completamente")
#         empty_x = np.empty((0, 28, 28, 1))
#         empty_y = np.empty((0,))
#         return (empty_x, empty_y), (empty_x, empty_y)

#     def load_data_for_client(self, client_id: int):
#         """Carga datos para un cliente específico según el tipo de dataset."""
#         dataset_type = self.data_config.get('dataset_type', 'consumption')
#         self.logger.info(f"Iniciando carga para cliente {client_id} con dataset_type: '{dataset_type}'")

#         if dataset_type == 'pleidata':
#             feature_cols = self.data_config.get('feature_cols', [])
#             target_col = self.data_config.get('target_col', '')
#             window_size = self.data_config.get('window_size', 12)
#             separator = self.data_config.get('separator', ';')

#             if not feature_cols or not target_col:
#                 raise ValueError("Faltan 'feature_cols' o 'target_col' en la configuración para pleidata")

#             # Construye la ruta al fichero del cliente
#             data_dir = self.base_dir / 'data' / 'pleidata'
#             file_path = data_dir / f"data_party{client_id}.csv"
#             self.logger.info(f"Cargando datos desde: {file_path}")

#             if not file_path.exists():
#                 self.logger.error(f"Fichero no encontrado: {file_path}")
#                 raise FileNotFoundError(f"Fichero de datos no encontrado para el cliente {client_id} en {file_path}")

#             # Leemos el CSV, indicando que la primera columna es el índice (para ignorar 'Unnamed: 0')
#             df = pd.read_csv(file_path, sep=separator, index_col=0)

#             # Preprocesamiento y creación de secuencias
#             processed_df = self._preprocess_dataframe(df, feature_cols)
#             features = processed_df[feature_cols].values.astype(np.float32)
#             target = processed_df[target_col].values.astype(np.float32)

#             model_type = self.data_config.get('type', 'convlstm') # Asumimos que podemos pasar el tipo de modelo

#             x, y = self._create_convlstm_sequences(features, target, window_size)

#             if model_type == 'transformer':
#                 x, y = self._create_transformer_sequences(features, target, window_size)
#             else:
#                  x, y = self._create_convlstm_sequences(features, target, window_size)

#             if x.size == 0 or y.size == 0:
#                 self.logger.warning(f"Cliente {client_id}: No se generaron secuencias. Comprueba window_size y el tamaño de los datos.")
#                 # Devolvemos tuplas vacías con la forma correcta para evitar errores posteriores
#                 return (np.empty((0, window_size, 1, len(feature_cols), 1)), np.empty((0,))), \
#                         (np.empty((0, window_size, 1, len(feature_cols), 1)), np.empty((0,)))

#             X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=False)
#             return (X_train, y_train), (X_test, y_test)

#         elif dataset_type == 'femnist':
#             self.logger.warning("La carga de datos para 'femnist' no está implementada.")
#             raise NotImplementedError("La carga de datos para 'femnist' no está implementada.")

#         else:
#             raise ValueError(f"Dataset type '{dataset_type}' no soportado. Soportados: ['pleidata', 'femnist']")



