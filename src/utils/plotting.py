
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # ← AÑADIR r2_score

from src.models.convlstm_model_builder import ConvLSTMModelBuilder
from src.models.transformer_model_builder import TransformerModelBuilder
from src.utils.data_loader import DataLoader

# class ResultPlotter:
#     """Se encarga de evaluar el modelo global final y generar todas las gráficas."""
#     def __init__(self, config: Dict, base_dir: str):
#         self.config = config
#         self.base_dir = Path(base_dir)
#         self.logger = logging.getLogger(__name__)
#         self.plot_dir = self.base_dir / "plots"
#         self.plot_dir.mkdir(exist_ok=True)

#     def generate_plots(self):
#         """Función principal que orquesta la generación de todas las métricas y plots."""
#         self.logger.info("--- INICIANDO FASE DE PLOTEO Y EVALUACIÓN FINAL ---")
#         try:
#             model_type = self.config['model']['type']
#             self.logger.info(f"Generando plots para el modelo final de tipo: {model_type}")

#             # 1. Cargar el dataset COMPLETO para una evaluación global
#             full_data_path = self.base_dir / "data" / "pleidata" / "data-model-consumoA-60min.csv"
#             if not full_data_path.exists():
#                 self.logger.error(f"No se encontró el dataset global en {full_data_path}. No se pueden generar plots.")
#                 return
#             df_full = pd.read_csv(full_data_path, sep=';')

#             # 2. Cargar los pesos del modelo final guardado por el servidor
#             model_params_path = self.base_dir / "final_global_model.npz"
#             if not model_params_path.exists():
#                 self.logger.error(f"No se encontró el fichero de pesos del modelo en {model_params_path}.")
#                 return
            
#             params_npz = np.load(model_params_path)
#             final_weights = [params_npz[key] for key in params_npz.files]

#             # 3. Preparar datos, predecir y des-escalar
#             y_test_inverted, y_pred_inverted, df_model = self._get_inverted_predictions(df_full, final_weights)
            
#             # 4. Calcular y mostrar métricas
#             self._calculate_and_log_metrics(y_test_inverted, y_pred_inverted, df_model, df_full)

#             # 5. Generar gráficas
#             self._plot_point_consumption(y_test_inverted, y_pred_inverted)
#             self._plot_cumulative_consumption(y_test_inverted, y_pred_inverted, df_full, df_model)
            
#             self.logger.info(f"--- PLOTEO FINALIZADO. Gráficas guardadas en: '{self.plot_dir}' ---")

#         except Exception as e:
#             self.logger.error(f"Error durante la generación de plots: {e}", exc_info=True)

#     def _get_inverted_predictions(self, df: pd.DataFrame, final_weights: list):
#         """Prepara datos, evalúa y devuelve predicciones des-escaladas."""
#         target_col = self.config['data']['target_col']
#         feature_cols = self.config['data']['feature_cols']
        
#         df_model = df[feature_cols + ['cons_total']].copy()
#         df_model.dropna(inplace=True)

#         scaler = MinMaxScaler()
#         df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns)
        
#         X_all = df_scaled[feature_cols].values.astype(np.float32)
#         y_all = df_scaled[[target_col]].values.astype(np.float32)

#         loader = DataLoader(self.base_dir, self.config['data'])
#         model_type = self.config['model']['type']
#         window_size = self.config['data']['window_size']

#         if model_type == 'transformer':
#             X_seq, y_seq = loader._create_transformer_sequences(X_all, y_all.ravel(), window_size)
#         else:
#             X_seq, y_seq = loader._create_convlstm_sequences(X_all, y_all.ravel(), window_size)
            
#         _, X_test, _, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

#         if model_type == 'transformer':
#             builder = TransformerModelBuilder(input_shape=X_test.shape[1:])
#         else:
#             builder = ConvLSTMModelBuilder(input_shape=X_test.shape[1:])
        
#         model = builder.build()
#         model.set_weights(final_weights)
        
#         y_pred = model.predict(X_test)
        
#         temp_test = pd.DataFrame(np.zeros((len(y_test), len(df_model.columns))), columns=df_model.columns)
#         temp_pred = temp_test.copy()
#         temp_test[target_col] = y_test
#         temp_pred[target_col] = y_pred
        
#         test_inverted_full = scaler.inverse_transform(temp_test)
#         pred_inverted_full = scaler.inverse_transform(temp_pred)

#         target_idx = df_model.columns.get_loc(target_col)
#         y_test_inverted = test_inverted_full[:, target_idx]
#         y_pred_inverted = pred_inverted_full[:, target_idx]
        
#         return y_test_inverted, y_pred_inverted, df_model

#     def _calculate_and_log_metrics(self, y_true, y_pred, df_model, df_full):
#         """Calcula e imprime las métricas."""
#         # Métricas de la diferencia
#         mse_dif = mean_squared_error(y_true, y_pred)
#         mae_dif = mean_absolute_error(y_true, y_pred)
#         mask = y_true != 0
#         mape_dif = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
#         print("\n" + "="*55)
#         print("--- MÉTRICAS DEL MODELO GLOBAL FINAL (escala original) ---")
#         print(f"  MSE (dif_cons_real): {mse_dif:.4f}")
#         print(f"  MAE (dif_cons_real): {mae_dif:.4f}")
#         print(f"  MAPE (dif_cons_real): {mape_dif:.2f} %")

#         # Métricas del reconstruido
#         last_real_value = df_full['cons_total'].iloc[df_model.index[0] + len(y_true) -1]
#         y_true_rec = np.cumsum(y_true) + last_real_value
#         y_pred_rec = np.cumsum(y_pred) + last_real_value
#         mse_rec = mean_squared_error(y_true_rec, y_pred_rec)
#         mae_rec = mean_absolute_error(y_true_rec, y_pred_rec)
#         mask_rec = y_true_rec != 0
#         mape_rec = np.mean(np.abs((y_true_rec[mask_rec] - y_pred_rec[mask_rec]) / y_true_rec[mask_rec])) * 100

#         print("\n--- MÉTRICAS DEL CONSUMO ACUMULADO RECONSTRUIDO ---")
#         print(f"  MSE (reconstruido): {mse_rec:.4f}")
#         print(f"  MAE (reconstruido): {mae_rec:.4f}")
#         print(f"  MAPE (reconstruido): {mape_rec:.2f} %")
#         print("="*55 + "\n")

#     def _plot_point_consumption(self, y_true, y_pred):
#         """Genera la gráfica del consumo puntual."""
#         plt.figure(figsize=(15, 6))
#         plt.plot(y_true, label='Real (dif_cons)', color='blue', alpha=0.8)
#         plt.plot(y_pred, label='Predicho (dif_cons)', color='red', linestyle='--')
#         plt.title(f"Consumo Puntual: Real vs. Predicho (Modelo: {self.config['model']['type']})")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.6)
        
#         filename = self.plot_dir / f"plot_puntual_{self.config['model']['type']}.png"
#         plt.savefig(filename)
#         plt.close()
#         self.logger.info(f"Gráfica de consumo puntual guardada en: {filename}")

#     def _plot_cumulative_consumption(self, y_test_inverted, y_pred_inverted, df_full, df_model):
#         """Reconstruye y genera la gráfica del consumo acumulado."""
#         last_real_value = df_full['cons_total'].iloc[df_model.index[0] + len(y_test_inverted) -1]
#         y_test_rec = np.cumsum(y_test_inverted) + last_real_value
#         y_pred_rec = np.cumsum(y_pred_inverted) + last_real_value
        
#         plt.figure(figsize=(15, 7))
#         plt.plot(y_test_rec, label='Real (Acumulado)', color='blue')
#         plt.plot(y_pred_rec, label='Predicho (Acumulado)', color='red', linestyle='--')
#         plt.title(f"Consumo Acumulado Reconstruido (Modelo: {self.config['model']['type']})")
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.6)
        
#         filename = self.plot_dir / f"plot_acumulado_{self.config['model']['type']}.png"
#         plt.savefig(filename)
#         plt.close()
#         self.logger.info(f"Gráfica de consumo acumulado guardada en: {filename}")
