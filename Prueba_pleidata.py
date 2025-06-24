import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================================
# 1. Cargar datos
# ============================================================================
df = pd.read_csv("data-model-consumoA-60T.csv", sep=';')

# Ejemplo: asumiendo que la columna objetivo es 'cons_total'
target_col = 'cons_total'

# Suponemos que la columna a predecir es 'cons_total'
target_col = 'cons_total'

# ============================================================================
# 2. Seleccionar features y target
# (ajusta las columnas según tu dataset real)
# ============================================================================
feature_cols = [
    'dif_cons_real',
    'dif_cons_smooth',
    'V2', 'V4', 'V12', 'V26',
    'Hour_1', 'Hour_2', 'Hour_3',
    'Season_1', 'Season_2', 'Season_3', 'Season_4',
    'tmed', 'hrmed', 'radmed', 'vvmed', 'dvmed', 'prec', 'dewpt', 'dpv'
]
# Crea un df que contiene solamente las columnas de entrada y la objetivo y lo guarda en df_model
df_model = df[feature_cols + [target_col]].copy() 
df_model.dropna(inplace=True)  # eliminar posibles valores faltantes

# ============================================================================
# 3. Escalado
# ============================================================================
scaler = MinMaxScaler()
#Aplica el escalado y lo guarda en df_scaled
df_scaled = pd.DataFrame(scaler.fit_transform(df_model), columns=df_model.columns) 

#Conjunto con las variables
X_all = df_scaled[feature_cols].values 
#Conjunto solo con la variable objetivo
y_all = df_scaled[target_col].values


# ============================================================================
# 4. Crear secuencias (ventanas de tiempo) para ConvLSTM2D

#El objetivo es que el modelo aprenda la relación entre las condiciones 
# de las últimas 3 horas (features) y el consumo que habrá en la hora siguiente 
# (target).

# Imagina que tienes datos de consumo eléctrico por hora durante un día (24 valores):
# [5, 6, 5, 7, 8, 9, 10, 11, 12, 10, 9, 8, 7, 8, 9, 10, 12, 14, 15, 13, 11, 10, 9, 8]

# Y quieres entrenar un modelo para que, viendo el consumo de las últimas 3 horas (window_size = 3), prediga el consumo de la hora siguiente.

# La función create_sequences hace lo siguiente (como una "ventana deslizante"):

# Primera Ventana (Primer ejemplo para entrenar):

# Entrada (X): Mira las 3 primeras horas (índices 0, 1, 2): [5, 6, 5]
# Salida (y) que debe predecir: El valor de la hora siguiente (índice 3): 7
# El modelo aprende: "Si ves [5, 6, 5], intenta predecir 7".
# ============================================================================

def create_sequences(features, target, window_size=3):
    """
    Crea secuencias de tamaño 'window_size' para la serie temporal,
    retornando X, y preparados para un modelo ConvLSTM2D.
    """
    X_list, Y_list = [], []
    for i in range(len(features) - window_size): #0-numvariables-1
        seq_x = features[i: i + window_size] 
        seq_y = target[i + window_size]
        X_list.append(seq_x)
        Y_list.append(seq_y)
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)

    # Redimensionar de (samples, timesteps, features) a (samples, timesteps, height=1, width=n_features, channels=1)
    n_samples, timesteps, n_features = X_arr.shape # de aqui saca las 3 variables
    X_arr = X_arr.reshape((n_samples, timesteps, 1, n_features, 1)) #le pone las 5 pa ConvLSTM2D 
    return X_arr, Y_arr


# Se utiliza window_size=3 para un intervalo más corto.
window_size = 3
X_seq, y_seq = create_sequences(X_all, y_all, window_size=window_size)

print("Forma de X_seq:", X_seq.shape)
print("Forma de y_seq:", y_seq.shape)

# ============================================================================
# 5. Separar en datos de entrenamiento y test (sin mezclar el orden temporal)
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.05, shuffle=False
)
print("X_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_test:", X_test.shape, "| y_test:", y_test.shape)

# ============================================================================
# 6. Construir el modelo ConvLSTM2D (más capas, dropout, etc.)
# ============================================================================
model = Sequential() #Crea el modelo 

# Primera capa ConvLSTM2D
model.add(ConvLSTM2D(
    filters=64, #64 filtros o patrones va a aprender
    kernel_size=(1, 3), #tamaño del filtro de convolucion
    activation='relu', #añade no linealidad pa detectar patrones complejos
    input_shape=(window_size, 1, len(feature_cols), 1), #forma de los datos de entrada
    padding='same', #mantiene el tamaño de la salida igual q el de la entrada aplicando relleno si es necesario
    return_sequences=True 
))
model.add(Dropout(0.2)) #durante el entrenamiento desactiva el 20% de las neuronas para evitar sobreaprendizajke y q no aprende demasiado de patrones especificos

# Segunda capa ConvLSTM2D
model.add(ConvLSTM2D(
    filters=32,
    kernel_size=(1, 3),
    activation='relu',
    padding='same',
    return_sequences=False
))
model.add(Dropout(0.2))

# Aplanar la salida (aplia, junta)
model.add(Flatten()) 

# Capa densa intermedia que encuentra patrones importantes conectada con 128 neuronaas
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

# Capa de salida para regresión. la unica neurona q da la prediccion final.
model.add(Dense(1, activation='linear')) #no se aplica trnasformacion a la salida

# Compilar el modelo
model.compile(
    optimizer=RMSprop(learning_rate=0.0005), #algoritmo de optimización. el learning rate controla que tan rapido aprende el modelo. un valor bajo es mas conservador
    loss='mse', #FUNCION DE PERDIDA. error cuadratico medio. Es la funcion que intenta minimizar el modelo durante el entrenamiento.
    metrics=['mae'] # Se usa para evaluar el error absoluto medio. 
)

model.summary()

# ============================================================================
# 7. Entrenar el modelo
# ============================================================================
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Graficar la evolución de la pérdida
plt.figure()
plt.plot(history.history['loss'], label='Pérdida Train')
plt.plot(history.history['val_loss'], label='Pérdida Val')
plt.title('Evolución de la pérdida (MSE)')
plt.legend()
plt.show()

# ============================================================================
# 8. Evaluación y predicción en Test
# ============================================================================
test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
print("MSE (Test, normalizado):", test_mse)
print("MAE (Test, normalizado):", test_mae)

# Predicciones
y_pred = model.predict(X_test)

# ============================================================================
# 9. Invertir el escalado a la escala original
# ============================================================================
y_pred_df = pd.DataFrame(y_pred, columns=[target_col])
y_test_df = pd.DataFrame(y_test, columns=[target_col])

temp_test = pd.DataFrame(
    np.zeros((len(y_test_df), len(df_model.columns))),
    columns=df_model.columns
)
temp_pred = temp_test.copy()

temp_test[target_col] = y_test_df[target_col]
temp_pred[target_col] = y_pred_df[target_col]

test_inverted = scaler.inverse_transform(temp_test)
pred_inverted = scaler.inverse_transform(temp_pred)

target_idx = df_model.columns.get_loc(target_col)
y_test_inverted = test_inverted[:, target_idx]
y_pred_inverted = pred_inverted[:, target_idx]

# ============================================================================
# 10. Métricas en escala original
# ============================================================================
mse_original = mean_squared_error(y_test_inverted, y_pred_inverted)
mae_original = mean_absolute_error(y_test_inverted, y_pred_inverted)
print("MSE (original):", mse_original)
print("MAE (original):", mae_original)

mape = np.mean(np.abs((y_test_inverted - y_pred_inverted) / y_test_inverted)) * 100
print("MAPE (original):", mape, "%")

# ============================================================================
# 11. Gráfica final
# ============================================================================
plt.figure(figsize=(10, 5))
plt.plot(y_test_inverted, label='Real')
plt.plot(y_pred_inverted, label='Predicción')
plt.title("Consumo eléctrico: Real vs Predicho")
plt.legend()
plt.show()


