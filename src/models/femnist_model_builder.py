from tensorflow.keras import layers, Sequential
import tensorflow as tf
from .model_builder import ModelBuilder

class FemnistModelBuilder(ModelBuilder):
    """
    Constructor para modelos basados en el conjunto de datos FEMNIST.

    Esta clase implementa un modelo convolucional diseñado para trabajar
    específicamente con el conjunto de datos FEMNIST. El modelo tiene la 
    arquitectura siguiente:

    - Reshape: Convierte las entradas planas de 784 dimensiones a imágenes 28x28 con un canal.
    - Conv2D + MaxPooling2D: Tres bloques de convoluciones con activación ReLU seguidos por operaciones de agrupamiento máximo.
    - Flatten: Aplana las salidas convolucionales para ingresarlas a la capa densa.
    - Dense: Una capa completamente conectada con 128 neuronas y activación ReLU.
    - Output Layer: Capa de salida con 62 neuronas (para 62 clases en FEMNIST) y activación softmax.

    El modelo se compila con:
    - Optimizador: Adam
    - Función de pérdida: `sparse_categorical_crossentropy`
    - Métricas: Precisión (`accuracy`)

    Returns:
        tf.keras.Model: Modelo compilado listo para ser entrenado o evaluado.
    """

    
    def build(self) -> tf.keras.Model:
        model = Sequential([
            layers.Reshape((28, 28, 1), input_shape=(784,)),
            layers.Conv2D(8, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(24, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(62, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
