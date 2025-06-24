# Pega este código completo en src/models/mlp_model_builder.py

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.constraints import max_norm

from .model_builder import ModelBuilder

class MLPModelBuilder(ModelBuilder):
    """
    Constructor para modelos Multilayer Perceptron (MLP) que ahora acepta
    la forma de entrada de los datos de secuencia y los aplana.
    """

    def build(self) -> tf.keras.Model:
        """
        Construye y compila el modelo MLP.
        
        La primera capa Flatten se encarga de convertir los datos de entrada
        con forma de secuencia (ej: [12, 1, 21, 1]) a un vector plano que
        las capas Dense puedan procesar.
        """
        # La forma de entrada es la que genera tu DataLoader para el ConvLSTM
        input_shape = (12, 1, 21, 1)
        
        model = Sequential([
            # 1. APLANA LA ENTRADA: Convierte los datos de secuencia en un vector.
            layers.Flatten(input_shape=input_shape),
            
            # 2. El resto de tu modelo MLP original.
            layers.Dense(350, activation='relu', kernel_constraint=max_norm(4)),
            layers.Dropout(0.0),
            layers.Dense(50, activation='relu'),
            layers.Dense(6, activation='sigmoid')
        ])
        
        # 3. Compila el modelo.
        model.compile(
            optimizer=Nadam(learning_rate=0.005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True
        )
        
        return model