
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from .model_builder import ModelBuilder

class TransformerModelBuilder(ModelBuilder):
    """
    Constructor para un modelo basado en Transformer para regresión de series temporales.
    Equivalente al modelo no modular proporcionado (1 capa Transformer + densa).
    """
    def __init__(self, input_shape,
                 head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=1,
                 dense_units_1=64, dense_units_2=32,
                 dropout_rate=0.1, learning_rate=1e-4,
                 early_stopping_patience=10, early_stopping_monitor='val_loss',
                 early_stopping_restore_best_weights=True):

        self.input_shape = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_restore_best_weights = early_stopping_restore_best_weights

    def _transformer_encoder(self, inputs):
        """Crea un bloque codificador del Transformer."""
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.head_size,
            dropout=self.dropout_rate
        )(x, x)
        x = layers.Dropout(self.dropout_rate)(x)
        res = x + inputs

        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Dense(self.input_shape[-1], activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x + res

    def build(self) -> tf.keras.Model:
        """Construye y compila el modelo Transformer."""
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(self.dense_units_1, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.dense_units_2, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(1, activation='linear')(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def get_callbacks(self):
        """Devuelve los callbacks configurados, incluyendo early stopping."""
        return [
            EarlyStopping(
                monitor=self.early_stopping_monitor,
                patience=self.early_stopping_patience,
                restore_best_weights=self.early_stopping_restore_best_weights,
                verbose=1
            )
        ]
