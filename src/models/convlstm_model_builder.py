import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from .model_builder import ModelBuilder

class ConvLSTMModelBuilder(ModelBuilder):
    def __init__(self, input_shape, convlstm_filters_1=64, convlstm_filters_2=32,
                 dense_units_1=64, dense_units_2=32, dropout_rate=0.1, learning_rate=0.0003):
        self.input_shape = input_shape
        self.convlstm_filters_1 = convlstm_filters_1
        self.convlstm_filters_2 = convlstm_filters_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self) -> tf.keras.Model:
        model = Sequential([
            Input(shape=self.input_shape),
            ConvLSTM2D(self.convlstm_filters_1, (1, 3), activation='relu', 
                    padding='same', return_sequences=True),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            ConvLSTM2D(self.convlstm_filters_2, (1, 3), activation='relu', 
                    padding='same', return_sequences=False),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(self.dense_units_1, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(self.dense_units_2, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
