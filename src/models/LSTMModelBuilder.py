import tensorflow as tf
from tensorflow.keras.models import Sequential
# CAMBIO: Importar la capa LSTM en lugar de ConvLSTM2D
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .model_builder import ModelBuilder # Asumo que tienes una clase base ModelBuilder

class LSTMModelBuilder(ModelBuilder):
    def __init__(self, input_shape, 
                 lstm_units_1=32, lstm_units_2=16,
                 dense_units_1=32, dense_units_2=16, dropout_rate=0.2, learning_rate=0.0001,
                 early_stopping_patience=3, early_stopping_monitor='val_loss',
                 early_stopping_restore_best_weights=True):
        
        self.input_shape = input_shape
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units_1 = dense_units_1
        self.dense_units_2 = dense_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_restore_best_weights = early_stopping_restore_best_weights

    def build(self) -> tf.keras.Model:
        """
        Builds and compiles the LSTM model.
        """
        model = Sequential([
            Input(shape=self.input_shape),
            

            LSTM(self.lstm_units_1,
                 return_sequences=True,  
                 recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            LSTM(self.lstm_units_2,
                 return_sequences=False, 
                 recurrent_dropout=0.1),
            BatchNormalization(),
            Dropout(self.dropout_rate),
        
            
            Dense(self.dense_units_1, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(self.dropout_rate),
            
            Dense(self.dense_units_2, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(self.dropout_rate),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def get_callbacks(self):
        callbacks = []
        early_stopping = EarlyStopping(
            monitor=self.early_stopping_monitor,
            patience=self.early_stopping_patience,
            restore_best_weights=self.early_stopping_restore_best_weights,
            verbose=1
        )
        callbacks.append(early_stopping)
        return callbacks