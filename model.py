# model.py (Corrected: LSTM on 3D CNN Output)
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    LSTM,
    GlobalAveragePooling1D,  # Add for post-LSTM aggregation
    BatchNormalization,
    MaxPooling1D,
    Bidirectional,
)
from keras.regularizers import l2

def build_cnn_lstm(input_shape, n_classes, lstm_units=128, dropout_rate=0.5, bidirectional=True, l2_reg=0.001):
    """
    input_shape: (seq_len, n_features)  # 1D sequences
    """
    seq_len, n_feat = input_shape
    inp = Input(shape=input_shape, name='input_sequences')
    
    # 1D CNN layers for spatial feature extraction
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Bidirectional LSTM on 3D CNN output (temporal fusion)
    if bidirectional:
        x = Bidirectional(LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg)))(x)
    else:
        x = LSTM(lstm_units, return_sequences=False, kernel_regularizer=l2(l2_reg))(x)
    
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(n_classes, activation='softmax', name='out')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model