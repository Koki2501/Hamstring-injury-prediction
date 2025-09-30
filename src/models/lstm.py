from typing import Tuple
import tensorflow as tf


def build_lstm(input_timesteps: int, input_channels: int, num_classes: int, units: int = 64) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_timesteps, input_channels))
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.LSTM(units)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


