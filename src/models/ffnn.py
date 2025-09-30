from typing import Tuple
import tensorflow as tf


def build_ffnn(input_dim: int, num_classes: int, hidden_units: int = 128) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


