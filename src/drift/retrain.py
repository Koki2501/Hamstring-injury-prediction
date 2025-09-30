from typing import Optional, Tuple
import numpy as np
import tensorflow as tf


def partial_retrain_classifier(
    model: tf.keras.Model,
    x_new: np.ndarray,
    y_new: np.ndarray,
    epochs: int = 3,
    batch_size: int = 64,
    validation_split: float = 0.0,
) -> tf.keras.callbacks.History:
    """
    Perform a short fine-tuning on new data after drift detection.
    """
    history = model.fit(x_new, y_new, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    return history


