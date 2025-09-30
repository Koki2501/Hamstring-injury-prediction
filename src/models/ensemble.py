from typing import List, Tuple
import numpy as np
import tensorflow as tf


class EnsembleModel:
    def __init__(self, models: List[tf.keras.Model], weights: List[float] = None):
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        pred_probs = self.predict(X)
        return np.argmax(pred_probs, axis=1)


