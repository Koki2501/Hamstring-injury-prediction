from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


