from typing import Dict, List
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_curve(history, metric: str, title: str, out_path: str) -> None:
    plt.figure()
    plt.plot(history.history.get(metric, []), label=metric)
    if f"val_{metric}" in history.history:
        plt.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


