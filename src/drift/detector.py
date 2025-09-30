from typing import Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def detect_drift_kmeans_silhouette(
    baseline_features: np.ndarray,
    current_features: np.ndarray,
    k: int = 2,
    silhouette_drop_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Compare clustering quality between baseline and current features using silhouette.
    If silhouette score drops by more than threshold, flag drift.
    """
    km_base = KMeans(n_clusters=k, n_init=10, random_state=42)
    base_labels = km_base.fit_predict(baseline_features)
    base_sil = silhouette_score(baseline_features, base_labels) if baseline_features.shape[0] >= k + 1 else 0.0

    km_curr = KMeans(n_clusters=k, n_init=10, random_state=42)
    curr_labels = km_curr.fit_predict(current_features)
    curr_sil = silhouette_score(current_features, curr_labels) if current_features.shape[0] >= k + 1 else 0.0

    drop = float(base_sil - curr_sil)
    drift = float(drop > silhouette_drop_threshold)
    return {"base_silhouette": float(base_sil), "current_silhouette": float(curr_sil), "drop": drop, "drift": drift}


