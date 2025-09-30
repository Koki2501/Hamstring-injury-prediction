from typing import Tuple
import numpy as np


def zscore_normalize_per_channel(
    data: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize per channel for input shaped (num_samples, num_channels).

    Returns normalized data and (mean, std) for inverse transform or future sessions.
    """
    if data.ndim != 2:
        raise ValueError("data must be shaped (num_samples, num_channels)")

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std_safe = np.where(std < eps, 1.0, std)
    normalized = (data - mean) / std_safe
    return normalized, mean, std_safe


def apply_zscore(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("data must be shaped (num_samples, num_channels)")
    return (data - mean) / np.where(std == 0, 1.0, std)


