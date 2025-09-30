from typing import Dict
import numpy as np


def mean_absolute_value(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def root_mean_square(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def variance(x: np.ndarray) -> float:
    return float(np.var(x))


def zero_crossings(x: np.ndarray, threshold: float = 0.0) -> int:
    s = np.sign(x - threshold)
    return int(np.sum(np.abs(np.diff(s)) > 0))


def slope_sign_changes(x: np.ndarray, threshold: float = 0.0) -> int:
    diff1 = np.diff(x)
    sign_changes = np.logical_and(diff1[:-1] * diff1[1:] < 0, np.abs(diff1[:-1] - diff1[1:]) > threshold)
    return int(np.sum(sign_changes))


def waveform_length(x: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(x))))


def extract_time_domain_features(window: np.ndarray) -> Dict[str, float]:
    """
    Compute MAV, RMS, VAR, ZC, SSC, WL for a 1D window.
    """
    return {
        "MAV": mean_absolute_value(window),
        "RMS": root_mean_square(window),
        "VAR": variance(window),
        "ZC": float(zero_crossings(window)),
        "SSC": float(slope_sign_changes(window)),
        "WL": waveform_length(window),
    }


