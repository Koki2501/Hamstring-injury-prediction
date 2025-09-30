from typing import Dict
import numpy as np
import pywt


def extract_wavelet_features(window: np.ndarray, wavelet: str = "db4", level: int = 3) -> Dict[str, float]:
    x = window - np.mean(window)
    coeffs = pywt.wavedec(x, wavelet, level=level)
    features: Dict[str, float] = {}
    for i, c in enumerate(coeffs):
        features[f"WL_energy_L{i}"] = float(np.sum(c ** 2))
        features[f"WL_std_L{i}"] = float(np.std(c))
    return features


