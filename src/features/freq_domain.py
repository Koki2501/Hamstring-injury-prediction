from typing import Dict
import numpy as np


def mean_frequency(power_spectrum: np.ndarray, freqs: np.ndarray) -> float:
    psd = power_spectrum
    total_power = np.sum(psd) + 1e-12
    return float(np.sum(freqs * psd) / total_power)


def median_frequency(power_spectrum: np.ndarray, freqs: np.ndarray) -> float:
    cumsum = np.cumsum(power_spectrum)
    half = cumsum[-1] / 2.0
    idx = np.searchsorted(cumsum, half)
    return float(freqs[min(idx, len(freqs) - 1)])


def extract_frequency_features(window: np.ndarray, fs: float) -> Dict[str, float]:
    x = window - np.mean(window)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.fft.rfft(x)
    psd = (np.abs(fft_vals) ** 2) / n
    return {
        "MNF": mean_frequency(psd, freqs),
        "MDF": median_frequency(psd, freqs),
    }


