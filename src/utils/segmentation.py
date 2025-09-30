from typing import Tuple
import numpy as np


def sliding_window_segments(
    data: np.ndarray,
    window_ms: float,
    overlap: float,
    sampling_rate_hz: float,
) -> Tuple[np.ndarray, int]:
    """
    Segment multichannel time-series with fixed-length sliding windows.

    Args:
        data: Array shaped (num_samples, num_channels).
        window_ms: Window length in milliseconds.
        overlap: Fractional overlap in [0,1). E.g., 0.5 for 50%.
        sampling_rate_hz: Sampling frequency of the signal.

    Returns:
        windows: Array shaped (num_windows, window_len, num_channels).
        step: Hop length in samples.
    """
    if data.ndim != 2:
        raise ValueError("data must be shaped (num_samples, num_channels)")
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0,1)")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    window_len = int(round(window_ms * 1e-3 * sampling_rate_hz))
    if window_len <= 1:
        raise ValueError("window length too small given sampling rate")

    step = max(1, int(round(window_len * (1 - overlap))))
    num_samples, num_channels = data.shape
    if num_samples < window_len:
        return np.empty((0, window_len, num_channels), dtype=data.dtype), step

    starts = np.arange(0, num_samples - window_len + 1, step)
    windows = np.stack([data[s : s + window_len] for s in starts], axis=0)
    return windows, step


