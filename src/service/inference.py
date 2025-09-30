from typing import Optional, Deque
from collections import deque
import os
import json
import numpy as np
import tensorflow as tf

from src.utils.segmentation import sliding_window_segments
from src.utils.normalization import apply_zscore
from src.features.time_domain import extract_time_domain_features
from src.features.freq_domain import extract_frequency_features
from src.features.wavelet import extract_wavelet_features


class StreamingEMGInference:
    def __init__(
        self,
        model_path: str,
        norm_path: str,
        sampling_rate_hz: float = 2000.0,
        window_ms: float = 300.0,
        overlap: float = 0.5,
        num_channels: Optional[int] = None,
    ) -> None:
        self.model = tf.keras.models.load_model(model_path)
        with open(norm_path, "r") as f:
            stats = json.load(f)
        self.mean = np.asarray(stats["mean"], dtype=float)
        self.std = np.asarray(stats["std"], dtype=float)

        self.fs = sampling_rate_hz
        self.window_ms = window_ms
        self.overlap = overlap
        self.window_len = int(round(window_ms * 1e-3 * sampling_rate_hz))
        self.step = max(1, int(round(self.window_len * (1 - overlap))))
        self.num_channels = num_channels if num_channels is not None else len(self.mean)
        self.buffer: Deque[np.ndarray] = deque()
        self.buffer_len = 0

    def add_samples(self, samples: np.ndarray) -> None:
        """
        Append new sEMG samples shaped (num_new_samples, num_channels).
        """
        if samples.ndim != 2 or samples.shape[1] != self.num_channels:
            raise ValueError("samples must be shaped (n, num_channels)")
        self.buffer.append(samples)
        self.buffer_len += samples.shape[0]

        # Keep buffer from growing unbounded: retain last few windows worth
        max_keep = self.window_len * 4
        while self.buffer_len > max_keep:
            left = self.buffer[0]
            if self.buffer_len - left.shape[0] >= max_keep:
                self.buffer.popleft()
                self.buffer_len -= left.shape[0]
            else:
                break

    def _current_buffer_array(self) -> np.ndarray:
        if not self.buffer:
            return np.empty((0, self.num_channels))
        return np.concatenate(list(self.buffer), axis=0)

    def try_predict(self) -> Optional[int]:
        """
        If a new window is available, compute features and return predicted class.
        Returns None if not enough samples yet.
        """
        buf = self._current_buffer_array()
        if buf.shape[0] < self.window_len:
            return None

        # Take the latest complete window
        start = buf.shape[0] - self.window_len
        window = buf[start : start + self.window_len]

        # Normalize
        norm = apply_zscore(window, self.mean, self.std)

        # Extract features for each channel and concatenate
        feat_vec = []
        for ch in range(self.num_channels):
            x = norm[:, ch]
            td = extract_time_domain_features(x)
            fd = extract_frequency_features(x, fs=self.fs)
            wd = extract_wavelet_features(x)
            feat_vec.extend(list(td.values()) + list(fd.values()) + list(wd.values()))
        X = np.asarray(feat_vec, dtype=float)[None, :]

        y_prob = self.model.predict(X, verbose=0)
        y_pred = int(np.argmax(y_prob, axis=1)[0])
        return y_pred




