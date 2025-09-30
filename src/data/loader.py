from typing import Dict, Optional, List
import h5py
import numpy as np


def _find_first_matching_key(keys: List[str], candidates: List[str]) -> Optional[str]:
    lower = {k.lower(): k for k in keys}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # substring search fallback
    for k in keys:
        for c in candidates:
            if c.lower() in k.lower():
                return k
    return None


def load_hdf5_session(
    path: str,
    emg_key: Optional[str] = None,
    imu_key: Optional[str] = None,
    label_key: Optional[str] = None,
    subject_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load a session from MyPredict/MyLeg HDF5.

    Expected structure (generic, may need to adjust keys after inspecting your files):
      - EMG: (num_samples, num_emg_channels)
      - IMU: (num_samples, num_imu_channels)
      - labels: (num_samples,) integer activity labels
      - optionally per-subject grouping
    """
    with h5py.File(path, "r") as f:
        grp = f[subject_key] if subject_key else f
        grp_keys = list(grp.keys())

        # Auto-detect keys if not provided
        if emg_key is None:
            emg_key = _find_first_matching_key(grp_keys, ["EMG", "sEMG", "emg", "SEMG"]) or "EMG"
        if imu_key is None:
            imu_key = _find_first_matching_key(grp_keys, ["IMU", "imu", "ACC", "GYRO"]) or "IMU"
        if label_key is None:
            label_key = _find_first_matching_key(grp_keys, ["labels", "y", "activity", "Activities", "Label"]) or "labels"

        emg = np.asarray(grp[emg_key]) if emg_key in grp else None
        imu = np.asarray(grp[imu_key]) if imu_key in grp else None
        labels = np.asarray(grp[label_key]) if label_key in grp else None

    result: Dict[str, np.ndarray] = {}
    if emg is not None:
        result["emg"] = emg
    if imu is not None:
        result["imu"] = imu
    if labels is not None:
        result["labels"] = labels
    return result


