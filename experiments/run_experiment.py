import os
from typing import List, Dict
import numpy as np
import json

from src.data.loader import load_hdf5_session
from src.utils.segmentation import sliding_window_segments
from src.utils.normalization import zscore_normalize_per_channel
from src.features.time_domain import extract_time_domain_features
from src.features.freq_domain import extract_frequency_features
from src.features.wavelet import extract_wavelet_features
from src.models.ffnn import build_ffnn
from src.models.lstm import build_lstm
from src.models.ensemble import EnsembleModel
from src.drift.detector import detect_drift_kmeans_silhouette
from src.drift.retrain import partial_retrain_classifier
from src.utils.metrics import classification_metrics, compute_confusion
from src.utils.plotting import plot_confusion, plot_metric_curve


def extract_features_for_windows(windows: np.ndarray, fs: float) -> np.ndarray:
    num_windows, win_len, num_channels = windows.shape
    feats: List[Dict[str, float]] = []
    feat_names: List[str] = []
    
    for w in windows:
        feat_vec = []
        for ch in range(num_channels):
            x = w[:, ch]
            td = extract_time_domain_features(x)
            fd = extract_frequency_features(x, fs)
            wd = extract_wavelet_features(x)
            if not feat_names:
                feat_names = [f"ch{ch}_" + k for k in list(td.keys()) + list(fd.keys()) + list(wd.keys())]
            feat_vec.extend(list(td.values()) + list(fd.values()) + list(wd.values()))
        feats.append(feat_vec)
    return np.asarray(feats, dtype=float)


def majority_label_per_window(labels: np.ndarray, win_len: int, step: int) -> np.ndarray:
    
    starts = np.arange(0, len(labels) - win_len + 1, step)
    y = []
    for s in starts:
        seg = labels[s : s + win_len]
        vals, counts = np.unique(seg, return_counts=True)
        y.append(vals[np.argmax(counts)])
    return np.asarray(y)


def run_pipeline(day_files: List[str], emg_fs: float = 2000.0, imu_fs: float = 240.0):
    
    day_data = []
    for p in day_files:
        d = load_hdf5_session(p)
        day_data.append(d)

    
    day_features = []
    day_labels = []
    for d in day_data:
        emg = d.get("emg")
        labels = d.get("labels")
        if emg is None or labels is None:
            raise ValueError("HDF5 missing required 'emg' or 'labels' datasets for a day.")

        emg_norm, mean, std = zscore_normalize_per_channel(emg)
        windows, step = sliding_window_segments(emg_norm, window_ms=300.0, overlap=0.5, sampling_rate_hz=emg_fs)
        X = extract_features_for_windows(windows, fs=emg_fs)
        y = majority_label_per_window(labels, win_len=windows.shape[1], step=step)
        day_features.append(X)
        day_labels.append(y)

    X_train, y_train = day_features[0], day_labels[0]

    num_classes = int(np.max(np.concatenate(day_labels)) + 1)
    
    ffnn = build_ffnn(input_dim=X_train.shape[1], num_classes=num_classes)
    hist_ff = ffnn.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
    plot_metric_curve(hist_ff, "loss", "FFNN Loss", os.path.join("outputs", "ffnn_loss.png"))
    plot_metric_curve(hist_ff, "accuracy", "FFNN Accuracy", os.path.join("outputs", "ffnn_acc.png"))

    # Train LSTM (reshape for sequence input)
    X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # (samples, timesteps, features)
    lstm = build_lstm(input_timesteps=1, input_channels=X_train.shape[1], num_classes=num_classes)
    hist_lstm = lstm.fit(X_train_seq, y_train, epochs=10, batch_size=64, verbose=0)
    plot_metric_curve(hist_lstm, "loss", "LSTM Loss", os.path.join("outputs", "lstm_loss.png"))
    plot_metric_curve(hist_lstm, "accuracy", "LSTM Accuracy", os.path.join("outputs", "lstm_acc.png"))

    # Create ensemble
    ensemble = EnsembleModel([ffnn, lstm], weights=[0.6, 0.4])

    # Save artifacts for hardware inference
    os.makedirs("artifacts", exist_ok=True)
    ffnn.save(os.path.join("artifacts", "ffnn_model.keras"))
    lstm.save(os.path.join("artifacts", "lstm_model.keras"))
    
    # Save normalization from Day 1 EMG
    # Recompute mean/std on Day 1 raw EMG for service; we cached earlier from zscore
    emg_day1 = day_data[0]["emg"]
    mean = emg_day1.mean(axis=0).tolist()
    std = (emg_day1.std(axis=0) + 1e-8).tolist()
    with open(os.path.join("artifacts", "norm_stats.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)

    logs = []
    for i in range(1, len(day_features)):
        X_eval, y_eval = day_features[i], day_labels[i]
        X_eval_seq = X_eval.reshape(X_eval.shape[0], 1, X_eval.shape[1])
        
        # Test all models
        y_pred_ffnn = np.argmax(ffnn.predict(X_eval, verbose=0), axis=1)
        y_pred_lstm = np.argmax(lstm.predict(X_eval_seq, verbose=0), axis=1)
        y_pred_ensemble = ensemble.predict_classes(X_eval)
        
        # Plot confusion matrices
        plot_confusion(y_eval, y_pred_ffnn, f"Day {i+1} FFNN Confusion", os.path.join("outputs", f"confusion_day{i+1}_ffnn.png"))
        plot_confusion(y_eval, y_pred_lstm, f"Day {i+1} LSTM Confusion", os.path.join("outputs", f"confusion_day{i+1}_lstm.png"))
        plot_confusion(y_eval, y_pred_ensemble, f"Day {i+1} Ensemble Confusion", os.path.join("outputs", f"confusion_day{i+1}_ensemble.png"))

        # Drift detection and adaptation
        drift_info = detect_drift_kmeans_silhouette(day_features[0], X_eval, k=2, silhouette_drop_threshold=0.05)
        
        pre_metrics = {
            "ffnn": classification_metrics(y_eval, y_pred_ffnn),
            "lstm": classification_metrics(y_eval, y_pred_lstm),
            "ensemble": classification_metrics(y_eval, y_pred_ensemble)
        }
        
        if drift_info["drift"]:
            print(f"Drift detected on Day {i+1}! Retraining models...")
            partial_retrain_classifier(ffnn, X_eval, y_eval, epochs=3)
            partial_retrain_classifier(lstm, X_eval_seq, y_eval, epochs=3)
            
            # Re-evaluate after retraining
            y_pred_ffnn_after = np.argmax(ffnn.predict(X_eval, verbose=0), axis=1)
            y_pred_lstm_after = np.argmax(lstm.predict(X_eval_seq, verbose=0), axis=1)
            y_pred_ensemble_after = ensemble.predict_classes(X_eval)
            
            post_metrics = {
                "ffnn": classification_metrics(y_eval, y_pred_ffnn_after),
                "lstm": classification_metrics(y_eval, y_pred_lstm_after),
                "ensemble": classification_metrics(y_eval, y_pred_ensemble_after)
            }
        else:
            post_metrics = pre_metrics

        logs.append({
            "day": i + 1,
            "pre": pre_metrics,
            "post": post_metrics,
            "drift": drift_info,
        })

    return logs


if __name__ == "__main__":
    # Replace with actual paths: Day1..Day4 HDF5 files
    day_files = [
        os.path.join("data", "Day1.h5"),
        os.path.join("data", "Day2.h5"),
        os.path.join("data", "Day3.h5"),
        os.path.join("data", "Day4.h5"),
    ]
    results = run_pipeline(day_files)
    for r in results:
        print(r)


