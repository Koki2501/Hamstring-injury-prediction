"""
Hardware Integration Example for MyPredict EMG Concept Drift Pipeline

This example shows how to integrate the trained models with real-time sEMG hardware
for hamstring injury prediction and activity classification.
"""

import numpy as np
import time
from src.service.inference import StreamingEMGInference


def simulate_hardware_data(sampling_rate=2000, num_channels=8, duration_seconds=10):
    """
    Simulate sEMG data from hardware (replace with your actual hardware interface).
    In real implementation, this would be replaced with your sEMG acquisition code.
    """
    num_samples = int(sampling_rate * duration_seconds)
    # Simulate realistic sEMG signals with different activities
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Simulate different muscle activities
    activities = []
    for i in range(0, num_samples, sampling_rate):  # Change activity every second
        if i // sampling_rate % 4 == 0:
            activities.extend(['walking'] * sampling_rate)
        elif i // sampling_rate % 4 == 1:
            activities.extend(['running'] * sampling_rate)
        elif i // sampling_rate % 4 == 2:
            activities.extend(['ramp_ascent'] * sampling_rate)
        else:
            activities.extend(['ramp_descent'] * sampling_rate)
    
    # Generate synthetic sEMG data
    emg_data = np.zeros((num_samples, num_channels))
    for ch in range(num_channels):
        for i, activity in enumerate(activities[:num_samples]):
            if activity == 'walking':
                # Lower amplitude, regular pattern
                emg_data[i, ch] = 0.1 * np.sin(2 * np.pi * 2 * t[i]) + 0.05 * np.random.randn()
            elif activity == 'running':
                # Higher amplitude, faster pattern
                emg_data[i, ch] = 0.3 * np.sin(2 * np.pi * 4 * t[i]) + 0.1 * np.random.randn()
            elif activity == 'ramp_ascent':
                # Moderate amplitude, steady increase
                emg_data[i, ch] = 0.2 * np.sin(2 * np.pi * 1.5 * t[i]) + 0.08 * np.random.randn()
            elif activity == 'ramp_descent':
                # High amplitude, hamstring-intensive
                emg_data[i, ch] = 0.4 * np.sin(2 * np.pi * 3 * t[i]) + 0.12 * np.random.randn()
    
    return emg_data, activities[:num_samples]


def real_time_inference_example():
    """
    Example of real-time inference with streaming sEMG data.
    """
    print("Initializing EMG inference service...")
    
    # Initialize the inference service
    infer = StreamingEMGInference(
        model_path="artifacts/ffnn_model.keras",
        norm_path="artifacts/norm_stats.json",
        sampling_rate_hz=2000.0,
        window_ms=300.0,
        overlap=0.5,
        num_channels=8  # Adjust based on your hardware
    )
    
    print("Service initialized. Starting real-time inference...")
    print("Press Ctrl+C to stop...")
    
    # Activity labels (adjust based on your dataset)
    activity_labels = {
        0: "walking",
        1: "running", 
        2: "ramp_ascent",
        3: "ramp_descent",
        4: "standing",
        5: "sitting"
    }
    
    try:
        # Simulate real-time data acquisition
        chunk_size = 100  # Process 100 samples at a time
        emg_data, true_activities = simulate_hardware_data(duration_seconds=30)
        
        for i in range(0, len(emg_data), chunk_size):
            # Get new data chunk
            chunk = emg_data[i:i+chunk_size]
            if len(chunk) < chunk_size:
                break
                
            # Add samples to inference buffer
            infer.add_samples(chunk)
            
            # Try to get prediction
            prediction = infer.try_predict()
            if prediction is not None:
                activity = activity_labels.get(prediction, f"unknown_{prediction}")
                print(f"Predicted activity: {activity} (class {prediction})")
                
                # Check for hamstring-intensive activities
                if activity in ["ramp_descent", "running"]:
                    print("  ⚠️  HAMSTRING-INTENSIVE ACTIVITY DETECTED!")
                    print("  💡 Consider injury risk assessment")
            
            # Simulate real-time processing delay
            time.sleep(chunk_size / 2000.0)  # Match sampling rate
            
    except KeyboardInterrupt:
        print("\nStopping inference...")
    except Exception as e:
        print(f"Error during inference: {e}")
    
    print("Inference stopped.")


def batch_inference_example():
    """
    Example of batch inference on pre-recorded data.
    """
    print("Running batch inference example...")
    
    # Load pre-recorded data (replace with your data loading)
    emg_data, true_activities = simulate_hardware_data(duration_seconds=60)
    
    # Initialize inference service
    infer = StreamingEMGInference(
        model_path="artifacts/ffnn_model.keras",
        norm_path="artifacts/norm_stats.json",
        sampling_rate_hz=2000.0,
        window_ms=300.0,
        overlap=0.5,
        num_channels=8
    )
    
    # Process data in chunks
    chunk_size = 200
    predictions = []
    
    for i in range(0, len(emg_data), chunk_size):
        chunk = emg_data[i:i+chunk_size]
        if len(chunk) < chunk_size:
            break
            
        infer.add_samples(chunk)
        prediction = infer.try_predict()
        if prediction is not None:
            predictions.append(prediction)
    
    # Analyze results
    if predictions:
        unique, counts = np.unique(predictions, return_counts=True)
        print(f"Predicted activities: {dict(zip(unique, counts))}")
        
        # Count hamstring-intensive activities
        hamstring_activities = [p for p in predictions if p in [1, 3]]  # running, ramp_descent
        print(f"Hamstring-intensive activities: {len(hamstring_activities)}/{len(predictions)} ({100*len(hamstring_activities)/len(predictions):.1f}%)")


if __name__ == "__main__":
    print("MyPredict EMG Hardware Integration Example")
    print("=" * 50)
    
    # Check if artifacts exist
    import os
    if not os.path.exists("artifacts/ffnn_model.keras"):
        print("❌ Model artifacts not found!")
        print("Please run 'python experiments/run_experiment.py' first to train models.")
        exit(1)
    
    print("✅ Model artifacts found.")
    print("\nChoose an example:")
    print("1. Real-time inference simulation")
    print("2. Batch inference example")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        real_time_inference_example()
    elif choice == "2":
        batch_inference_example()
    else:
        print("Invalid choice. Running real-time example...")
        real_time_inference_example()
