# Lower Limb Activity Prediction with sEMG under Concept Drift (MyPredict Dataset)

This project investigates **multi-day activity prediction for lower limb motion** using surface electromyography (sEMG) and kinematic data from the [MyPredict / MyLeg Database](https://data.4tu.nl/datasets/01d30db7-95a8-4c39-afb9-4eb1a2f27539). The primary goal is to address **concept drift** challenges in EMG-based activity recognition across different days.

## 📦 Dataset

- **Dataset Name:** MyLeg / MyPredict Database
- **Provider:** Roessingh Research & Development
- **Subjects:** 55 able-bodied individuals, over 85 measurement sessions
- **Modalities:** sEMG (bipolar & multi-array), IMU-based kinematics
- **Sampling Rate:** sEMG up to 2000Hz, IMUs at 240Hz
- **Activities:** Sitting, standing, walking, stairs, ramp, uneven terrain
- **Link:** [MyPredict Dataset](https://data.4tu.nl/datasets/01d30db7-95a8-4c39-afb9-4eb1a2f27539)

## 🎯 Problem Statement

In long-term activity prediction systems based on EMG signals, **concept drift** — changes in data distribution over time — can significantly reduce model accuracy. This project focuses on:

- **Detecting concept drift** across different days
- **Adapting deep learning models** (NN & RNN) to handle drift
- **Evaluating performance** on ramp ascending/descending tasks using MyPredict 3 data

## 🧠 Methodology

### 🔍 Features Used

- **Kinematics:** Acceleration, angular velocity (thigh, shank, foot), joint angles (knee, ankle)
- **EMG Features:**  
  - Time-domain: MAV, RMS, VAR, WL, SSC, ZC, AR, CC  
  - Frequency-domain: MNF, MDF  
  - Time-frequency: WTWL, WTVAR, WTMAV

### 🔧 Modeling

- **Models:** Neural Network (1 hidden layer), Simple RNN
- **Drift Detection:** K-Means clustering + silhouette score monitoring
- **Adaptation Strategy:** Online retraining based on drift detection
- **Ensemble:** Averaged predictions from NN & RNN

### 🔁 Workflow

1. Split MyPredict 3 trials by day (Day 1–4)
2. Extract 300ms windows per activity
3. Train base model on Day 1
4. Use K-means to detect concept drift
5. Retrain incrementally with new data
6. Compare performance with and without adaptation

## 📊 Results

- Concept drift (both **abrupt** and **gradual**) was detected across days.
- Adaptation strategy significantly improved model **stability and accuracy**.
- Ensemble model yielded **lowest error rates**, particularly in **ramp descent**.

## 📚 References

- Schulte et al., *Scientific Data*, 2023  
- Schulte et al., *Sensors*, 2022  
- Xiang et al., *Applied Sciences*, 2023 (on Concept Drift Adaptation)

## 🛠 Requirements

- Python 3.x
- Jupyter Notebook
- NumPy, pandas, SciPy
- scikit-learn
- TensorFlow or PyTorch
- matplotlib, seaborn
