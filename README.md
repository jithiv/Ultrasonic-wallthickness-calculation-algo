# Ultrasonic Wall Thickness Calculation Engine (MBE)

A modular Python framework for non-destructive testing (NDT) pipe wall thickness measurement using **Model-Based Estimation (MBE)** and **Levenberg-Marquardt (LM) Optimization**.

## 📌 Overview
High-precision ultrasonic thickness gauging often suffers from electronic noise and pulse distortion. This project implements a zero-phase filtering and Gaussian wavelet fitting engine to extract microsecond Time-of-Flight (TOF) with sub-sample accuracy.

## 🛠️ Key Components
- **`src/config.py`**: Central configuration dataclass for sampling rate (100 MHz) and steel sound velocity (5900 m/s).
- **`src/filter.py`**: Zero-phase Butterworth bandpass filter.
- **`src/model.py`**: Gaussian wavelet modeling & synthetic A-scan signal generator.
- **`src/optimizer.py`**: Levenberg-Marquardt non-linear least squares optimization engine.

## 📚 Methodology & Citation
This algorithm is based on academic research in ultrasonic signal processing and Model-Based Estimation (MBE). 

> *Reference Paper details will be updated here.*

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
