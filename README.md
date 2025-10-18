# 🫀 ECG-DKS-SelfAttention  
**Ultra Low Power SHAKTI RISC-V Based Lightweight Edge AI Processor for IoT-enabled Healthcare Applications**  
> Developed under the *Chips to Startup (C2S) Internship Programme*, funded by **MeitY, Government of India**

---

![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange?logo=tensorflow)
![TFLite](https://img.shields.io/badge/Quantized-TensorFlow%20Lite-blue?logo=tensorflow)
![Python](https://img.shields.io/badge/Language-Python%203.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Hardware](https://img.shields.io/badge/Target-RISC--V%20%7C%20NPU%20%7C%20FPGA-red?logo=riscv)
![Optimization](https://img.shields.io/badge/Optimization-INT8%20%7C%20Pruning%20%7C%20Clustering-yellow)
![hls4ml](https://img.shields.io/badge/hls4ml-Compatible-lightgrey)
![C2S](https://img.shields.io/badge/Program-C2S%20%7C%20MeitY-blue)
![EdgeAI](https://img.shields.io/badge/Focus-Edge%20AI%20Healthcare-purple)

---

### 📑 Overview

This repository presents an **ECG classification model** integrating **Dynamic Kernel Switching (DKS)** and **Self-Attention** mechanisms, optimized for **Neural Processing Units (NPUs)** and **RISC-V Edge AI SoCs**.  
The project focuses on **ultra-low-power, quantized CNN deployment** for **IoT-enabled healthcare monitoring systems**.

---


## 🔬 Project Overview

This repository presents a **CNN-based ECG Classification System** enhanced with **Dynamic Kernel Switching (DKS)** and **Self-Attention** mechanisms.  
The model classifies ECG signals from the **MIT-BIH Arrhythmia Database** and is optimized for **Edge AI hardware deployment** via **INT8 quantization** using TensorFlow Lite.

> 🧠 Developed under the **Chips to Startup (C2S) Internship Programme** as part of the project:
> **“Ultra Low Power SHAKTI RISC-V Based Lightweight Edge AI Processor for IoT-enabled Healthcare Applications”**,  
> funded by the **Ministry of Electronics and IT (MeitY), Govt. of India**.

---

## 🧩 Architecture
ECG Signal → Preprocessing → Segmentation → CNN (with DKS + Self-Attention) → Classification → Quantized TFLite Deployment

## 📈 Model Performance and Training Summary

### 🧠 Phase 1: Model Development (FP32)

The model was trained on **99,443 heartbeats** extracted from the MIT-BIH Arrhythmia Database using **R-peak detection (ANNOTATION method)**.

#### Data Summary
| Metric | Value |
|---------|-------|
| Total Records Processed | 43 |
| Training Samples | 79,554 |
| Testing Samples | 19,889 |
| Input Shape | (188, 1) |
| Output Classes | 5 (F, N, Q, S, V) |

#### Class Distribution
| Label | Meaning | Count | Percentage |
|--------|----------|--------|-------------|
| F | Fusion beats | 798 | 0.8% |
| N | Normal beats | 87,078 | 87.6% |
| Q | Unclassifiable | 1,802 | 1.8% |
| S | Supraventricular | 2,750 | 2.8% |
| V | Ventricular | 7,015 | 7.0% |

---

### 🧩 Model Architecture Summary
Input Layer → DKS Block (3,5,7 kernels, 64 filters)
→ Self-Attention (128 units)
→ Conv1D (128 filters, kernel=5) + MaxPooling
→ Conv1D (256 filters, kernel=5) + MaxPooling
→ GlobalAveragePooling + Dropout
→ Dense (256) + Dropout
→ Output Layer (Softmax)


| Layer | Output Shape | Parameters |
|--------|---------------|-------------|
| Dynamic Kernel Switching | (188, 64) | 1,743 |
| Self-Attention | (188, 128) | 16,769 |
| Conv1D-1 | (188, 128) | 82,048 |
| Conv1D-2 | (94, 256) | 164,096 |
| Dense | (256) | 65,792 |
| Output | (5) | 1,285 |
| **Total Parameters** |  | **331,733 (≈1.27 MB)** |

---

### 🏋️ Training Summary

The model was trained for **30 epochs** with **class weights** to handle imbalance.  
Below is a snapshot of the progression:

| Epoch | Train Accuracy | Val Accuracy | Val Loss |
|-------:|----------------:|--------------:|-----------:|
| 1 | 57.27% | 74.44% | 0.926 |
| 6 | 87.19% | 93.93% | 0.265 |
| 12 | 90.80% | 95.59% | 0.169 |
| 18 | 92.04% | 96.62% | 0.126 |
| 26 | 94.81% | 96.37% | 0.111 |
| **30 (Final)** | **95.19%** | **96.51%** | **0.133** |

🧾 **Test Accuracy:** 96.51 %  
📦 **Saved Model:** `ecg_dks_attention_model_annotation_r30.h5`

---

### ⚙️ Key Block Contributions

| Block | Purpose | Conceptual Accuracy Boost |
|--------|----------|---------------------------|
| Dynamic Kernel Switching | Multi-scale feature extraction | +1–3 % |
| Self-Attention | Focus on discriminative ECG regions | +2–5 % |
| Deep Conv Layers | Abstract feature consolidation | +1–2 % |
| **Final Model Accuracy** | **96.51 %** | |

---
### Setup Instructions
# 1️⃣ Clone the repository
git clone https://github.com/GokulR2003/ECG-DKS-Attention.git
cd ECG-DKS-Attention

# 2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # (on macOS/Linux)
venv\Scripts\activate      # (on Windows)

# 3️⃣ Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4️⃣ Verify installation
python -c "import tensorflow, wfdb, sklearn; print('✅ Setup Successful!')"
---

### 💬 Acknowledgements

This work was carried out under the Chips to Startup (C2S) Internship Programme,
as part of the project "Ultra Low Power SHAKTI RISC-V Based Lightweight Edge AI Processor for IoT-enabled Healthcare Applications"
funded by the Ministry of Electronics and Information Technology (MeitY), Government of India.


