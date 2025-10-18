"""
train_ecg_model.py
Train CNN model with Dynamic Kernel Switching (DKS) and Self-Attention
for ECG classification using MIT-BIH Arrhythmia Database.
"""

# === Imports ===
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import wfdb
import wfdb.processing
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import zscore
import matplotlib.pyplot as plt

# === Configuration Parameters ===
SAMPLING_RATE = 360
SEGMENT_LENGTH = 188
LOWCUT_FREQ = 0.5
HIGHCUT_FREQ = 40
NOTCH_FREQ = 50
NOTCH_Q = 30
DKS_FILTERS = 64
DKS_KERNEL_SIZES = [3, 5, 7]
ATTENTION_UNITS = 128
RECORDS = ['100', '101', '103', '105', '106', '108', '109', '111', '112', '113', '114',
           '115', '116', '117', '118', '119', '121', '122', '200', '201', '202', '203',
           '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219',
           '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']

CLASS_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'B': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'r': 'V',
    'F': 'F',
    'Q': 'Q', '/', 'f', '~', '"', '[', ']', 'x', '?', '|', 'o', '!', '+',
    '(', ')', 'p', 't', 'u', '`', '^'
}

# === Preprocessing functions, DKS and Attention classes ===
# (Paste all preprocessing and custom layer definitions here — identical to your provided code.)

# === Model creation function ===
# (Paste your `create_ecg_dks_attention_dilated_model()` here.)

# === Training phase ===
if __name__ == "__main__":
    print("--- Phase 1: Model Development (FP32) ---")

    X, y_one_hot, label_encoder, num_classes = load_and_preprocess_mitbih_enhanced(
        RECORDS, SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
        LOWCUT_FREQ, HIGHCUT_FREQ, NOTCH_FREQ, NOTCH_Q,
        r_peak_detector='annotation'
    )

    X = X[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )

    model = create_ecg_dks_attention_dilated_model(
        (SEGMENT_LENGTH, 1), num_classes, DKS_FILTERS, DKS_KERNEL_SIZES, ATTENTION_UNITS
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    y_train_labels = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = {i: total_samples / (num_classes * count) for i, count in enumerate(class_counts)}

    history = model.fit(X_train, y_train, epochs=30, batch_size=64,
                        validation_split=0.1, class_weight=class_weights)

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ecg_dks_attention_model_annotation_r30.h5")
    model.save(model_path)
    print(f"✅ Model saved at {model_path}")

    # Save results
    np.save("results/train_history.npy", history.history)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")
