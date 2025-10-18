"""
evaluate_model.py
Evaluate trained ECG-DKS model and generate classification report + plots.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os

from train_ecg_model import DynamicKernelSwitching, SelfAttention, SEGMENT_LENGTH, CLASS_MAPPING, SAMPLING_RATE, RECORDS, \
    load_and_preprocess_mitbih_enhanced

model_path = "models/ecg_dks_attention_model_annotation_r30.h5"
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'DynamicKernelSwitching': DynamicKernelSwitching,
                                                   'SelfAttention': SelfAttention})

X_test, y_test_onehot, label_encoder, _ = load_and_preprocess_mitbih_enhanced(
    RECORDS[5:10], SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
    0.5, 40, 50, 30, r_peak_detector='annotation'
)
X_test = X_test[..., np.newaxis]

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_onehot, axis=1)
report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
print(report)

cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(cmap='Blues')
plt.title("Confusion Matrix - ECG DKS + Self-Attention")
plt.savefig("results/confusion_matrix.png")
plt.show()
