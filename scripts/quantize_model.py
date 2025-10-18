"""
quantize_model.py
Convert trained ECG model to INT8 TFLite format for Edge AI deployment.
"""

import tensorflow as tf
import os
from train_ecg_model import DynamicKernelSwitching, SelfAttention

model_path = "models/ecg_dks_attention_model_annotation_r30.h5"
quantized_path = "models/ecg_dks_attention_model_quantized_dynamic_range_annotation_r30.tflite"

model = tf.keras.models.load_model(
    model_path,
    custom_objects={'DynamicKernelSwitching': DynamicKernelSwitching,
                    'SelfAttention': SelfAttention}
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(quantized_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Quantized model saved to {quantized_path}")
