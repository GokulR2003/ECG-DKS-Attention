# IMPORTANT: If you encounter 'ModuleNotFoundError: No module named 'wfdb'',
# UNCOMMENT THE LINE BELOW and run this cell ONCE to install it.
# You MUST RESTART YOUR PYTHON RUNTIME/KERNEL AFTER INSTALLATION for changes to take effect.
# !pip install wfdb

# 1️⃣ Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # NEW: For stable training
import sklearn.metrics # MODIFIED: For robust classification report
import wfdb # For MIT-BIH database access
from tqdm import tqdm # For progress bars
from scipy.signal import butter, filtfilt, iirnotch # For signal processing
from scipy.stats import zscore # For normalization
import matplotlib.pyplot as plt # For plotting ECG segments
import wfdb.processing # For R-peak detection
import os # For creating directories

# --- Configuration Parameters ---
SAMPLING_RATE = 360
SEGMENT_LENGTH = 188
LOWCUT_FREQ = 0.5
HIGHCUT_FREQ = 40
NOTCH_FREQ = 50
NOTCH_Q = 30
DKS_FILTERS = 64
DKS_KERNEL_SIZES = [3, 5, 7]
ATTENTION_UNITS = 128
RECORDS = ['100', '101', '103', '105', '106', '108', '109', '111', '112',
           '113', '114', '115', '116', '117', '118', '119', '121', '122',
           '200', '201', '202', '203', '205', '207', '208', '209', '210',
           '212', '213', '214', '215', '217', '219', '220', '221', '222',
           '223', '228', '230', '231', '232', '233', '234']

CLASS_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'B': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'r': 'V',
    'F': 'F',
    'Q': 'Q', '/': 'Q', 'f': 'Q', '~': 'Q', '"': 'Q', '[': 'Q',
    ']': 'Q', 'x': 'Q', '?': 'Q', '|': 'Q', 'o': 'Q', '!': 'Q', '+': 'Q',
    '(': 'Q', ')': 'Q', 'p': 'Q', 't': 'Q', 'u': 'Q', '`': 'Q', '^': 'Q'
}

# --- Preprocessing Functions ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if data.ndim > 1:
        data = data.squeeze()
    y = filtfilt(b, a, data)
    return y

def iir_notch(f0, Q, fs):
    nyq = 0.5 * fs
    w0 = f0 / nyq
    b, a = iirnotch(w0, Q)
    return b, a

def iir_notch_filter(data, f0, Q, fs):
    b, a = iir_notch(f0, Q, fs)
    if data.ndim > 1:
        data = data.squeeze()
    y = filtfilt(b, a, data)
    return y

def segment_heartbeats_fixed_window(ecg_signal, r_peaks, segment_length,
                                     pre_r_percent=0.3, post_r_percent=0.7):
    if not np.isclose(pre_r_percent + post_r_percent, 1.0):
        raise ValueError("pre_r_percent + post_r_percent must sum to 1.0")
    half_segment_before = int(segment_length * pre_r_percent)
    half_segment_after = segment_length - half_segment_before
    segments = []
    r_peak_indices_in_segment = []
    for r_peak in r_peaks:
        start_idx = r_peak - half_segment_before
        end_idx = r_peak + half_segment_after
        temp_segment = np.zeros(segment_length, dtype=ecg_signal.dtype)
        src_start = max(0, start_idx)
        src_end = min(len(ecg_signal), end_idx)
        dest_start = 0 if start_idx >= 0 else -start_idx
        dest_end = dest_start + (src_end - src_start)
        temp_segment[dest_start:dest_end] = ecg_signal[src_start:src_end]
        if start_idx < 0:
            temp_segment[0:dest_start] = ecg_signal[0]
        if end_idx > len(ecg_signal):
            temp_segment[dest_end:segment_length] = ecg_signal[-1]
        segments.append(temp_segment)
        r_peak_indices_in_segment.append(r_peak)
    return np.array(segments), np.array(r_peak_indices_in_segment)

# --- Data Loading and Full Preprocessing Pipeline ---

def load_and_preprocess_mitbih_enhanced(
    records, sampling_rate, segment_length, class_mapping,
    lowcut, highcut, notch_freq, notch_Q,
    r_peak_detector='xqrs',
    segment_pre_r_percent=0.3,
    segment_post_r_percent=0.7
):
    all_segments = []
    all_labels = []
    print(f"Using R-peak detector: {r_peak_detector.upper()}")
    for rec_name in tqdm(records, desc="Processing MIT-BIH Records"):
        try:
            record = wfdb.rdrecord(rec_name, pn_dir='mitdb/1.0.0')
            annotation = wfdb.rdann(rec_name, 'atr', pn_dir='mitdb/1.0.0')
            ecg_signal = record.p_signal[:, 0].astype(np.float32)
            filtered_signal_bp = butter_bandpass_filter(ecg_signal, lowcut, highcut, sampling_rate)
            denoised_signal = iir_notch_filter(filtered_signal_bp, notch_freq, notch_Q, sampling_rate)
            r_peaks = []
            if r_peak_detector == 'xqrs':
                xqrs = wfdb.processing.XQRS(sig=denoised_signal, fs=sampling_rate)
                xqrs.detect()
                r_peaks = xqrs.qrs_inds
            elif r_peak_detector == 'annotation':
                r_peaks = [s for i, s in enumerate(annotation.sample) if annotation.symbol[i] in class_mapping.keys()]
                r_peaks = np.array(r_peaks)
            else:
                raise ValueError(f"Unknown R-peak detector: {r_peak_detector}")
            r_peaks = np.unique(r_peaks)
            r_peaks.sort()
            segments, r_peak_orig_indices = segment_heartbeats_fixed_window(
                denoised_signal, r_peaks, segment_length,
                pre_r_percent=segment_pre_r_percent, post_r_percent=segment_post_r_percent
            )
            normalized_segments = np.array([zscore(s) for s in segments], dtype=np.float32)
            labels_for_segments = []
            for i, r_idx in enumerate(r_peak_orig_indices):
                closest_ann_idx = np.argmin(np.abs(annotation.sample - r_idx))
                rhythm_char = annotation.symbol[closest_ann_idx]
                if np.abs(annotation.sample[closest_ann_idx] - r_idx) <= 5:
                    mapped_label = class_mapping.get(rhythm_char, 'Q')
                else:
                    mapped_label = 'Q'
                labels_for_segments.append(mapped_label)
            all_segments.extend(normalized_segments)
            all_labels.extend(labels_for_segments)
        except Exception as e:
            print(f"Error processing record {rec_name}: {e}")
            continue
    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    num_classes = len(label_encoder.classes_)
    one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)
    print(f"\nTotal heartbeats extracted: {len(all_segments)}")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(all_labels)*100:.2f}%)")
    return all_segments, one_hot_labels, label_encoder, num_classes

# --- Custom Keras Layers ---
class DynamicKernelSwitching(layers.Layer):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super(DynamicKernelSwitching, self).__init__(**kwargs)
        self.filters = filters; self.kernel_sizes = kernel_sizes
        self.conv_layers = [layers.Conv1D(filters=filters, kernel_size=k_size, padding='same', activation='relu', kernel_initializer='he_normal') for k_size in kernel_sizes]
        self.gating_dense1 = layers.Dense(len(kernel_sizes), activation='relu')
        self.gating_dense2 = layers.Dense(len(kernel_sizes), activation='softmax')
    def call(self, inputs):
        conv_outputs = [conv_layer(inputs) for conv_layer in self.conv_layers]
        pooled_features = [K.mean(conv_out, axis=1) for conv_out in conv_outputs]
        gating_input = K.concatenate(pooled_features, axis=-1)
        weights = self.gating_dense2(self.gating_dense1(gating_input))
        expanded_weights = K.expand_dims(weights, axis=1)
        weighted_sum = None
        for i, conv_out in enumerate(conv_outputs):
            current_weight = K.expand_dims(expanded_weights[:, :, i], axis=-1)
            if weighted_sum is None: weighted_sum = conv_out * current_weight
            else: weighted_sum = weighted_sum + (conv_out * current_weight)
        return weighted_sum
    def get_config(self):
        config = super(DynamicKernelSwitching, self).get_config()
        config.update({'filters': self.filters, 'kernel_sizes': self.kernel_sizes}); return config

class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units; self.W = layers.Dense(units); self.U = layers.Dense(units); self.V = layers.Dense(1)
    def call(self, inputs):
        query = self.W(inputs); key = self.U(inputs)
        score = K.tanh(query + key)
        attention_weights = K.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector_summed = K.sum(context_vector, axis=1)
        context_vector_repeated = K.repeat_elements(K.expand_dims(context_vector_summed, axis=1), K.int_shape(inputs)[1], axis=1)
        return K.concatenate([inputs, context_vector_repeated], axis=-1)
    def get_config(self):
        config = super(SelfAttention, self).get_config(); config.update({'units': self.units}); return config

# --- Model Definition ---
def create_ecg_dks_attention_dilated_model(input_shape, num_classes, dks_filters, dks_kernel_sizes, attention_units):
    inputs = layers.Input(shape=input_shape)
    x = DynamicKernelSwitching(filters=dks_filters, kernel_sizes=dks_kernel_sizes, name='dks_block')(inputs)
    x = SelfAttention(units=attention_units, name='attention_mechanism')(x)
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', dilation_rate=2, kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same', dilation_rate=4, kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Phase 1: Model Development (FP32) ---")
    R_PEAK_DETECTOR_STRATEGY = 'annotation'
    SEGMENT_PRE_R_PERCENT = 0.3
    SEGMENT_POST_R_PERCENT = 0.7
    X, y_one_hot, label_encoder, num_classes = load_and_preprocess_mitbih_enhanced(
        RECORDS, SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
        LOWCUT_FREQ, HIGHCUT_FREQ, NOTCH_FREQ, NOTCH_Q,
        r_peak_detector=R_PEAK_DETECTOR_STRATEGY,
        segment_pre_r_percent=SEGMENT_PRE_R_PERCENT,
        segment_post_r_percent=SEGMENT_POST_R_PERCENT
    )
    X = X[..., np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )
    print(f"\nTraining data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}, Class labels: {label_encoder.classes_}")

    input_shape = (SEGMENT_LENGTH, 1)
    model = create_ecg_dks_attention_dilated_model(
        input_shape, num_classes, DKS_FILTERS, DKS_KERNEL_SIZES, ATTENTION_UNITS
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\n--- Model Summary ---")
    model.summary()

    print("\n--- Model Training ---")
    y_train_labels = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = {i: total_samples / (num_classes * count) if count > 0 else 0 for i, count in enumerate(class_counts)}
    print(f"Calculated Class Weights: {class_weights}")

    # --- NEW: Add Callbacks for Stable Training ---
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True # Restores model weights from the epoch with the best value of the monitored quantity.
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=3,
        min_lr=1e-6
    )

    # --- MODIFIED: Train with callbacks ---
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=64,
        validation_split=0.1,
        class_weight=class_weights,
        verbose=1,
        callbacks=[early_stopping, reduce_lr] # Pass callbacks to training
    )

    # --- Evaluate the model ---
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Final Model Evaluation on Test Set ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # --- Calculate Precision, Recall, F1-Score ---
    print("\n--- Classification Report ---")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    class_names = label_encoder.classes_
    
    # MODIFIED: Call classification_report from the sklearn.metrics module
    report = sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # --- Save the best model ---
    MODEL_SAVE_DIR = '/content/internship'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_filename = f'ecg_dks_attention_model_{R_PEAK_DETECTOR_STRATEGY}_stable.h5'
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Phase 2: Model Quantization (Conceptual) ---
    # The quantization steps remain the same as before. You would load the newly saved
    # stable model ('..._stable.h5') and proceed with the TFLite conversion.
    print("\n--- Phase 2: Model Quantization can now proceed with the stable model. ---")
    # --- Phase 2: Model Quantization ---
    print("\n--- Phase 2: Model Quantization ---")
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, model_filename)
    print(f"Attempting to load model from: {MODEL_PATH}")
    try:
        model_for_quantization = models.load_model(
            MODEL_PATH,
            custom_objects={'DynamicKernelSwitching': DynamicKernelSwitching, 'SelfAttention': SelfAttention}
        )
        print("Model loaded successfully for quantization!")
    except Exception as e: print(f"Error loading model for quantization: {e}"); exit()

    print("\n--- Starting Post-Training Full Integer Quantization ---")
    print("Preparing a representative dataset for quantization calibration...")
    X_calibration, _, _, _ = load_and_preprocess_mitbih_enhanced(
        RECORDS[:5], SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
        LOWCUT_FREQ, HIGHCUT_FREQ, NOTCH_FREQ, NOTCH_Q,
        r_peak_detector=R_PEAK_DETECTOR_STRATEGY,
        segment_pre_r_percent=SEGMENT_PRE_R_PERCENT,
        segment_post_r_percent=SEGMENT_POST_R_PERCENT
    )
    X_calibration = X_calibration[..., np.newaxis].astype(np.float32)
    def representative_dataset_generator():
        num_calibration_samples = min(100, len(X_calibration))
        indices = np.random.choice(len(X_calibration), num_calibration_samples, replace=False)
        for i in indices: yield [X_calibration[i:i+1]]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_quantized_model = converter.convert()
        print("\nPost-Training Full Integer Quantization successful!")
    except Exception as e: print(f"\nError during quantization: {e}"); exit()
    
    quantized_model_filename = f'ecg_dks_attention_model_quantized_int8_{R_PEAK_DETECTOR_STRATEGY}_r{int(SEGMENT_PRE_R_PERCENT*100)}.tflite'
    quantized_model_path = os.path.join(MODEL_SAVE_DIR, quantized_model_filename)
    with open(quantized_model_path, 'wb') as f: f.write(tflite_quantized_model)
    print(f"Quantized model saved to: {quantized_model_path}")