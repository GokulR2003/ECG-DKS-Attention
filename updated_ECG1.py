# IMPORTANT: If you encounter 'ModuleNotFoundError: No module named 'wfdb'',
# UNCOMMENT THE LINE BELOW and run this cell ONCE to install it.
# You MUST RESTART YOUR PYTHON RUNTIME/KERNEL AFTER INSTALLATION for changes to take effect.
# !pip install wfdb

# 1️⃣ Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
# In the "Import Libraries" section
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import wfdb # For MIT-BIH database access
from tqdm import tqdm # For progress bars
from scipy.signal import butter, filtfilt, iirnotch # For signal processing
from scipy.stats import zscore # For normalization
import matplotlib.pyplot as plt # For plotting ECG segments
import wfdb.processing # For R-peak detection
import os # For creating directories

# --- Configuration Parameters ---
SAMPLING_RATE = 360  # Hz, standard for MIT-BIH
SEGMENT_LENGTH = 188 # samples, common choice for MIT-BIH heartbeat classification
# This segment length is crucial. It defines how many data points represent one heartbeat.
# A length of 188 samples at 360 Hz covers approximately 0.52 seconds.

# Preprocessing filter parameters
LOWCUT_FREQ = 0.5    # Hz for band-pass
HIGHCUT_FREQ = 40    # Hz for band-pass
NOTCH_FREQ = 50      # Hz for notch filter (adjust to 60 if using US power grid data)
NOTCH_Q = 30         # Q-factor for notch filter

# Model parameters
DKS_FILTERS = 64
DKS_KERNEL_SIZES = [3, 5, 7] # Kernel sizes for parallel convolutions in DKS
ATTENTION_UNITS = 128

# Define the dataset records to use (subset for demonstration, use all for full training)
RECORDS = ['100', '101', '103', '105', '106', '108', '109', '111', '112',
           '113', '114', '115', '116', '117', '118', '119', '121', '122',
           '200', '201', '202', '203', '205', '207', '208', '209', '210',
           '212', '213', '214', '215', '217', '219', '220', '221', '222',
           '223', '228', '230', '231', '232', '233', '234'] # All 44 records

# Class mapping based on AAMI standards (useful for MIT-BIH)
# N: Normal, LBBB, RBBB, AB, N, B, e, j
# S: SVEB, A, J, a
# V: VEB, V, r
# F: Fusion Beat, F
# Q: Unclassifiable, /, f, Q, ~, ", [, ], x, ?, |, o, !, +, (, ), p, t, u, `, ^
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
    # Ensure data is 1D for filtering
    if data.ndim > 1:
        data = data.squeeze() # Assuming single channel, remove extra dim
    y = filtfilt(b, a, data)
    return y

def iir_notch(f0, Q, fs):
    nyq = 0.5 * fs
    w0 = f0 / nyq
    b, a = iirnotch(w0, Q)
    return b, a

def iir_notch_filter(data, f0, Q, fs):
    b, a = iir_notch(f0, Q, fs)
    # Ensure data is 1D for filtering
    if data.ndim > 1:
        data = data.squeeze()
    y = filtfilt(b, a, data)
    return y

def segment_heartbeats_fixed_window(ecg_signal, r_peaks, segment_length,
                                     pre_r_percent=0.3, post_r_percent=0.7):
    """
    Segme nts ECG signal into heartbeats with a fixed length,
    but with a flexible split around the R-peak.

    Args:
        ecg_signal (np.array): The 1D ECG signal.
        r_peaks (np.array): Array of R-peak indices.
        segment_length (int): Desired length of each segment.
        pre_r_percent (float): Percentage of segment length before R-peak (0.0 to 1.0).
        post_r_percent (float): Percentage of segment length after R-peak (0.0 to 1.0).
                                 pre_r_percent + post_r_percent must be 1.0.
    Returns:
        tuple: (np.array of segments, np.array of corresponding R-peak indices)
    """
    if not np.isclose(pre_r_percent + post_r_percent, 1.0):
        raise ValueError("pre_r_percent + post_r_percent must sum to 1.0")

    half_segment_before = int(segment_length * pre_r_percent)
    half_segment_after = segment_length - half_segment_before

    segments = []
    r_peak_indices_in_segment = []

    for r_peak in r_peaks:
        start_idx = r_peak - half_segment_before
        end_idx = r_peak + half_segment_after

        # Create a temporary segment, handle padding
        temp_segment = np.zeros(segment_length, dtype=ecg_signal.dtype)

        src_start = max(0, start_idx)
        src_end = min(len(ecg_signal), end_idx)

        dest_start = 0 if start_idx >= 0 else -start_idx
        dest_end = dest_start + (src_end - src_start)

        temp_segment[dest_start:dest_end] = ecg_signal[src_start:src_end]

        # Fill padded regions with edge values
        if start_idx < 0:
            temp_segment[0:dest_start] = ecg_signal[0]
        if end_idx > len(ecg_signal):
            temp_segment[dest_end:segment_length] = ecg_signal[-1]

        segments.append(temp_segment)
        r_peak_indices_in_segment.append(r_peak) # Store the original R-peak index for later annotation

    return np.array(segments), np.array(r_peak_indices_in_segment)


# --- Data Loading and Full Preprocessing Pipeline ---

def load_and_preprocess_mitbih_enhanced(
    records, sampling_rate, segment_length, class_mapping,
    lowcut, highcut, notch_freq, notch_Q,
    r_peak_detector='xqrs', # 'xqrs', 'gqrs', or 'annotation'
    segment_pre_r_percent=0.3, # For segment_heartbeats_fixed_window
    segment_post_r_percent=0.7
):
    all_segments = []
    all_labels = []

    print(f"Using R-peak detector: {r_peak_detector.upper()}")

    for rec_name in tqdm(records, desc="Processing MIT-BIH Records"):
        try:
            record = wfdb.rdrecord(rec_name, pn_dir='mitdb/1.0.0')
            annotation = wfdb.rdann(rec_name, 'atr', pn_dir='mitdb/1.0.0')

            # Use channel 0 (MLII lead) for simplicity
            ecg_signal = record.p_signal[:, 0].astype(np.float32)

            # --- Preprocessing Step 1: Denoising ---
            filtered_signal_bp = butter_bandpass_filter(ecg_signal, lowcut, highcut, sampling_rate)
            denoised_signal = iir_notch_filter(filtered_signal_bp, notch_freq, notch_Q, sampling_rate)

            # --- Preprocessing Step 2: R-peak Detection ---
            r_peaks = []
            if r_peak_detector == 'xqrs':
                xqrs = wfdb.processing.XQRS(sig=denoised_signal, fs=sampling_rate)
                xqrs.detect()
                r_peaks = xqrs.qrs_inds
            elif r_peak_detector == 'gqrs':
                # gqrs can be slow for very long signals, consider chunking for very large datasets
                # Adjust threshold or other parameters for better performance if needed
                r_peaks = wfdb.processing.gqrs_detect(sig=denoised_signal, fs=sampling_rate)
            elif r_peak_detector == 'annotation':
                # Filter annotations to include only R-peaks (or relevant beat types)
                # 'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F' are usually associated with R-peaks
                r_peaks = [s for i, s in enumerate(annotation.sample) if annotation.symbol[i] in ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', '/', 'f']]
                r_peaks = np.array(r_peaks)
            else:
                raise ValueError(f"Unknown R-peak detector: {r_peak_detector}")

            # Ensure R-peaks are sorted and unique
            r_peaks = np.unique(r_peaks)
            r_peaks.sort()

            # --- Preprocessing Step 3: Heartbeat Segmentation ---
            segments, r_peak_orig_indices = segment_heartbeats_fixed_window(
                denoised_signal, r_peaks, segment_length,
                pre_r_percent=segment_pre_r_percent, post_r_percent=segment_post_r_percent
            )

            # --- Preprocessing Step 4: Normalization (Z-score) ---
            # Explicitly cast to float32 to prevent type issues during quantization
            normalized_segments = np.array([zscore(s) for s in segments], dtype=np.float32)

            # Get labels for each detected R-peak
            labels_for_segments = []
            for i, r_idx in enumerate(r_peak_orig_indices):
                # Find the closest annotation to the detected R-peak
                # We need to ensure the annotation symbol corresponds to the R-peak's type.
                # MIT-BIH annotations mark the R-peak itself.
                closest_ann_idx = np.argmin(np.abs(annotation.sample - r_idx))
                rhythm_char = annotation.symbol[closest_ann_idx]

                # Check if the closest annotation is within a reasonable window
                if np.abs(annotation.sample[closest_ann_idx] - r_idx) <= 5: # Within 5 samples tolerance
                    mapped_label = class_mapping.get(rhythm_char, 'Q') # Default to 'Q' if not found
                else:
                    mapped_label = 'Q' # Treat as unclassifiable if no close annotation

                labels_for_segments.append(mapped_label)

            all_segments.extend(normalized_segments)
            all_labels.extend(labels_for_segments)

        except Exception as e:
            print(f"Error processing record {rec_name}: {e}")
            continue

    all_segments = np.array(all_segments)
    all_labels = np.array(all_labels)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)

    # Convert to one-hot encoding for Keras
    num_classes = len(label_encoder.classes_)
    one_hot_labels = to_categorical(encoded_labels, num_classes=num_classes)

    print(f"\nTotal heartbeats extracted: {len(all_segments)}")
    print(f"Distribution of classes (before one-hot):")
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(all_labels)*100:.2f}%)")

    return all_segments, one_hot_labels, label_encoder, num_classes

# --- Custom Keras Layers ---

# Custom Layer for Dynamic Kernel Switching
class DynamicKernelSwitching(layers.Layer):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super(DynamicKernelSwitching, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.conv_layers = []
        for k_size in kernel_sizes:
            self.conv_layers.append(layers.Conv1D(filters=filters, kernel_size=k_size,
                                                 padding='same', activation='relu',
                                                 kernel_initializer='he_normal')) # He initializer for ReLU
        # Gating mechanism: A small network to learn weights for each kernel output
        self.gating_dense1 = layers.Dense(len(kernel_sizes), activation='relu')
        self.gating_dense2 = layers.Dense(len(kernel_sizes), activation='softmax') # Softmax for weights

    def call(self, inputs):
        conv_outputs = [conv_layer(inputs) for conv_layer in self.conv_layers]

        pooled_features = [K.mean(conv_out, axis=1) for conv_out in conv_outputs]
        gating_input = K.concatenate(pooled_features, axis=-1)

        weights = self.gating_dense2(self.gating_dense1(gating_input))

        expanded_weights = K.expand_dims(weights, axis=1)

        weighted_sum = None
        for i, conv_out in enumerate(conv_outputs):
            current_weight = expanded_weights[:, :, i]
            current_weight_broadcast = K.expand_dims(current_weight, axis=-1)
            weighted_conv = conv_out * current_weight_broadcast
            if weighted_sum is None:
                weighted_sum = weighted_conv
            else:
                weighted_sum = weighted_sum + weighted_conv

        return weighted_sum

    def get_config(self):
        config = super(DynamicKernelSwitching, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes,
        })
        return config


# Custom Layer for Self-Attention Mechanism (Bahdanau style for sequences)
class SelfAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units) # Query projection
        self.U = layers.Dense(units) # Key projection
        self.V = layers.Dense(1)     # Score projection

    def call(self, inputs): # inputs shape: (batch_size, sequence_length, features)
        query = self.W(inputs)
        key = self.U(inputs)

        score = K.tanh(query + key)

        attention_weights = K.softmax(self.V(score), axis=1)

        context_vector = attention_weights * inputs

        context_vector_summed = K.sum(context_vector, axis=1)

        context_vector_repeated = K.expand_dims(context_vector_summed, axis=1)
        context_vector_repeated = K.repeat_elements(context_vector_repeated, K.int_shape(inputs)[1], axis=1)

        output = K.concatenate([inputs, context_vector_repeated], axis=-1)
        return output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'units': self.units,
        })
        return config


# --- Model Definition ---

# --- MODIFIED Model Definition with Dilation ---

def create_ecg_dks_attention_dilated_model(input_shape, num_classes,
                                             dks_filters, dks_kernel_sizes,
                                             attention_units):

    inputs = layers.Input(shape=input_shape)

    # Dynamic Kernel Switching Block (Unchanged)
    # This block is excellent at finding important local, multi-scale features.
    x = DynamicKernelSwitching(filters=dks_filters, kernel_sizes=dks_kernel_sizes, name='dks_block')(inputs)
    print(f"Output shape after DKS Block: {x.shape}")

    # Attention Mechanism (Unchanged)
    # This focuses the model on the most relevant parts of the sequence from the DKS block.
    x = SelfAttention(units=attention_units, name='attention_mechanism')(x)
    print(f"Output shape after Attention Mechanism: {x.shape}")

    # --- MODIFICATION STARTS HERE ---
    # Further Feature Extraction using DILATED convolutions to capture broader context.
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', padding='same',
                      dilation_rate=2,  # <-- ADDED DILATION
                      kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    print(f"Output shape after Dilated Conv1D (128 filters, rate 2) and MaxPooling: {x.shape}")

    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu', padding='same',
                      dilation_rate=4,  # <-- ADDED DILATION
                      kernel_initializer='he_normal')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    print(f"Output shape after Dilated Conv1D (256 filters, rate 4) and MaxPooling: {x.shape}")
    # --- MODIFICATION ENDS HERE ---

    # Classification Head (Unchanged)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    print(f"Output shape after GlobalAveragePooling1D and Dropout: {x.shape}")

    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    print(f"Output shape of Classification Head: {outputs.shape}")

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# --- Main Execution ---

if __name__ == "__main__":
    # --- Phase 1: Model Development (FP32) ---
    print("--- Phase 1: Model Development (FP32) ---")
    print("Loading and Preprocessing MIT-BIH Arrhythmia Database...")

    # You can choose the R-peak detector here: 'xqrs', 'gqrs', or 'annotation'
    R_PEAK_DETECTOR_STRATEGY = 'annotation' # Change this to 'xqrs' or 'gqrs' to compare

    # You can adjust segment percentages (e.g., to emphasize P-wave or T-wave)
    # Ensure pre_r_percent + post_r_percent = 1.0
    SEGMENT_PRE_R_PERCENT = 0.3
    SEGMENT_POST_R_PERCENT = 0.7

    # 1. Load and preprocess data using the enhanced function
    X, y_one_hot, label_encoder, num_classes = load_and_preprocess_mitbih_enhanced(
        RECORDS, SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
        LOWCUT_FREQ, HIGHCUT_FREQ, NOTCH_FREQ, NOTCH_Q,
        r_peak_detector=R_PEAK_DETECTOR_STRATEGY,
        segment_pre_r_percent=SEGMENT_PRE_R_PERCENT,
        segment_post_r_percent=SEGMENT_POST_R_PERCENT
    )

    # Add a channel dimension for CNN
    X = X[..., np.newaxis]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot
    )

    print(f"\nTraining data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}, Class labels: {label_encoder.classes_}")

    # Optional: Plot a few preprocessed ECG segments to verify
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(X_train[i].flatten())
        plt.title(f"Sample Preprocessed ECG - Class: {label_encoder.inverse_transform([np.argmax(y_train[i])])[0]}")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2. Create and compile the model
    input_shape = (SEGMENT_LENGTH, 1) # (sequence_length, channels)
    
    model = create_ecg_dks_attention_dilated_model(
    input_shape, num_classes, DKS_FILTERS, DKS_KERNEL_SIZES, ATTENTION_UNITS
)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Use categorical_crossentropy for one-hot labels
                  metrics=['accuracy'])

    print("\n--- Model Summary ---")
    model.summary()

    # 3. Train the model
    print("\n--- Model Training ---")
    # Due to class imbalance, it's often beneficial to use class weights
    # Calculate class weights
    y_train_labels = np.argmax(y_train, axis=1) # Convert one-hot back to integer labels
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    # Handle cases where a class might have 0 samples (though unlikely with stratify)
    class_weights = {i: total_samples / (num_classes * count) if count > 0 else 0 for i, count in enumerate(class_counts)}
    print(f"Calculated Class Weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        epochs=30, # Increased epochs for better training
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights, # Apply class weights
        verbose=1
    )

    # 4. Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n--- Final Model Evaluation on Test Set ---")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    # --- NEW: Calculate Precision, Recall, F1-Score ---
    print("\n--- Classification Report ---")
    # Get model predictions
    y_pred_probs = model.predict(X_test)
    # Convert predictions from probabilities to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)
    # Convert one-hot encoded true labels to class labels
    y_true = np.argmax(y_test, axis=1)
    # Get class names from the label encoder
    class_names = label_encoder.classes_
    # Generate and print the report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 5. Save the best model
    # Define the directory to save the model
    MODEL_SAVE_DIR = '/content/internship'
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Create the directory if it doesn't exist

    model_filename = f'ecg_dks_attention_model_{R_PEAK_DETECTOR_STRATEGY}_r{int(SEGMENT_PRE_R_PERCENT*100)}.h5'
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    print("\n--- Accuracy Progression during Training (Conceptual Impact) ---")
    print("Note: These are overall model accuracies per epoch, not layer-specific.")
    for epoch, (acc, val_acc) in enumerate(zip(history.history['accuracy'], history.history['val_accuracy'])):
        print(f"Epoch {epoch+1}: Training Accuracy = {acc*100:.2f}%, Validation Accuracy = {val_acc*100:.2f}%")

    print("\n--- Conceptual Accuracy Levels After Each Filter/Block (As per Research Plan) ---")
    print("1. After Raw ECG Data:")
    print("  - Accuracy: Highly variable, often 60-70% if directly fed to a simple classifier (very noisy).")
    print("2. After Preprocessing (Band-pass, Notch Filters & R-peak detection/Segmentation):")
    print("  - R-peak Detection Strategy Used: " + R_PEAK_DETECTOR_STRATEGY.upper())
    print(f"  - Segment Window Split: {SEGMENT_PRE_R_PERCENT*100:.0f}% before R-peak, {SEGMENT_POST_R_PERCENT*100:.0f}% after R-peak.")
    print("  - Impact: Signal-to-Noise Ratio (SNR) significantly improved. Heartbeat segments are isolated and cleaner. Different R-peak detectors capture different nuances.")
    print("  - Conceptual Accuracy: If a basic classifier were used on this cleaner data, accuracy could jump to 80-90%. This is the foundation.")
    print("3. After Dynamic Kernel Switching (DKS) Block:")
    print(f"  - Kernel Sizes Used: {DKS_KERNEL_SIZES}. Filters Used: {DKS_FILTERS} per kernel size.")
    print("  - Impact: Extracts multi-scale features, adapting to different temporal patterns in ECG.")
    print("  - Conceptual Accuracy Improvement: Over a standard CNN with fixed kernels, DKS offers richer feature learning, potentially adding 1-3% to accuracy (e.g., from 88% to 90%).")
    print("4. After Attention Mechanism:")
    print(f"  - Units Used: {ATTENTION_UNITS}.")
    print("  - Impact: Allows the model to focus on the most discriminative parts of the ECG segment.")
    print("  - Conceptual Accuracy Improvement: By highlighting crucial features, it can lead to another 2-5% accuracy boost (e.g., from 90% to 93-95%).")
    print("5. After Deeper CNN Layers and Final Classification Head:")
    print(f"  - Conv1D Layer 1: Filters=128, Kernel=5")
    print(f"  - Conv1D Layer 2: Filters=256, Kernel=5")
    print("  - Impact: Further learns abstract representations and performs final classification.")
    print(f"  - Final Achieved Accuracy (on Test Set): {accuracy*100:.2f}% (This is the result of the entire model).")

    # --- Phase 2: Model Quantization ---
    print("\n--- Phase 2: Model Quantization ---")

    # Define the path to your saved model. This now points to the 'internship' directory.
    # This MODEL_FILENAME should match the one generated by the training part above.
    MODEL_FILENAME_FOR_QUANTIZATION = model_filename # Use the filename generated during training
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME_FOR_QUANTIZATION)

    print(f"Attempting to load model from: {MODEL_PATH}")

    try:
        # Load the model with custom objects
        # It's crucial to provide the custom layers when loading a model that uses them.
        model_for_quantization = models.load_model(
            MODEL_PATH,
            custom_objects={
                'DynamicKernelSwitching': DynamicKernelSwitching,
                'SelfAttention': SelfAttention
            }
        )
        print("Model loaded successfully for quantization!")
        # model_for_quantization.summary() # Uncomment to see summary of loaded model
    except Exception as e:
        print(f"Error loading model for quantization: {e}")
        print("Please ensure the model file exists at the specified path and the custom layers are correctly defined.")
        print("If you haven't trained and saved the model yet, please run the training part of this script first.")
        exit() # Exit if model loading fails

    print("\n--- Starting Post-Training Dynamic Range Quantization ---")
    print("This method quantizes weights to 8-bit integers and dynamically quantizes activations during inference.")
    print("It's a good balance of model size reduction, speedup, and minimal accuracy loss.")

    # Create the TFLite converter from the loaded Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_quantization)

    # Set optimizations for dynamic range quantization (default behavior of Optimize.DEFAULT)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # For dynamic range quantization, a representative dataset is NOT strictly required,
    # but it can still be provided for better calibration of activations if desired.
    # However, to avoid the previous XNNPACK error, we will NOT enforce full integer ops
    # and thus a representative dataset for full integer calibration is not mandatory here.

    # Remove the strict target_spec.supported_ops for dynamic range quantization
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] # REMOVED

    # Convert the model
    try:
        tflite_quantized_model = converter.convert()
        print("\nPost-Training Dynamic Range Quantization successful!")
    except Exception as e:
        print(f"\nError during quantization: {e}")
        print("This might happen if some operations are not supported for dynamic range quantization.")
        print("Consider trying Quantization Aware Training if this persists.")
        exit()

    # Define the output path for the quantized model
    quantized_model_filename = f'ecg_dks_attention_model_quantized_dynamic_range_{R_PEAK_DETECTOR_STRATEGY}_r{int(SEGMENT_PRE_R_PERCENT*100)}.tflite'
    quantized_model_path = os.path.join(MODEL_SAVE_DIR, quantized_model_filename) # Save in the same 'internship' directory

    # Save the quantized model
    with open(quantized_model_path, 'wb') as f:
        f.write(tflite_quantized_model)
    print(f"Quantized model saved to: {quantized_model_path}")

    # --- Optional: Verify the Quantized TFLite Model ---
    print("\n--- Verifying Quantized TFLite Model ---")

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_quantized_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load a small test dataset for verification
    print("Loading test data for TFLite model verification...")
    X_test_verify, y_one_hot_verify, label_encoder_verify, _ = load_and_preprocess_mitbih_enhanced(
        RECORDS[5:10], SAMPLING_RATE, SEGMENT_LENGTH, CLASS_MAPPING,
        LOWCUT_FREQ, HIGHCUT_FREQ, NOTCH_FREQ, NOTCH_Q,
        r_peak_detector=R_PEAK_DETECTOR_STRATEGY,
        segment_pre_r_percent=SEGMENT_PRE_R_PERCENT,
        segment_post_r_percent=SEGMENT_POST_R_PERCENT
    )
    # Ensure X_test_verify has the channel dimension and is float32, as expected by the dynamic range quantized model
    X_test_verify = X_test_verify[..., np.newaxis].astype(np.float32)

    if X_test_verify.shape[0] == 0:
        print("No test data available for TFLite verification. Skipping verification.")
    else:
        # Select a few samples for inference
        num_samples_to_test = min(5, X_test_verify.shape[0])
        test_samples = X_test_verify[:num_samples_to_test]
        true_labels = label_encoder_verify.inverse_transform(np.argmax(y_one_hot_verify[:num_samples_to_test], axis=1))

        print(f"\nRunning inference on {num_samples_to_test} test samples with TFLite model...")
        tflite_predictions = []
        for i in range(num_samples_to_test):
            # For dynamic range quantization, input data remains float32
            input_data = test_samples[i:i+1] # No scaling needed for input for dynamic range

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Output data from dynamic range quantized models is typically float32
            tflite_predictions.append(output_data)

            predicted_class_idx = np.argmax(output_data)
            predicted_label = label_encoder_verify.inverse_transform([predicted_class_idx])[0]

            print(f"Sample {i+1}: True Label: {true_labels[i]}, Predicted Label: {predicted_label}")
            # print(f"  Raw TFLite Output: {output_data.flatten()}") # Uncomment to see raw output

        print("\nTFLite model verification complete.")

    print("\n--- Phase 3: RTL Generation (Conceptual) ---")
    print("With the .tflite model, you can now proceed to hardware deployment.")
    print("This involves specialized tools like hls4ml (for FPGAs) or other hardware-specific compilers to translate the quantized model into Hardware Description Languages (Verilog/SystemVerilog/VHDL).")
    print("This phase includes simulation with testbenches and eventual implementation on FPGA/ASIC for real-time inference.")

    # --- Conceptual: Quantization Aware Training (QAT) ---
    print("\n--- Conceptual: Quantization Aware Training (QAT) ---")
    print("If the accuracy of the Post-Training Quantized model is not satisfactory,")
    print("Quantization Aware Training (QAT) is the next step. This involves:")
    print("1. Applying `tfmot.quantization.keras.quantize_model` to your original Keras model.")
    print("2. Retraining the 'quantized' model for a few epochs. During this retraining,")
    print("   the model learns weights that are robust to quantization effects.")
    print("3. Converting the QAT-trained model to TFLite, which typically yields higher accuracy.")
    print("\nExample (conceptual, requires TensorFlow Model Optimization Toolkit):")
    print(" # import tensorflow_model_optimization as tfmot")
    print(" # quantized_model_for_qat = tfmot.quantization.keras.quantize_model(model)")
    print(" # quantized_model_for_qat.compile(...)")
    print(" # quantized_model_for_qat.fit(X_train, y_train, ...)")
    print(" # converter_qat = tf.lite.TFLiteConverter.from_keras_model(quantized_model_for_qat)")
    print(" # converter_qat.optimizations = [tf.lite.Optimize.DEFAULT]")
    print(" # tflite_qat_model = converter_qat.convert()")
