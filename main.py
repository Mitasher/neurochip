"""
AltAI-2 compatible CNN → SNN pipeline.

Architecture constraints (AltAI-2 / KNP ann2snn):
  - Activations: heaviside (binary {0, 1} outputs)
  - Weights: ternary {-1, 0, 1} (auto-quantized by TernaryConv2D/TernaryDense)
  - Bias: integer, range [-16384, 16383]
  - No MaxPooling — use stride in conv layers
  - BatchNormalization with scale=False
  - Input data: binary {0, 1}
  - Max 32 filters for Conv2D, max 512 neurons for Dense
  - Kernel size: 3x3 or 5x5
  - Input resolution: up to 128x128, up to 10 channels
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from knp_ann2snn.altainn.ternary_tf2 import (
    TernaryConv2D, TernaryDense, heaviside
)
from knp_ann2snn import Placer
from knp_ann2snn.python_altai import Altai
from pathlib import Path
import time
from data_loader import get_data_splits

# ============================================================
# Configuration
# ============================================================
IMAGE_SIZE = (32, 32)
NUM_CHANNELS = 1  # grayscale binary
NUM_CLASSES = 2   # hat / person
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

MODEL_PATH = 'helmet_ternary_model.h5'
SNN_CONFIG_PATH = 'helmet_snn_config.json'
IMG_DIR = "helmets.yolov8/train/images"
LBL_DIR = "helmets.yolov8/train/labels"


# ============================================================
# Model
# ============================================================
def build_ternary_model():
    """
    AltAI-2-compatible ternary CNN.

    Architecture:
      Input (32×32×1, binary {0,1})
        → TernaryConv2D(7, 3×3, stride=2, heaviside) + BN    # → 16×16×7
        → TernaryConv2D(15, 3×3, stride=2, heaviside) + BN   # → 8×8×15
        → TernaryConv2D(15, 3×3, stride=1, heaviside) + BN   # → 8×8×15
        → Flatten                                             # → 960 (60 neurons per class if dense is 120?)
        → TernaryDense(64, heaviside)
        → TernaryDense(2, heaviside)

    All weights quantized to {-1, 0, 1}.
    All inter-layer activations are binary {0, 1} via heaviside.
    Loss = MSE with one-hot labels.
    """
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, NUM_CHANNELS)),

        # Conv block 1: 1→7 filters, stride=2 (32→16)
        TernaryConv2D(
            7, (3, 3),
            strides=(2, 2),
            padding='same',
            activation=heaviside,
            use_bias=True,
        ),
        layers.BatchNormalization(scale=False),

        # Conv block 2: 7→15 filters, stride=2 (16→8)
        TernaryConv2D(
            15, (3, 3),
            strides=(2, 2),
            padding='same',
            activation=heaviside,
            use_bias=True,
        ),
        layers.BatchNormalization(scale=False),

        # Conv block 3: 15→15 filters, stride=1 (8→8)
        TernaryConv2D(
            15, (3, 3),
            strides=(1, 1),
            padding='same',
            activation=heaviside,
            use_bias=True,
        ),
        layers.BatchNormalization(scale=False),

        # Classifier head
        layers.Flatten(),
        TernaryDense(64, activation=heaviside, use_bias=True),
        TernaryDense(NUM_CLASSES, activation=heaviside, use_bias=True),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['accuracy'],
    )
    return model


# ============================================================
# Training
# ============================================================
def train_and_save():
    """Load data, train model, save to H5."""
    print("Loading dataset...")
    x_train, y_train, x_val, y_val = get_data_splits(
        IMG_DIR, LBL_DIR, target_size=IMAGE_SIZE
    )
    print(f"Dataset loaded. Train: {len(x_train)}, Val: {len(x_val)}")
    print(f"Data shape: {x_train.shape}, unique values: {np.unique(x_train)}")

    # Convert labels to one-hot for MSE loss
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    model = build_ternary_model()
    model.summary()

    print("\nStarting training...")
    history = model.fit(
        x_train, y_train_oh,
        epochs=EPOCHS,
        validation_data=(x_val, y_val_oh),
        batch_size=BATCH_SIZE,
    )

    print(f"\nSaving model to {MODEL_PATH}")
    # Create an uncompiled clone to avoid Keras 3 serialization issues with losses/metrics
    # Placer only needs architecture and weights.
    model_for_placer = tf.keras.models.clone_model(model)
    model_for_placer.set_weights(model.get_weights())
    model_for_placer.save(MODEL_PATH, save_format='h5')

    # Nuclear fix: Manually remove training_config from H5 to ensure it's uncompiled
    import h5py
    with h5py.File(MODEL_PATH, 'a') as f:
        if 'training_config' in f.attrs:
            del f.attrs['training_config']
    print("Cleaned H5 metadata (removed training_config).")

    return model, x_val, y_val, y_val_oh, history


# ============================================================
# Conversion: CNN → SNN
# ============================================================
def convert_to_snn():
    """Convert trained Keras model to SNN config via Placer."""
    print("\n" + "=" * 60)
    print("Converting ANN → SNN via Placer...")
    print("=" * 60)

    import knp_ann2snn
    pkg_path = os.path.dirname(knp_ann2snn.__file__)
    hw_config_path = os.path.join(
        pkg_path, 'placer_build', 'resources', 'hw_config.yaml'
    )

    placer = Placer(
        path_to_model=MODEL_PATH,
        input_ns_mode=False,
        hw_config_path=hw_config_path,
    )

    placer.run_placement(SNN_CONFIG_PATH)
    print(f"SNN config saved to: {SNN_CONFIG_PATH}")

    if not os.path.exists(SNN_CONFIG_PATH):
        placer.save_config(SNN_CONFIG_PATH)


# ============================================================
# Evaluation: CNN
# ============================================================
def evaluate_cnn(model, x_test, y_test):
    """
    Evaluate CNN accuracy and per-sample latency on CPU.

    Returns dict with metrics.
    """
    print("\n" + "=" * 60)
    print("Evaluating CNN (TF on CPU)...")
    print("=" * 60)

    y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Accuracy
    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"CNN Accuracy: {acc:.4f}")

    # Per-sample latency
    latencies = []
    for i in range(len(x_test)):
        sample = x_test[i:i + 1]
        t0 = time.perf_counter()
        pred = model.predict(sample, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        'accuracy': float(acc),
        'loss': float(loss),
        'latency_median_ms': float(np.median(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_mean_ms': float(np.mean(latencies)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': len(x_test),
    }

    print(f"  Latency (median): {results['latency_median_ms']:.2f} ms")
    print(f"  Latency (p95):    {results['latency_p95_ms']:.2f} ms")
    print(f"  Latency (p99):    {results['latency_p99_ms']:.2f} ms")
    print(f"  Throughput:       {results['throughput_samples_per_sec']:.1f} samples/s")

    return results


# ============================================================
# Evaluation: SNN
# ============================================================
def evaluate_snn(x_test, y_test, num_ticks=100, num_samples=None):
    """
    Evaluate SNN accuracy and per-sample latency on Altai golden model.

    Parameters
    ----------
    x_test : np.ndarray
        Binary test images, shape (N, H, W, 1).
    y_test : np.ndarray
        Integer class labels.
    num_ticks : int
        Number of ticks for SNN simulation.
    num_samples : int or None
        Number of samples to evaluate (None = all).

    Returns dict with metrics.
    """
    print("\n" + "=" * 60)
    print(f"Evaluating SNN (Altai Golden Model, {num_ticks} ticks)...")
    print("=" * 60)

    config_to_load = SNN_CONFIG_PATH
    if not os.path.exists(config_to_load):
        print(f"ERROR: SNN config not found at {config_to_load}")
        return None

    altai = Altai()
    altai.build(config_path=Path(config_to_load), inference_type='gm')

    if num_samples is None:
        num_samples = len(x_test)
    num_samples = min(num_samples, len(x_test))

    subset_indices = np.random.choice(
        len(x_test), num_samples, replace=False
    )

    correct = 0
    latencies = []

    for i in subset_indices:
        # Altai expects int32 spikes
        input_data = x_test[i].astype(np.int32)

        t0 = time.perf_counter()
        altai.prepare_spikes(input_data)
        altai.start_ticks(ticks=num_ticks)
        spikes = altai.get_spikes()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        # Decode output spikes
        prediction = 0
        if len(spikes) > 0:
            flat = spikes.flatten().astype(int)
            counts = np.bincount(flat, minlength=NUM_CLASSES)
            prediction = int(np.argmax(counts))

        if prediction == int(y_test[i]):
            correct += 1

        altai.clear_input()

    latencies = np.array(latencies)
    acc = correct / num_samples

    results = {
        'accuracy': float(acc),
        'latency_median_ms': float(np.median(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'latency_mean_ms': float(np.mean(latencies)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': num_samples,
        'num_ticks': num_ticks,
    }

    print(f"  SNN Accuracy:     {results['accuracy']:.4f}")
    print(f"  Latency (median): {results['latency_median_ms']:.2f} ms")
    print(f"  Latency (p95):    {results['latency_p95_ms']:.2f} ms")
    print(f"  Latency (p99):    {results['latency_p99_ms']:.2f} ms")
    print(f"  Throughput:       {results['throughput_samples_per_sec']:.1f} samples/s")

    return results


# ============================================================
# Main pipeline
# ============================================================
if __name__ == "__main__":
    # 1. Train
    model, x_val, y_val, y_val_oh, history = train_and_save()

    # 2. Convert CNN → SNN
    convert_to_snn()

    # 3. Evaluate CNN
    cnn_results = evaluate_cnn(model, x_val, y_val)

    # 4. Evaluate SNN
    snn_results = evaluate_snn(x_val, y_val, num_ticks=100)

    # 5. Summary
    print("\n" + "=" * 60)
    print("COMPARISON: CNN vs SNN")
    print("=" * 60)
    if snn_results:
        print(f"{'Metric':<30} {'CNN':>12} {'SNN':>12}")
        print("-" * 54)
        print(f"{'Accuracy':<30} {cnn_results['accuracy']:>11.4f} {snn_results['accuracy']:>11.4f}")
        print(f"{'Latency median (ms)':<30} {cnn_results['latency_median_ms']:>11.2f} {snn_results['latency_median_ms']:>11.2f}")
        print(f"{'Latency p95 (ms)':<30} {cnn_results['latency_p95_ms']:>11.2f} {snn_results['latency_p95_ms']:>11.2f}")
        print(f"{'Latency p99 (ms)':<30} {cnn_results['latency_p99_ms']:>11.2f} {snn_results['latency_p99_ms']:>11.2f}")
        print(f"{'Throughput (samples/s)':<30} {cnn_results['throughput_samples_per_sec']:>11.1f} {snn_results['throughput_samples_per_sec']:>11.1f}")
