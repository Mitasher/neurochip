import os
import cv2
import time
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from tensorflow.keras import layers

from knp_ann2snn.altainn.ternary_tf2 import TernaryConv2D, TernaryDense, heaviside
from knp_ann2snn.altainn.ternary_tf2.ops import Clip
from tensorflow.keras.saving import register_keras_serializable
from knp_ann2snn.python_altai import Altai


# ============================================================
# Keras Patches & Registration
# ============================================================
def apply_keras_patches():
    """
    Applies Keras 3 compatibility patches and registers custom objects.
    Call this once at the start of any script using the model.
    """
    # 1. Keras 2 to Keras 3 add_weight bridge
    _original_add_weight = layers.Layer.add_weight

    def _patched_add_weight(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            name = args[0]
            args = args[1:]
            kwargs['name'] = name
        return _original_add_weight(self, *args, **kwargs)

    layers.Layer.add_weight = _patched_add_weight

    # 2. Register custom objects for Keras 3
    register_keras_serializable(name="heaviside_mod")(heaviside)
    register_keras_serializable(name="heaviside")(heaviside)
    register_keras_serializable(name="TernaryConv2D")(TernaryConv2D)
    register_keras_serializable(name="TernaryDense")(TernaryDense)
    register_keras_serializable(name="Clip")(Clip)

    # 3. Safely handle Placer's bias lookup
    TernaryConv2D.bias = property(lambda self: None)
    TernaryDense.bias = property(lambda self: None)


# ============================================================
# Visualization
# ============================================================
def save_visual_results(orig_images, y_true, y_pred, model_type, save_dir="visual_results", num_samples=20):
    """
    Сохраняет предсказания модели в виде картинок.
    Зеленый текст - правильное предсказание, Красный - ошибка.
    """
    if orig_images is None:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    class_names = {0: 'Hat', 1: 'Person'} 

    indices = np.random.choice(len(orig_images), min(num_samples, len(orig_images)), replace=False)

    for i, idx in enumerate(indices):
        # Используем оригинальные цветные кропы 256x256 вместо бинарных 32x32
        img_bgr = orig_images[idx].copy()

        t_lbl = int(y_true[idx])
        p_lbl = int(y_pred[idx])
        
        true_name = class_names.get(t_lbl, str(t_lbl))
        pred_name = class_names.get(p_lbl, str(p_lbl))

        color = (0, 255, 0) if t_lbl == p_lbl else (0, 0, 255) 

        text = f"T: {true_name} | P: {pred_name}"
        cv2.putText(img_bgr, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        status = 'ok' if t_lbl == p_lbl else 'err'
        filename = os.path.join(save_dir, f"{model_type}_{status}_{i}.png")
        cv2.imwrite(filename, img_bgr)
        
    print(f"\n[+] Сохранено {num_samples} картинок с предсказаниями {model_type} в {save_dir}/")


# ============================================================
# Evaluation Logic
# ============================================================
def print_detailed_results(r):
    """Pretty-print benchmark results."""
    print(f"  {'Type:':<25} {r['type']}")
    print(f"  {'Accuracy:':<25} {r['accuracy']:.4f}")
    if 'loss' in r:
        print(f"  {'Loss:':<25} {r['loss']:.4f}")
    print(f"  {'Samples:':<25} {r['total_samples']}")
    print(f"  {'Latency median (ms):':<25} {r['latency_median_ms']:.2f}")
    print(f"  {'Latency mean (ms):':<25} {r['latency_mean_ms']:.2f}")
    if 'latency_std_ms' in r:
        print(f"  {'Latency std (ms):':<25} {r['latency_std_ms']:.2f}")
    if 'latency_p90_ms' in r:
        print(f"  {'Latency p90 (ms):':<25} {r['latency_p90_ms']:.2f}")
    print(f"  {'Latency p95 (ms):':<25} {r['latency_p95_ms']:.2f}")
    print(f"  {'Latency p99 (ms):':<25} {r['latency_p99_ms']:.2f}")
    print(f"  {'Throughput (sps):':<25} {r['throughput_samples_per_sec']:.1f}")
    if r.get('per_class_accuracy'):
        for cls, acc in r['per_class_accuracy'].items():
            print(f"  {'  Class ' + cls + ' accuracy:':<25} {acc:.4f}")


def evaluate_cnn(model, x_test, y_test, num_classes, warmup=0, orig_images=None):
    """
    Evaluate CNN accuracy and detailed latency stats.
    """
    print("\n" + "=" * 60)
    print("Evaluating CNN...")
    print("=" * 60)

    y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes)

    # Accuracy
    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)

    # Warmup
    for i in range(min(warmup, len(x_test))):
        model.predict(x_test[i:i + 1], verbose=0)

    # Per-sample latency
    latencies = []
    predictions = []
    
    for i in range(len(x_test)):
        sample = x_test[i:i + 1]
        t0 = time.perf_counter()
        pred = model.predict(sample, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        predictions.append(int(np.argmax(pred[0])))

    latencies = np.array(latencies)
    predictions = np.array(predictions)

    if orig_images is not None:
        save_visual_results(orig_images, y_test, predictions, model_type="CNN")

    per_class_correct = {}
    per_class_total = {}
    
    for cls in range(num_classes):
        mask = y_test == cls
        per_class_total[str(cls)] = int(mask.sum())
        per_class_correct[str(cls)] = int((predictions[mask] == cls).sum())

    results = {
        'type': 'CNN',
        'accuracy': float(acc),
        'loss': float(loss),
        'latency_median_ms': float(np.median(latencies)),
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': len(x_test),
        'per_class_accuracy': {
            k: per_class_correct[k] / per_class_total[k]
            if per_class_total[k] > 0 else 0.0
            for k in per_class_total
        },
    }

    print_detailed_results(results)
    return results


def evaluate_snn(x_test, y_test, snn_config_path, num_classes, num_ticks=100, warmup=0, num_samples=None, orig_images=None):
    """
    Evaluate SNN inference on Altai golden model.
    """
    print("\n" + "=" * 60)
    print(f"Evaluating SNN (Altai Golden Model, {num_ticks} ticks)...")
    print("=" * 60)

    if not os.path.exists(snn_config_path):
        print(f"ERROR: SNN config not found: {snn_config_path}")
        return None

    altai = Altai()
    altai.build(config_path=str(Path(snn_config_path)), inference_type='gm')

    if num_samples is None:
        num_samples = len(x_test)
    num_samples = min(num_samples, len(x_test))

    indices = np.random.choice(len(x_test), num_samples, replace=False)

    # Warmup
    for i in range(min(warmup, num_samples)):
        idx = indices[i]
        inp = x_test[idx].astype(np.int32)
        altai.prepare_spikes(inp)
        altai.start_ticks(ticks=num_ticks)
        altai.get_spikes()
        altai.clear_input()

    # Actual benchmark
    correct = 0
    latencies = []
    predictions = []

    for idx in tqdm(indices, desc="Simulating SNN"):
        inp = x_test[idx].astype(np.int32)

        t0 = time.perf_counter()
        altai.prepare_spikes(inp)
        altai.start_ticks(ticks=num_ticks)
        spikes = altai.get_spikes()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)

        prediction = 0
        if len(spikes) > 0:
            flat = spikes.flatten().astype(int)
            valid_spikes = flat[flat >= 0]
            
            if len(valid_spikes) > 0:
                counts = np.bincount(valid_spikes, minlength=num_classes)
                prediction = int(np.argmax(counts))

        predictions.append(prediction)
        if prediction == int(y_test[idx]):
            correct += 1

        altai.clear_input()

    latencies = np.array(latencies)
    predictions = np.array(predictions)
    
    if orig_images is not None:
        save_visual_results(orig_images[indices], y_test[indices], predictions, model_type="SNN")
    
    acc = correct / num_samples

    per_class_correct = {}
    per_class_total = {}
    for cls in range(num_classes):
        mask = y_test[indices] == cls
        per_class_total[str(cls)] = int(mask.sum())
        per_class_correct[str(cls)] = int((predictions[mask] == cls).sum())

    results = {
        'type': 'SNN',
        'accuracy': float(acc),
        'num_ticks': num_ticks,
        'latency_median_ms': float(np.median(latencies)),
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': num_samples,
        'per_class_accuracy': {
            k: per_class_correct[k] / per_class_total[k]
            if per_class_total[k] > 0 else 0.0
            for k in per_class_total
        },
    }

    print_detailed_results(results)
    return results


def print_comparison(cnn_r, snn_r):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("COMPARISON: CNN vs SNN")
    print("=" * 60)
    print(f"{'Metric':<30} {'CNN':>12} {'SNN':>12}")
    print("-" * 54)

    metrics = [
        ('Accuracy', 'accuracy', '.4f'),
        ('Latency median (ms)', 'latency_median_ms', '.2f'),
        ('Latency p95 (ms)', 'latency_p95_ms', '.2f'),
        ('Latency p99 (ms)', 'latency_p99_ms', '.2f'),
        ('Throughput (sps)', 'throughput_samples_per_sec', '.1f'),
    ]

    for label, key, fmt in metrics:
        cnn_val = cnn_r.get(key, 'N/A')
        snn_val = snn_r.get(key, 'N/A')
        cnn_str = f"{cnn_val:{fmt}}" if isinstance(cnn_val, (int, float)) else 'N/A'
        snn_str = f"{snn_val:{fmt}}" if isinstance(snn_val, (int, float)) else 'N/A'
        print(f"{label:<30} {cnn_str:>12} {snn_str:>12}")
