"""
Benchmark: CNN vs SNN performance comparison.

Standalone script that loads a trained model + SNN config
and produces comprehensive metrics for both.

Usage:
    python benchmark.py [--num-samples 100] [--num-ticks 100] [--warmup 5]
"""

import os
import cv2
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from knp_ann2snn.altainn.ternary_tf2 import TernaryConv2D, TernaryDense, heaviside
from knp_ann2snn.python_altai import Altai
from data_loader import get_data_splits

# === ДОБАВИТЬ ВОТ ЭТОТ БЛОК ===
from tensorflow.keras.saving import register_keras_serializable
from knp_ann2snn.altainn.ternary_tf2.ops import Clip

# Регистрируем функции активации
register_keras_serializable(name="heaviside_mod")(heaviside)
register_keras_serializable(name="heaviside")(heaviside)

# Регистрируем слои
register_keras_serializable(name="TernaryConv2D")(TernaryConv2D)
register_keras_serializable(name="TernaryDense")(TernaryDense)

# Регистрируем кастомное ограничение весов
register_keras_serializable(name="Clip")(Clip)
# ==============================


# ============================================================
# Configuration
# ============================================================
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 2
MODEL_PATH = 'helmet_ternary_model.h5'
SNN_CONFIG_PATH = 'helmet_snn_config.json'
IMG_DIR = "helmets.yolov8/train/images"
LBL_DIR = "helmets.yolov8/train/labels"
REPORT_PATH = 'benchmark_report.json'

def save_visual_results(x_data, y_true, y_pred, model_type, save_dir="visual_results", num_samples=20):
    """
    Сохраняет предсказания модели в виде картинок.
    Зеленый текст - правильное предсказание, Красный - ошибка.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Классы согласно вашей функции load_yolo_dataset
    class_names = {0: 'Hat', 1: 'Person'} 

    # Выбираем случайные индексы для сохранения
    indices = np.random.choice(len(x_data), min(num_samples, len(x_data)), replace=False)

    for i, idx in enumerate(indices):
        # x_data имеет значения {0.0, 1.0} и размер (H, W, 1)
        img = x_data[idx] 
        
        # Переводим обратно в 0-255 и конвертируем в 3 канала для цветного текста
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Увеличиваем картинку, чтобы текст был читаемым (оригинал 32x32 или 64x64 слишком мал)
        img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)

        t_lbl = int(y_true[idx])
        p_lbl = int(y_pred[idx])
        
        true_name = class_names.get(t_lbl, str(t_lbl))
        pred_name = class_names.get(p_lbl, str(p_lbl))

        # Логика цвета: BGR формат (Синий, Зеленый, Красный)
        color = (0, 255, 0) if t_lbl == p_lbl else (0, 0, 255) 

        text = f"T: {true_name} | P: {pred_name}"
        cv2.putText(img_bgr, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Формируем имя файла (ok/err позволяет быстро отсортировать ошибки в папке)
        status = 'ok' if t_lbl == p_lbl else 'err'
        filename = os.path.join(save_dir, f"{model_type}_{status}_{i}.png")
        cv2.imwrite(filename, img_bgr)
        
    print(f"\n[+] Сохранено {num_samples} картинок с предсказаниями {model_type} в {save_dir}/")



# ============================================================
# CNN Benchmark
# ============================================================
def benchmark_cnn(model, x_test, y_test, warmup=5):
    """
    Benchmark CNN inference: accuracy + detailed latency stats.
    """
    print("\n" + "=" * 60)
    print("Benchmarking CNN...")
    print("=" * 60)

    y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

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

    # СОХРАНЯЕМ КАРТИНКИ
    save_visual_results(x_test, y_test, predictions, model_type="CNN")

    # Confusion-style stats
    per_class_correct = {}
    per_class_total = {}
    
    for cls in range(NUM_CLASSES):
        mask = y_test == cls
        per_class_total[str(cls)] = int(mask.sum())
        per_class_correct[str(cls)] = int((predictions[mask] == cls).sum())

    results = {
        'type': 'CNN',
        'accuracy': float(acc),
        'loss': float(loss),
        'latency_median_ms': float(np.median(latencies)),
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p90_ms': float(np.percentile(latencies, 90)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': len(x_test),
        'warmup_samples': warmup,
        'per_class_accuracy': {
            k: per_class_correct[k] / per_class_total[k]
            if per_class_total[k] > 0 else 0.0
            for k in per_class_total
        },
    }

    _print_results(results)
    return results


# ============================================================
# SNN Benchmark
# ============================================================
def benchmark_snn(x_test, y_test, num_ticks=100, warmup=5, num_samples=None):
    """
    Benchmark SNN inference on Altai golden model.
    """
    print("\n" + "=" * 60)
    print(f"Benchmarking SNN ({num_ticks} ticks)...")
    print("=" * 60)

    if not os.path.exists(SNN_CONFIG_PATH):
        print(f"ERROR: SNN config not found: {SNN_CONFIG_PATH}")
        return None

    altai = Altai()
    altai.build(config_path=str(Path(SNN_CONFIG_PATH)), inference_type='gm')

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
            # Отфильтровываем отрицательные значения (индикаторы "тишины")
            valid_spikes = flat[flat >= 0]
            
            # Считаем бины только если остались валидные спайки
            if len(valid_spikes) > 0:
                counts = np.bincount(valid_spikes, minlength=NUM_CLASSES)
                prediction = int(np.argmax(counts))

        predictions.append(prediction)
        if prediction == int(y_test[idx]):
            correct += 1

        altai.clear_input()

    latencies = np.array(latencies)
    predictions = np.array(predictions)
    
    # СОХРАНЯЕМ КАРТИНКИ
    save_visual_results(x_test[indices], y_test[indices], predictions, model_type="SNN")
    
    acc = correct / num_samples

    # Per-class stats
    per_class_correct = {}
    per_class_total = {}
    for cls in range(NUM_CLASSES):
        mask = y_test[indices] == cls
        per_class_total[str(cls)] = int(mask.sum())
        per_class_correct[str(cls)] = int((predictions[mask] == cls).sum())

    results = {
        'type': 'SNN',
        'accuracy': float(acc),
        'num_ticks': num_ticks,
        'latency_median_ms': float(np.median(latencies)),
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'latency_p50_ms': float(np.percentile(latencies, 50)),
        'latency_p90_ms': float(np.percentile(latencies, 90)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'throughput_samples_per_sec': 1000.0 / float(np.mean(latencies)),
        'total_samples': num_samples,
        'warmup_samples': warmup,
        'per_class_accuracy': {
            k: per_class_correct[k] / per_class_total[k]
            if per_class_total[k] > 0 else 0.0
            for k in per_class_total
        },
    }

    _print_results(results)
    return results


# ============================================================
# Helpers
# ============================================================
def _print_results(r):
    """Pretty-print benchmark results."""
    print(f"  {'Type:':<25} {r['type']}")
    print(f"  {'Accuracy:':<25} {r['accuracy']:.4f}")
    print(f"  {'Samples:':<25} {r['total_samples']}")
    print(f"  {'Latency median (ms):':<25} {r['latency_median_ms']:.2f}")
    print(f"  {'Latency mean (ms):':<25} {r['latency_mean_ms']:.2f}")
    print(f"  {'Latency std (ms):':<25} {r['latency_std_ms']:.2f}")
    print(f"  {'Latency p90 (ms):':<25} {r['latency_p90_ms']:.2f}")
    print(f"  {'Latency p95 (ms):':<25} {r['latency_p95_ms']:.2f}")
    print(f"  {'Latency p99 (ms):':<25} {r['latency_p99_ms']:.2f}")
    print(f"  {'Throughput (sps):':<25} {r['throughput_samples_per_sec']:.1f}")
    if r.get('per_class_accuracy'):
        for cls, acc in r['per_class_accuracy'].items():
            print(f"  {'  Class ' + cls + ' accuracy:':<25} {acc:.4f}")


def print_comparison(cnn_r, snn_r):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 62)
    print("  COMPARISON: CNN vs SNN")
    print("=" * 62)
    print(f"  {'Metric':<30} {'CNN':>12} {'SNN':>12}")
    print("  " + "-" * 56)

    metrics = [
        ('Accuracy', 'accuracy', '.4f'),
        ('Latency median (ms)', 'latency_median_ms', '.2f'),
        ('Latency mean (ms)', 'latency_mean_ms', '.2f'),
        ('Latency p90 (ms)', 'latency_p90_ms', '.2f'),
        ('Latency p95 (ms)', 'latency_p95_ms', '.2f'),
        ('Latency p99 (ms)', 'latency_p99_ms', '.2f'),
        ('Throughput (sps)', 'throughput_samples_per_sec', '.1f'),
        ('Samples evaluated', 'total_samples', 'd'),
    ]

    for label, key, fmt in metrics:
        cnn_val = cnn_r.get(key, 'N/A')
        snn_val = snn_r.get(key, 'N/A')
        cnn_str = f"{cnn_val:{fmt}}" if isinstance(cnn_val, (int, float)) else 'N/A'
        snn_str = f"{snn_val:{fmt}}" if isinstance(snn_val, (int, float)) else 'N/A'
        print(f"  {label:<30} {cnn_str:>12} {snn_str:>12}")

    print("=" * 62)


def save_report(cnn_r, snn_r, path=REPORT_PATH):
    """Save benchmark results to JSON."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'image_size': IMAGE_SIZE,
            'num_channels': 1,
            'num_classes': NUM_CLASSES,
            'model_path': MODEL_PATH,
            'snn_config_path': SNN_CONFIG_PATH,
        },
        'cnn': cnn_r,
        'snn': snn_r,
    }
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CNN vs SNN benchmark')
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--num-ticks', type=int, default=40,
        help='Number of SNN simulation ticks (default: 100)'
    )
    parser.add_argument(
        '--warmup', type=int, default=5,
        help='Number of warmup iterations (default: 5)'
    )
    args = parser.parse_args()

    # Load data
    print("Loading validation data...")
    _, _, x_val, y_val = get_data_splits(IMG_DIR, LBL_DIR, target_size=IMAGE_SIZE)
    print(f"Loaded {len(x_val)} validation samples")

    # Load CNN model
    print("Loading CNN model...")
    custom_objects = {
        'TernaryConv2D': TernaryConv2D,
        'TernaryDense': TernaryDense,
        'heaviside_mod': heaviside,
    }
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    # Compile the model since it was saved uncompiled
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # use MSE or whatever main.py used
        metrics=['accuracy']
    )

    # Benchmark CNN
    cnn_results = benchmark_cnn(
        model, x_val, y_val,
        warmup=args.warmup,
    )

    # Benchmark SNN
    snn_results = benchmark_snn(
        x_val, y_val,
        num_ticks=args.num_ticks,
        warmup=args.warmup,
        num_samples=args.num_samples,
    )

    # Comparison
    if snn_results:
        print_comparison(cnn_results, snn_results)
        save_report(cnn_results, snn_results)
    else:
        print("\nSNN benchmark skipped (config not found).")
        save_report(cnn_results, {})



if __name__ == "__main__":
    main()
