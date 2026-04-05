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

from utils import apply_keras_patches, evaluate_cnn, evaluate_snn, print_comparison

# Применяем патчи Keras
apply_keras_patches()


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
    _, _, x_val, y_val, orig_x_train, orig_x_val = get_data_splits(IMG_DIR, LBL_DIR, target_size=IMAGE_SIZE)
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
    cnn_results = evaluate_cnn(
        model, x_val, y_val,
        num_classes=NUM_CLASSES,
        warmup=args.warmup,
        orig_images=orig_x_val
    )

    # Benchmark SNN
    snn_results = evaluate_snn(
        x_val, y_val,
        snn_config_path=SNN_CONFIG_PATH,
        num_classes=NUM_CLASSES,
        num_ticks=args.num_ticks,
        warmup=args.warmup,
        num_samples=args.num_samples,
        orig_images=orig_x_val
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
