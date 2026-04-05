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

from knp_ann2snn.altainn.ternary_tf2 import TernaryConv2D, TernaryDense, heaviside
from knp_ann2snn import Placer
from pathlib import Path
import time
from data_loader import get_data_splits
from tqdm import tqdm

from utils import apply_keras_patches, evaluate_cnn, evaluate_snn, print_comparison

# Применяем патчи Keras
apply_keras_patches()

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
    inputs = layers.Input(shape=(*IMAGE_SIZE, NUM_CHANNELS))

    # Слой 1: 32x32 -> 16x16
    x = TernaryConv2D(
        16, (3, 3), strides=(2, 2), padding='same', 
        activation=heaviside, use_bias=False
    )(inputs)

    # Слой 2: 16x16 -> 8x8
    x = TernaryConv2D(
        32, (3, 3), strides=(2, 2), padding='same', 
        activation=heaviside, use_bias=False
    )(x)

    # Слой 3: 8x8 -> 4x4 (сжимаем, чтобы влезть в лимиты Placer)
    x = TernaryConv2D(
        16, (3, 3), strides=(2, 2), padding='same', 
        activation=heaviside, use_bias=False
    )(x)

    # Голова классификатора
    x = layers.Flatten()(x)
    x = TernaryDense(128, activation=heaviside, use_bias=False)(x)
    outputs = TernaryDense(NUM_CLASSES, activation=heaviside, use_bias=False)(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse', # Возвращаем MSE для стабильности с Heaviside
        metrics=['accuracy'],
    )
    return model

# ============================================================
# Training
# ============================================================
def train_and_save():
    """Load data, train model, save to H5."""
    print("Loading dataset...")
    x_train, y_train, x_val, y_val, orig_x_train, orig_x_val = get_data_splits(
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
    
    # Коллбэки для контроля обучения
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=20, # ждем 20 эпох, если нет улучшений - стоп
        restore_best_weights=True # откатываемся к лучшей эпохе
    )
    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, # уменьшаем LR в 2 раза
        patience=5, # если loss не падает 5 эпох
        min_lr=1e-5
    )

    history = model.fit(
        x_train, y_train_oh,
        epochs=100, # Увеличили со 10 до 100
        validation_data=(x_val, y_val_oh),
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, lr_decay] # Добавили коллбэки
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

    return model, x_val, y_val, orig_x_val, y_val_oh, history


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
# Main pipeline
# ============================================================
if __name__ == "__main__":
    # 1. Train
    model, x_val, y_val, orig_x_val, y_val_oh, history = train_and_save()

    # 2. Convert CNN → SNN
    convert_to_snn()

    # 3. Evaluate CNN
    cnn_results = evaluate_cnn(model, x_val, y_val, num_classes=NUM_CLASSES, orig_images=orig_x_val)

    # 4. Evaluate SNN
    import numpy as np

    snn_results = evaluate_snn(x_val, y_val, SNN_CONFIG_PATH, NUM_CLASSES, num_ticks=100, orig_images=orig_x_val)

    # 5. Summary
    if snn_results:
        print_comparison(cnn_results, snn_results)
