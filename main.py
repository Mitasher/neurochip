import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from knp_ann2snn.altainn.ternary_tf2 import TernaryConv2D, TernaryDense
from knp_ann2snn import Placer
from knp_ann2snn.python_altai import Altai
from pathlib import Path
import time
from data_loader import get_data_splits

# Configuration
IMAGE_SIZE = (64, 64)
MODEL_PATH = 'helmet_ternary_model.h5'
SNN_CONFIG_PATH = 'helmet_snn_config.yaml'
IMG_DIR = "helmets.yolov8/train/images"
LBL_DIR = "helmets.yolov8/train/labels"

def build_ternary_model():
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5)
    
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        TernaryConv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
        layers.MaxPooling2D((2, 2)),
        TernaryConv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=initializer),
        layers.Flatten(),
        TernaryDense(32, activation='relu', kernel_initializer=initializer),
        TernaryDense(2, activation='softmax', kernel_initializer=initializer)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_save():
    print("Loading dataset...")
    x_train, y_train, x_val, y_val = get_data_splits(IMG_DIR, LBL_DIR, target_size=IMAGE_SIZE)
    print(f"Dataset loaded. Balanced Train: {len(x_train)}, Val: {len(x_val)}")
    
    model = build_ternary_model()
    model.summary()
    
    print("Starting training...")
    model.fit(x_train, y_train, epochs=15, validation_data=(x_val, y_val), batch_size=32)
    
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    return x_val, y_val

def convert_to_snn():
    print("Converting ANN to SNN...")
    import knp_ann2snn
    pkg_path = os.path.dirname(knp_ann2snn.__file__)
    hw_config_path = os.path.join(pkg_path, 'placer_build', 'resources', 'hw_config.yaml')
    
    # Correct Placer initialization: Placer(path_to_model, ..., hw_config_path=hw_config_path)
    placer = Placer(MODEL_PATH, input_ns_mode=False, hw_config_path=hw_config_path)
    
    # run_placement expects output_path
    placer.run_placement(SNN_CONFIG_PATH)
    
    print(f"Saving SNN config to {SNN_CONFIG_PATH}")
    if not os.path.exists(SNN_CONFIG_PATH):
        placer.save_config(SNN_CONFIG_PATH)

def evaluate_ann(model, x_test, y_test):
    print("\nEvaluating ANN...")
    start_time = time.time()
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    end_time = time.time()
    print(f"ANN Accuracy: {acc:.4f}")
    print(f"ANN Inference time (total for {len(x_test)} samples): {end_time - start_time:.4f}s")
    return acc

def evaluate_snn(x_test, y_test, num_samples=20):
    print(f"\nEvaluating SNN (on {num_samples} samples)...")
    config_to_load = SNN_CONFIG_PATH
    if not os.path.exists(config_to_load) and os.path.exists(config_to_load + ".yaml"):
        config_to_load += ".yaml"
        
    altai = Altai()
    altai.build(config_path=Path(config_to_load), inference_type='gm')
    
    correct = 0
    total_time = 0
    
    subset_indices = np.random.choice(len(x_test), min(num_samples, len(x_test)), replace=False)
    
    for i in subset_indices:
        input_data = x_test[i:i+1]
        
        start_time = time.time()
        altai.prepare_spikes(input_data)
        altai.start_ticks(ticks=100)
        spikes = altai.get_spikes()
        end_time = time.time()
        
        prediction = np.argmax(spikes)
        if prediction == int(y_test[i]):
            correct += 1
        
        total_time += (end_time - start_time)
        altai.clear_input()
        
    acc = correct / len(subset_indices)
    print(f"SNN Accuracy: {acc:.4f}")
    print(f"SNN Inference time (average per sample): {total_time / len(subset_indices):.4f}s")
    return acc

if __name__ == "__main__":
    # 1. Train
    x_val, y_val = train_and_save()
    
    # 2. Convert
    convert_to_snn()
    
    # 3. Evaluate ANN
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'TernaryConv2D': TernaryConv2D,
        'TernaryDense': TernaryDense
    })
    evaluate_ann(model, x_val, y_val)
    
    # 4. Evaluate SNN
    evaluate_snn(x_val, y_val, num_samples=20)
