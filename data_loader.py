import os
import cv2
import numpy as np
import random
from tqdm import tqdm

def load_yolo_dataset(img_dir, label_dir, target_size=(64, 64), max_samples=5000):
    images = []
    labels = []
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    count = 0
    for img_file in tqdm(img_files, desc="Loading data"):
        if count >= max_samples:
            break
            
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            cls = int(parts[0])
            # Filter classes: 0=hat, 1=person. We take both.
            if cls not in [0, 1]:
                continue
                
            x_c, y_c, bw, bh = map(float, parts[1:5])
            
            # Convert normalized to pixel coordinates
            x1 = int((x_c - bw/2) * w)
            y1 = int((y_c - bh/2) * h)
            x2 = int((x_c + bw/2) * w)
            y2 = int((y_c + bh/2) * h)
            
            # Clip to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, target_size)
            # Normalize to [-1, 1] to match ternary weights range
            crop = (crop.astype(np.float32) / 127.5) - 1.0
            
            images.append(crop)
            labels.append(cls)
            count += 1
            if count >= max_samples:
                break
                
    # Separate classes
    images = np.array(images)
    labels = np.array(labels)
    
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]
    
    min_count = min(len(idx0), len(idx1))
    
    selected_idx0 = np.random.choice(idx0, min_count, replace=False)
    selected_idx1 = np.random.choice(idx1, min_count, replace=False)
    
    balanced_idx = np.concatenate([selected_idx0, selected_idx1])
    np.random.shuffle(balanced_idx)
    
    return images[balanced_idx], labels[balanced_idx]

def get_data_splits(img_dir, label_dir, target_size=(64, 64), split_ratio=0.8):
    x, y = load_yolo_dataset(img_dir, label_dir, target_size)
    
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    
    split_idx = int(len(x) * split_ratio)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

if __name__ == "__main__":
    # Test loading
    IMG_DIR = "helmets.yolov8/train/images"
    LBL_DIR = "helmets.yolov8/train/labels"
    x_train, y_train, x_val, y_val = get_data_splits(IMG_DIR, LBL_DIR)
    print(f"Loaded {len(x_train)} train samples and {len(x_val)} val samples")
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_val.npy", x_val)
    np.save("y_val.npy", y_val)
