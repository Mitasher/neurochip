import os
import cv2
import numpy as np
from tqdm import tqdm


def load_yolo_dataset(img_dir, label_dir, target_size=(64, 64), max_samples=5000,
                      binarize_threshold=127):
    """
    Load YOLO-format dataset and produce binarized single-channel crops.

    AltAI-2 requirements:
      - Input data must be binary {0, 1}
      - Single channel (grayscale → threshold binarization)

    Parameters
    ----------
    img_dir : str
        Path to directory with images.
    label_dir : str
        Path to directory with YOLO-format label .txt files.
    target_size : tuple
        (width, height) to resize crops to.
    max_samples : int
        Maximum number of samples to load.
    binarize_threshold : int
        Pixel threshold for binarization (0-255).
        Pixels > threshold → 1, else → 0.

    Returns
    -------
    images : np.ndarray, shape (N, H, W, 1), dtype float32, values {0.0, 1.0}
    labels : np.ndarray, shape (N,), dtype int
    orig_images : np.ndarray, shape (N, 256, 256, 3), dtype uint8 (original colored crops)
    """
    images = []
    labels = []
    orig_images = []

    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    count = 0
    for img_file in tqdm(img_files, desc="Loading data"):
        if count >= max_samples:
            break

        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(
            label_dir, os.path.splitext(img_file)[0] + ".txt"
        )

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
            # Filter classes: 0=hat, 1=person
            if cls not in [0, 1]:
                continue

            x_c, y_c, bw, bh = map(float, parts[1:5])

            # Convert normalized to pixel coordinates
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)

            # Clip to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            
            # Keep metadata for full image visualization
            meta = {
                'img_path': img_path,
                'bbox': (x1, y1, x2, y2)
            }
            
            crop = cv2.resize(crop, target_size)

            # --- AltAI-2: бинаризация в 1 канал ---
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, binarize_threshold, 1, cv2.THRESH_BINARY
            )
            # shape: (H, W) → (H, W, 1)
            binary = binary[:, :, np.newaxis].astype(np.float32)

            images.append(binary)
            labels.append(cls)
            orig_images.append(meta)
            count += 1
            if count >= max_samples:
                break

    images = np.array(images)
    labels = np.array(labels)
    orig_images = np.array(orig_images, dtype=object)

    # Balance classes
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    min_count = min(len(idx0), len(idx1))
    if min_count == 0:
        print("WARNING: One of the classes has 0 samples!")
        return images, labels, orig_images

    selected_idx0 = np.random.choice(idx0, min_count, replace=False)
    selected_idx1 = np.random.choice(idx1, min_count, replace=False)

    balanced_idx = np.concatenate([selected_idx0, selected_idx1])
    np.random.shuffle(balanced_idx)

    return images[balanced_idx], labels[balanced_idx], orig_images[balanced_idx]


def get_data_splits(img_dir, label_dir, target_size=(64, 64), split_ratio=0.8):
    """Load dataset and split into train/val."""
    x, y, orig_x = load_yolo_dataset(img_dir, label_dir, target_size)

    indices = np.arange(len(x))
    np.random.shuffle(indices)

    split_idx = int(len(x) * split_ratio)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], orig_x[train_idx], orig_x[val_idx]


if __name__ == "__main__":
    # Test loading
    IMG_DIR = "helmets.yolov8/train/images"
    LBL_DIR = "helmets.yolov8/train/labels"
    x_train, y_train, x_val, y_val, orig_x_train, orig_x_val = get_data_splits(IMG_DIR, LBL_DIR)
    print(f"Loaded {len(x_train)} train samples and {len(x_val)} val samples")
    print(f"Data shape: {x_train.shape}, dtype: {x_train.dtype}")
    print(f"Original Data shape: {orig_x_train.shape}, dtype: {orig_x_train.dtype}")
    print(f"Unique values: {np.unique(x_train)}")
    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)
    np.save("x_val.npy", x_val)
    np.save("y_val.npy", y_val)
    np.save("orig_x_train.npy", orig_x_train, allow_pickle=True)
    np.save("orig_x_val.npy", orig_x_val, allow_pickle=True)
