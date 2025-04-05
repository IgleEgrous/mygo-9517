import os
import shutil
import numpy as np

# Paths and parameters
data_dir = "./archive/Aerial_Landscapes"  # Original dataset directory
output_dir = "./archive/imbalanced"  # Output directory
test_ratio = 0.2
seed = 42

# Long-tail parameters
n_classes = 15
max_samples = 800  # First class sample count
min_samples = 50   # Last class sample count
decrement_step = (max_samples - min_samples) // (n_classes - 1)  # Step size between classes

# Get sorted class directories
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
assert len(class_names) == n_classes, "Class count mismatch!"

# Generate target samples per class
class_samples = [max(max_samples - i * decrement_step, min_samples) for i in range(n_classes)]

# Create output directories
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

np.random.seed(seed)

# Process each class
for idx, (class_name, target) in enumerate(zip(class_names, class_samples)):
    src_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(src_path):
        continue

    # List all images
    all_images = [f for f in os.listdir(src_path) if f.endswith((".jpg", ".png"))]
    
    # Randomly select target samples
    selected_images = np.random.choice(all_images, size=target, replace=False)
    
    # Split into train/test
    split_idx = int(target * (1 - test_ratio))
    train_files = selected_images[:split_idx]
    test_files = selected_images[split_idx:]
    
    # Copy files
    for split, files in [("train", train_files), ("test", test_files)]:
        dest_path = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_path, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(src_path, f), os.path.join(dest_path, f))