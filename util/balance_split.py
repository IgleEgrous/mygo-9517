import os
import shutil
from sklearn.model_selection import train_test_split

# Paths and parameters
data_dir = "./mygo-9517/archive/Aerial_Landscapes"  # Original dataset directory (800 images per class)
output_dir = "./mygo-9517/archive/balanced"  # Output directory
test_ratio = 0.2
val_ratio = 0.25  # 25% of the remaining 80% (i.e., 20% of total data)
seed = 42

# Create directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Process each class
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # List all images
    images = [img for img in os.listdir(class_path) if img.endswith((".jpg", ".png"))]
    
    # First split: 80% trainval, 20% test
    trainval_files, test_files = train_test_split(
        images, test_size=test_ratio, random_state=seed
    )
    
    # Second split: 75% train, 25% val (from trainval)
    train_files, val_files = train_test_split(
        trainval_files, test_size=val_ratio, random_state=seed
    )
    
    # Copy files to target directories
    for split, files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files)
    ]:
        dest_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(dest_dir, f))
