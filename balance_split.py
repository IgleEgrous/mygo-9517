import os
import shutil
from sklearn.model_selection import train_test_split

# Paths and parameters
data_dir = "./archive/Aerial_Landscapes"  # Original dataset directory (800 images per class)
output_dir = "./archive/balanced"  # Output directory
test_ratio = 0.2
seed = 42  # Random seed

# Create train/test directories
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# Process each class
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    
    # List all image files
    images = [img for img in os.listdir(class_path) if img.endswith((".jpg", ".png"))]
    
    # Split into train/test
    train_files, test_files = train_test_split(
        images, test_size=test_ratio, random_state=seed
    )
    
    # Copy files to target directories
    for split, files in [("train", train_files), ("test", test_files)]:
        dest_path = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_path, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_path, f), os.path.join(dest_path, f))