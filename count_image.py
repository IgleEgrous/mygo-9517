import os
import numpy as np
from sklearn.model_selection import train_test_split

def check_counts(output_dir):
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        print(f"\n{split.upper()} SET:")
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            count = len(os.listdir(class_dir))
            print(f"  {class_name}: {count} images")

# Check balanced dataset
check_counts("./mygo-9517/archive/balanced")

# Check imbalanced dataset
check_counts("./mygo-9517/archive/imbalanced")