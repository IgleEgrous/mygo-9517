import os
import shutil
import numpy as np

# 设置路径和参数
data_dir = "./archive/Aerial_Landscapes"  # 原始数据目录
output_dir = "./archive/imbalanced"  # 输出目录
test_ratio = 0.2
seed = 42

# 定义长尾分布：类别1到15的样本数分别为800, 700, 600,...,50
n_classes = 15
max_samples = 800
min_samples = 50
step = (max_samples - min_samples) // (n_classes - 1)  # 每类减少50

# 生成每个类别的目标样本数
class_samples = [max_samples - i * step for i in range(n_classes)]
class_names = [f"class_{i+1}" for i in range(n_classes)]

# 创建输出目录
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

np.random.seed(seed)

for idx, (class_name, target_samples) in enumerate(zip(class_names, class_samples)):
    src_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(src_dir):
        continue
    
    # 从原始800张中随机选取目标样本数
    all_images = os.listdir(src_dir)
    selected_images = np.random.choice(all_images, target_samples, replace=False)
    
    # 按80-20划分
    split_idx = int(target_samples * (1 - test_ratio))
    train_files = selected_images[:split_idx]
    test_files = selected_images[split_idx:]
    
    # 复制文件到目标目录
    for split, files in [("train", train_files), ("test", test_files)]:
        dest_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))