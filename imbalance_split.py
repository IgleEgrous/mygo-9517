import os
import shutil
import numpy as np

# 设置路径和参数
data_dir = "./archive/Aerial_Landscapes"  # 原始数据目录（每个类别有800张图像）
output_dir = "./archive/imbalanced"  # 输出目录
test_ratio = 0.2
seed = 42

# 定义长尾分布参数
n_classes = 15  # 假设实际数据集有15个类别
max_samples = 800  # 第一个类别的样本数
min_samples = 50   # 最后一个类别的样本数
step = (max_samples - min_samples) // (n_classes - 1)  # 每类递减样本数

# 获取所有类别文件夹并按字母排序（确保顺序一致性）
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
assert len(class_names) == n_classes, "实际类别数量与预期不符！"

# 生成每个类别的目标样本数（从800递减到50）
class_samples = [max(max_samples - i * step, min_samples) for i in range(n_classes)]

# 创建输出目录
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

np.random.seed(seed)

# 遍历每个类别，按长尾分布采样
for idx, (class_name, target) in enumerate(zip(class_names, class_samples)):
    src_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(src_dir):
        continue

    # 获取所有图像文件名
    all_images = [f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))]
    
    # 随机选择目标数量的图像（确保不超过原始数量）
    selected_images = np.random.choice(all_images, size=target, replace=False)
    
    # 划分训练集和测试集
    split_idx = int(target * (1 - test_ratio))
    train_files = selected_images[:split_idx]
    test_files = selected_images[split_idx:]
    
    # 复制文件到目标目录
    for split, files in [("train", train_files), ("test", test_files)]:
        dest_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dest_dir, f))