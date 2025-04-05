import os
import shutil
from sklearn.model_selection import train_test_split

# 设置路径和参数
data_dir = "./archive/Aerial_Landscapes"  # 原始数据目录（每个类别有800张图像）
output_dir = "./archive/balanced"  # 输出目录
test_ratio = 0.2
seed = 42  # 随机种子

# 创建训练集和测试集目录
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# 遍历每个类别文件夹
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    # 获取所有图像文件名
    images = os.listdir(class_dir)
    images = [img for img in images if img.endswith((".jpg", ".png"))]
    
    # 按80-20划分
    train_files, test_files = train_test_split(
        images, test_size=test_ratio, random_state=seed
    )
    
    # 创建类别子目录并复制文件
    for split, files in [("train", train_files), ("test", test_files)]:
        dest_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(class_dir, f), os.path.join(dest_dir, f))