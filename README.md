# Aerial Scene Classification with ResNet50 and Grad-CAM

---

## Table of Contents  
- [Project Overview](#project-overview)  
- [Key Features](#key-features)  
- [Installation](#installation)  
- [Data Preparation](#data-preparation)  
- [Training & Evaluation](#training--evaluation)  
- [Interpretability Analysis](#interpretability-analysis)  
- [Example Results](#example-results)  
- [License](#license)  

---

## Project Overview  
This project implements an aerial scene classification system using **ResNet50** and **Grad-CAM** for model interpretability. It includes:  
1. **Data augmentation** to enhance generalization.  
2. **Transfer learning** with pre-trained ResNet50.  
3. **Visual explanations** via Grad-CAM attention maps.  
4. **Robustness testing** under noise and occlusion.  

---

## Key Features  
### Data Augmentation  
```python
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## Installation
### Dependencies
```python
pip install torch torchvision opencv-python albumentations matplotlib numpy
```
### Clone Repository
```python
git clone https://github.com/IgleEgrous/mygo-9517?tab=readme-ov-file
```

---

## Data Preparation
### Dataset Structure
archive/
  ├── balanced
    ├── train/
    │   ├── airport/
    │   ├── forest/
    │   └── ...
    ├── val/
    └── test/
  ├── imbalanced
### Download Dataset
---

## Project Overview 
### Dependencies

---

## Project Overview 
### Dependencies

---

## Project Overview 
### Dependencies

---

## Project Overview 
### Dependencies