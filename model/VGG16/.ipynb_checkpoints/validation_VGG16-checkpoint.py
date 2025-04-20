import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def validate_model():
    # Hardware configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validation data configuration
    val_dir = "./mygo-9517/archive/imbalanced/test"
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and loader
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model initialization
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze feature parameters
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modify classifier
    num_classes = len(val_dataset.classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    # Load trained weights
    model_path = "vgg16_model_Aerial_Landscapeswss_imbalanced.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Initialize metrics
    all_preds = []
    all_labels = []
    all_probs = []
    running_corrects = 0

    # Validation loop
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Get class probabilities
            
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    val_acc = running_corrects.double() / len(val_dataset)
    sk_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='macro')
    val_recall = recall_score(all_labels, all_preds, average='macro')
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    # Print comprehensive results
    print(f"\nValidation Metrics:")
    print(f"Manual Accuracy: {val_acc:.4f}")
    print(f"Sklearn Accuracy: {sk_acc:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    print(f"ROC-AUC: {val_roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

if __name__ == "__main__":
    validate_model()