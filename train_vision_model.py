import os
import json
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from resnet18_model import build_resnet18

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_image_train(img):
    img = img.resize((224, 224))
    
    # Anatomically valid ultrasound augmentations (NO vertical flip!)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        angle = random.uniform(-8, 8)
        img = img.rotate(angle)

    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if random.random() > 0.5:
        factor = random.uniform(0.9, 1.1)
        arr = np.clip(arr * factor, 0.0, 1.0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor

def preprocess_image_val(img):
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor

def preprocess_image(img, is_train=False):
    if is_train:
        return preprocess_image_train(img)
    else:
        return preprocess_image_val(img)

class UltrasoundDataset(Dataset):
    def __init__(self, image_paths, labels, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))

        if self.is_train:
            tensor = preprocess_image_train(img)
        else:
            tensor = preprocess_image_val(img)
            
        label = self.labels[idx]
        return tensor, label

def load_dataset_paths(dataset_dir, max_samples_per_class=700):
    classes = ['F0', 'F1', 'F2', 'F3', 'F4']
    image_paths = []
    labels = []

    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if os.path.exists(cls_dir):
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(valid_exts)]
            random.seed(42)
            if max_samples_per_class and len(files) > max_samples_per_class:
                files = random.sample(files, max_samples_per_class)
            image_paths.extend(files)
            labels.extend([label_idx] * len(files))

    return image_paths, labels, classes

def train_vision_pipeline(dataset_dir, models_dir, epochs=8, batch_size=32, lr=0.0003):
    os.makedirs(models_dir, exist_ok=True)
    set_seed(42)

    print("Loading dataset paths...")
    image_paths, labels, class_names = load_dataset_paths(dataset_dir, max_samples_per_class=700)
    print(f"Total samples: {len(image_paths)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = UltrasoundDataset(train_paths, train_labels, is_train=True)
    val_dataset = UltrasoundDataset(val_paths, val_labels, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet18(num_classes=len(class_names))

    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    try:
        print("Loading ImageNet pre-trained weights...")
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("ImageNet transfer learning weights loaded successfully!")
    except Exception as e:
        print(f"Could not load ImageNet weights ({e}), training from scratch.")

    model = model.to(device)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=np.array(train_labels)
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': lr * 2},
        {'params': model.layer4.parameters(), 'lr': lr},
        {'params': model.layer3.parameters(), 'lr': lr * 0.5},
        {'params': model.layer2.parameters(), 'lr': lr * 0.2},
        {'params': model.layer1.parameters(), 'lr': lr * 0.1},
        {'params': model.conv1.parameters(), 'lr': lr * 0.1},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    print(f"Starting fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_epoch_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(all_targets, all_preds)

        history['train_loss'].append(float(epoch_loss))
        history['val_loss'].append(float(val_epoch_loss))
        history['val_acc'].append(float(val_acc))

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, 'resnet18_cirrhosis.pth'))
            print(f"Saved best model (Val Acc: {val_acc*100:.2f}%)")

    # Final Evaluation & Metrics Export
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds).tolist()
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)

    metrics = {
        'best_accuracy': float(best_acc),
        'confusion_matrix': cm,
        'classification_report': report,
        'class_names': class_names,
        'history': history
    }

    with open(os.path.join(models_dir, 'vision_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Fine-Tuning Complete! Best Val Accuracy: {best_acc*100:.2f}%")
    return model, metrics

if __name__ == '__main__':
    base_dir = r'c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection'
    ds_dir = os.path.join(base_dir, 'Dataset')
    models_dir = os.path.join(base_dir, 'models')
    train_vision_pipeline(ds_dir, models_dir, epochs=8, batch_size=32)
