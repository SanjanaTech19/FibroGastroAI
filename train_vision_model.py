import os
import sys
import json
import time
import datetime
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from resnet18_model import build_resnet18
from preprocess import prepare_image

BASE_DIR = r"c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection"
MODELS_DIR = os.path.join(BASE_DIR, "models")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class CleanUltrasoundDataset(Dataset):
    def __init__(self, items, is_train=True):
        self.items = items
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        fpath = item['path']
        label = item['label']
        try:
            tensor, _, _ = prepare_image(fpath, is_train=self.is_train)
        except Exception:
            tensor, _, _ = prepare_image(Image.new('RGB', (224, 224), (0, 0, 0)), is_train=False)
        return tensor, label

def train_staged_vision_model(epochs=12, batch_size=32, seed=42):
    set_seed(seed)
    print("="*80)
    print("TASK 5 & 6 & 14: STAGED TRANSFER LEARNING & TRAINING PIPELINE")
    print("="*80)

    manifest_path = os.path.join(MODELS_DIR, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        from dataset_builder import build_clean_dataset_splits
        manifest = build_clean_dataset_splits(seed=seed)
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)

    train_items = manifest['train']
    val_items = manifest['val']
    test_items = manifest['test']
    class_names = manifest['class_names']

    train_dataset = CleanUltrasoundDataset(train_items, is_train=True)
    val_dataset = CleanUltrasoundDataset(val_items, is_train=False)
    test_dataset = CleanUltrasoundDataset(test_items, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Training Device: {device}")

    # Build ResNet18 model
    model = build_resnet18(num_classes=len(class_names))

    # Load ImageNet pre-trained weights
    url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    try:
        print("Loading ImageNet pre-trained backbone weights...")
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("ImageNet transfer learning weights loaded successfully!")
    except Exception as e:
        print(f"Could not load ImageNet weights ({e}), training from scratch.")

    model = model.to(device)

    # Class-weighted loss
    train_labels = [it['label'] for it in train_items]
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=np.array(train_labels))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Phase 1: Freeze backbone, train classifier head (3 epochs)
    print("\n--- PHASE 1: Training Classifier Head (Backbone Frozen) ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer_head = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(3):
        model.train()
        r_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer_head.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer_head.step()
            r_loss += loss.item() * imgs.size(0)
        
        print(f"Phase 1 Epoch [{ep+1}/3] - Head Train Loss: {r_loss/len(train_dataset):.4f}")

    # Phase 2: Unfreeze backbone, fine-tune end-to-end (9 epochs)
    print("\n--- PHASE 2: End-to-End Fine-Tuning (All Layers Unfrozen) ---")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 3e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 5e-5},
        {'params': model.layer2.parameters(), 'lr': 2e-5},
        {'params': model.layer1.parameters(), 'lr': 1e-5},
        {'params': model.conv1.parameters(), 'lr': 1e-5},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    for ep in range(epochs):
        model.train()
        r_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = r_loss / len(train_dataset)

        # Validation
        model.eval()
        v_loss = 0.0
        preds_all = []
        targets_all = []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(out, dim=1)
                preds_all.extend(preds.cpu().numpy())
                targets_all.extend(lbls.cpu().numpy())

        val_loss = v_loss / len(val_dataset)
        val_acc = accuracy_score(targets_all, preds_all)
        val_f1 = f1_score(targets_all, preds_all, average='macro')

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_f1'].append(float(val_f1))

        print(f"Phase 2 Epoch [{ep+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%")

        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            model_save_path = os.path.join(MODELS_DIR, 'resnet18_cirrhosis.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved Best Checkpoint (Val Macro F1: {val_f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {ep+1}")
                break

    # Task 14: Model Versioning & Metadata Export
    metadata = {
        'model_name': 'ResNet18 METAVIR Fibrosis Classifier',
        'architecture': 'ResNet18 (ImageNet Pre-trained with Staged Fine-Tuning)',
        'class_names': class_names,
        'random_seed': seed,
        'input_size': [224, 224, 3],
        'training_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_stats': {
            'total_unique_images': manifest['total_unique_images'],
            'train_samples': len(train_items),
            'val_samples': len(val_items),
            'test_samples': len(test_items)
        },
        'validation_metrics': {
            'best_val_accuracy': float(best_val_acc),
            'best_val_macro_f1': float(best_val_f1)
        },
        'training_history': history,
        'confidence_calibration': {
            'temperature': 1.0,
            'confidence_threshold': 0.45,
            'indeterminate_label': 'Indeterminate / Low confidence — additional clinical assessment required'
        }
    }

    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\nTraining Complete! Saved Best Model & Metadata to {meta_path}")
    return model, metadata

if __name__ == '__main__':
    train_staged_vision_model()
