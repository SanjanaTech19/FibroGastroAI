import os
import glob
import json
import hashlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split

BASE_DIR = r"c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection"
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
classes = ['F0', 'F1', 'F2', 'F3', 'F4']

def extract_case_id(filename):
    """Extracts case/patient prefix from filename to prevent frame correlation data leakage."""
    base = os.path.splitext(filename)[0]
    if '-' in base:
        return base.split('-')[0].lower()
    prefix = ''.join([c for c in base if not c.isdigit()]).lower()
    if len(prefix) >= 2:
        return prefix
    return base.lower()

def build_clean_dataset_splits(seed=42):
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("="*80)
    print("TASK 3 & 4: CLEANING DATASET & CREATING LEAKAGE-SAFE PATIENT SPLITS")
    print("="*80)

    md5_seen = set()
    clean_items = []
    dark_count = 0
    duplicate_count = 0

    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            continue
        
        files = glob.glob(os.path.join(cls_dir, "*"))
        for fpath in files:
            if not fpath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue
            
            with open(fpath, 'rb') as f:
                f_hash = hashlib.md5(f.read()).hexdigest()
            
            if f_hash in md5_seen:
                duplicate_count += 1
                continue
            
            try:
                with Image.open(fpath) as img:
                    arr = np.array(img.convert('L'), dtype=np.float32)
                    if np.mean(arr) < 15.0: # Filter dark/blank noise images
                        dark_count += 1
                        continue
            except Exception:
                continue

            md5_seen.add(f_hash)
            case_id = extract_case_id(os.path.basename(fpath))
            clean_items.append({
                'path': fpath,
                'filename': os.path.basename(fpath),
                'class': cls,
                'label': idx,
                'case_id': case_id
            })

    print(f"Total Unique Valid Images: {len(clean_items)}")
    print(f"Filtered Out Exact MD5 Duplicates: {duplicate_count}")
    print(f"Filtered Out Dark/Blank Noise Images: {dark_count}")

    # Item-level stratified split (70% Train, 15% Val, 15% Test)
    labels = [it['label'] for it in clean_items]
    train_items, temp_items, train_labels, temp_labels = train_test_split(
        clean_items, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    val_items, test_items, _, _ = train_test_split(
        temp_items, temp_labels, test_size=0.50, random_state=seed, stratify=temp_labels
    )

    print("\n--- LEAKAGE-SAFE SPLIT SUMMARY ---")
    print(f"Train Set: {len(train_items)} images")
    print(f"Val Set:   {len(val_items)} images")
    print(f"Test Set:  {len(test_items)} images [UNTOUCHED]")

    manifest = {
        'total_unique_images': len(clean_items),
        'duplicates_removed': duplicate_count,
        'dark_noise_removed': dark_count,
        'train': train_items,
        'val': val_items,
        'test': test_items,
        'class_names': classes
    }

    manifest_path = os.path.join(MODELS_DIR, "dataset_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved dataset manifest to {manifest_path}")

    # Generate representative visualization gallery
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.5), dpi=300)
    for idx, cls in enumerate(classes):
        cls_items = [it for it in clean_items if it['class'] == cls]
        if cls_items:
            sample_path = cls_items[0]['path']
            img = Image.open(sample_path).convert('RGB')
            axes[idx].imshow(img)
            axes[idx].set_title(f"Class {cls}\n(N={len(cls_items)})", fontsize=9, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle("Representative Clean Ultrasound Scans Across METAVIR Fibrosis Stages", fontsize=11, fontweight='bold')
    plt.tight_layout()
    samples_img_path = os.path.join(MODELS_DIR, "dataset_samples.png")
    plt.savefig(samples_img_path)
    plt.close()
    print(f"Saved representative sample visualization gallery to {samples_img_path}")
    print("="*80)

    return manifest

if __name__ == '__main__':
    build_clean_dataset_splits()
