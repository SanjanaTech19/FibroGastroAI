import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def strip_ui_text_and_borders(img_pil):
    """
    Strips scanner text headers (e.g. LOGIQ E9 text, patient info, depth scales)
    and outer black UI padding from ultrasound screenshots or document pages.
    """
    arr = np.array(img_pil)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
    h_orig, w_orig = gray.shape[:2]
    mean_val = np.mean(gray)

    # Case A: Medical Report Document Page (Light/White background)
    if mean_val > 120:
        _, dark_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        max_area = 0
        min_area = (h_orig * w_orig) * 0.03
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = w / float(h)
            if area > min_area and area > max_area and 0.3 < aspect < 3.0:
                roi = gray[y:y+h, x:x+w]
                if np.std(roi) > 12:
                    max_area = area
                    best_box = (x, y, w, h)
        
        if best_box:
            x, y, w, h = best_box
            return Image.fromarray(arr[y:y+h, x:x+w]), best_box
        return img_pil, (0, 0, w_orig, h_orig)

    # Case B: Scanner Screenshot with top text header or outer black padding
    _, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        header_h = int(h * 0.10)
        if y + header_h < h_orig:
            y_new = y + header_h
            h_new = h - header_h
        else:
            y_new, h_new = y, h
            
        pad_x = int(w * 0.02)
        pad_y = int(h_new * 0.02)
        
        x_final = max(0, x + pad_x)
        y_final = max(0, y_new + pad_y)
        w_final = min(w_orig - x_final, w - 2 * pad_x)
        h_final = min(h_orig - y_final, h_new - 2 * pad_y)
        
        if w_final > 50 and h_final > 50:
            return Image.fromarray(arr[y_final:y_final+h_final, x_final:x_final+w_final]), (x_final, y_final, w_final, h_final)

    return img_pil, (0, 0, w_orig, h_orig)

def prepare_image(img_input, is_train=False):
    """
    Unified, authoritative preprocessing pipeline using PIL & OpenCV.
    Steps:
      1. Load & convert to RGB
      2. Strip UI text & scanner borders (ROI extraction)
      3. Resize to 224x224
      4. Apply conservative ultrasound augmentations (Train only)
      5. Convert to Tensor & ImageNet normalize
    """
    if isinstance(img_input, str):
        img_pil = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, bytes):
        import io
        img_pil = Image.open(io.BytesIO(img_input)).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img_pil = img_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    # 1. ROI extraction & UI header removal
    roi_pil, bbox = strip_ui_text_and_borders(img_pil)

    # 2. Resize to 224x224
    img_resized = roi_pil.resize((224, 224), Image.BILINEAR)

    # 3. Augmentation (Train only - NO vertical flip!)
    if is_train:
        # Random horizontal flip (50%)
        if np.random.rand() > 0.5:
            img_resized = img_resized.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Small rotation (-8 to +8 degrees)
        if np.random.rand() > 0.5:
            angle = float(np.random.uniform(-8.0, 8.0))
            img_resized = img_resized.rotate(angle)

        # Mild brightness variation (0.9 to 1.1)
        if np.random.rand() > 0.5:
            b_factor = float(np.random.uniform(0.9, 1.1))
            img_resized = ImageEnhance.Brightness(img_resized).enhance(b_factor)
            
        # Mild contrast variation (0.9 to 1.1)
        if np.random.rand() > 0.5:
            c_factor = float(np.random.uniform(0.9, 1.1))
            img_resized = ImageEnhance.Contrast(img_resized).enhance(c_factor)

    # 4. Convert to Tensor & Normalize
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor, roi_pil, bbox
