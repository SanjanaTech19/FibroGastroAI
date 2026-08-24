import os
import torch
import numpy as np
import cv2
from PIL import Image

from resnet18_model import build_resnet18

def auto_crop_ultrasound(img_pil):
    arr = np.array(img_pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    
    # Threshold to find active non-black ultrasound sector
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # Crop ignoring outer 5% text padding
        y1 = y + int(h * 0.05)
        y2 = y + int(h * 0.90)
        x1 = x + int(w * 0.05)
        x2 = x + int(w * 0.95)
        if (x2 - x1) > 40 and (y2 - y1) > 40:
            return Image.fromarray(arr[y1:y2, x1:x2])
    return img_pil

print("Auto-crop module ready!")
