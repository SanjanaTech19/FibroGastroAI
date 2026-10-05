import cv2
import numpy as np
from PIL import Image

def auto_crop_ultrasound(img_pil):
    arr = np.array(img_pil)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    h_orig, w_orig = gray.shape[:2]

    # Find active non-black ultrasound sector
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # If the active region fills almost the entire image (>80% of total canvas), return original untouched
        area_ratio = (w * h) / (w_orig * h_orig)
        if area_ratio > 0.80:
            return img_pil

        # Only crop if there are large black UI margins
        if w > 40 and h > 40:
            return Image.fromarray(arr[y:y+h, x:x+w])

    return img_pil
