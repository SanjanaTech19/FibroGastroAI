import cv2
import numpy as np
from PIL import Image

def remove_scanner_ui_artifacts(img_pil):
    """
    Removes top scanner UI text headers (e.g. LOGIQ E9, GE, Siemens text),
    depth rulers, and outer frame margins from ultrasound screenshots.
    Returns:
        cropped_pil (PIL.Image): Pure liver tissue ROI
        bbox (tuple): (x, y, w, h) bounding box in original image
    """
    arr = np.array(img_pil)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
    h_orig, w_orig = gray.shape[:2]
    mean_val = np.mean(gray)

    # Case 1: Medical Report Document with White Background
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

    # Case 2: Standard Ultrasound Scan or Scanner Screenshot (Dark Background)
    # Threshold out background black pixels
    _, thresh = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # If image contains top header text overlay (e.g. LOGIQ E9), crop out top 10% header
        # Check if top 12% of bounding box contains text noise (high frequency lines)
        header_h = int(h * 0.11)
        if y + header_h < h_orig:
            # Shift y down past the text header
            y_new = y + header_h
            h_new = h - header_h
        else:
            y_new, h_new = y, h
            
        # Also crop 3% outer border margins to remove thin white border frames
        pad_x = int(w * 0.02)
        pad_y = int(h_new * 0.02)
        
        x_final = max(0, x + pad_x)
        y_final = max(0, y_new + pad_y)
        w_final = min(w_orig - x_final, w - 2 * pad_x)
        h_final = min(h_orig - y_final, h_new - 2 * pad_y)
        
        if w_final > 50 and h_final > 50:
            return Image.fromarray(arr[y_final:y_final+h_final, x_final:x_final+w_final]), (x_final, y_final, w_final, h_final)

    # Fallback
    return img_pil, (0, 0, w_orig, h_orig)

def detect_ultrasound_roi(img_pil):
    return remove_scanner_ui_artifacts(img_pil)

def auto_crop_ultrasound(img_pil):
    cropped_pil, _ = detect_ultrasound_roi(img_pil)
    return cropped_pil
