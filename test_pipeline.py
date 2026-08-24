import os
import torch
import numpy as np
import pickle
from PIL import Image

from resnet18_model import build_resnet18
from train_vision_model import preprocess_image
from gradcam import GradCAM, overlay_heatmap

def test_system():
    base_dir = r"c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection"
    models_dir = os.path.join(base_dir, "models")
    
    print("=== Testing Clinical XGBoost Model ===")
    xgb_path = os.path.join(models_dir, "xgboost_clinical.pkl")
    prep_path = os.path.join(models_dir, "clinical_preprocessor.pkl")
    
    assert os.path.exists(xgb_path), "XGBoost model file missing!"
    assert os.path.exists(prep_path), "Clinical preprocessor file missing!"

    with open(xgb_path, 'rb') as f:
        model_xgb = pickle.load(f)
    with open(prep_path, 'rb') as f:
        prep = pickle.load(f)

    # Test dummy clinical input
    import pandas as pd
    test_patient = pd.DataFrame([{
        'Age_Years': 55.0, 'Sex': 0, 'Ascites': 1, 'Hepatomegaly': 1, 'Spiders': 1,
        'Edema': 0.5, 'Bilirubin': 3.2, 'Cholesterol': 300.0, 'Albumin': 3.1,
        'Copper': 85.0, 'Alk_Phos': 1100.0, 'SGOT': 120.0, 'Tryglicerides': 110.0,
        'Platelets': 180.0, 'Prothrombin': 11.5, 'Stage': 3.0
    }])

    proba = model_xgb.predict_proba(test_patient)[0]
    risk_score = (proba[1] + proba[2]) * 100
    print(f"XGBoost Test Prediction Probabilities: {proba}")
    print(f"Calculated Patient Survival Risk Score: {risk_score:.2f}%")
    print("Clinical Pipeline: PASS [OK]\n")

    print("=== Testing Vision ResNet18 & Grad-CAM Model ===")
    model_vision = build_resnet18(num_classes=5)
    model_pth = os.path.join(models_dir, "resnet18_cirrhosis.pth")
    if os.path.exists(model_pth):
        model_vision.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
        print("Loaded trained ResNet18 weights.")
    else:
        print("Using initialized ResNet18 weights (training in progress).")
    
    model_vision.eval()

    # Create dummy ultrasound image
    dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    tensor = preprocess_image(dummy_img, is_train=False).unsqueeze(0)

    out = model_vision(tensor)
    pred_stage = torch.argmax(out, dim=1).item()
    print(f"ResNet18 Output Logits: {out.detach().numpy()}")
    print(f"Predicted Stage Index: {pred_stage} (Stage F{pred_stage})")

    # Grad-CAM Test
    grad_cam = GradCAM(model_vision, model_vision.layer4)
    cam_mask, _ = grad_cam(tensor, target_category=pred_stage)
    overlay, _ = overlay_heatmap(np.array(dummy_img), cam_mask)
    
    assert overlay.shape == (224, 224, 3), "Grad-CAM overlay shape mismatch!"
    print("Vision Pipeline & Grad-CAM: PASS [OK]\n")
    print("All core multi-modal components verified successfully!")

if __name__ == '__main__':
    test_system()
