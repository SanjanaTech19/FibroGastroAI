import os
import io
import json
import base64
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from resnet18_model import build_resnet18
from train_vision_model import preprocess_image
from gradcam import GradCAM, overlay_heatmap
from auto_crop import auto_crop_ultrasound

BASE_DIR = r"c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection"
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(BASE_DIR, "cirrhosis.csv")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="FibroGastro AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# METAVIR Staging Metadata
METAVIR_INFO = {
    'F0': {'title': 'F0 · No Fibrosis', 'desc': 'Healthy liver architecture with standard parenchyma and no structural scarring.', 'color': '#E89A8A', 'badge': 'Low Risk'},
    'F1': {'title': 'F1 · Portal Fibrosis', 'desc': 'Mild fibrous expansion localized within portal tracts, without septal formation.', 'color': '#D9A07B', 'badge': 'Mild Risk'},
    'F2': {'title': 'F2 · Periportal Fibrosis', 'desc': 'Moderate fibrosis with portal expansion and rare periportal septa.', 'color': '#C4554B', 'badge': 'Moderate Risk'},
    'F3': {'title': 'F3 · Septal Fibrosis', 'desc': 'Severe bridging fibrosis with numerous fibrous septa connecting portal areas.', 'color': '#A64B3B', 'badge': 'High Risk'},
    'F4': {'title': 'F4 · Cirrhosis', 'desc': 'Advanced cirrhosis characterized by regenerative nodular architectural distortion.', 'color': '#732B25', 'badge': 'Critical Risk'}
}

# Model Loaders
def load_vision_model():
    model_path = os.path.join(MODELS_DIR, "resnet18_cirrhosis.pth")
    model = build_resnet18(num_classes=5)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_clinical_model():
    xgb_path = os.path.join(MODELS_DIR, "xgboost_clinical.pkl")
    prep_path = os.path.join(MODELS_DIR, "clinical_preprocessor.pkl")
    model, preprocessor = None, None
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            model = pickle.load(f)
    if os.path.exists(prep_path):
        with open(prep_path, 'rb') as f:
            preprocessor = pickle.load(f)
    return model, preprocessor

vision_model = load_vision_model()
clinical_model, clinical_prep = load_clinical_model()

def pil_to_base64(img_pil):
    buffered = io.BytesIO()
    img_pil.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@app.get("/api/sample-patients")
def get_sample_patients():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df_clean = df.dropna(subset=['Bilirubin', 'Albumin', 'Platelets', 'Prothrombin']).head(25)
        records = []
        for _, row in df_clean.iterrows():
            records.append({
                'id': int(row['ID']),
                'age_years': float(round(row['Age']/365.25, 1)),
                'sex': 'Male' if row['Sex'] == 'M' else 'Female',
                'stage': int(row['Stage']) if not pd.isna(row['Stage']) else 3,
                'bilirubin': float(row['Bilirubin']),
                'albumin': float(row['Albumin']),
                'copper': float(row['Copper']) if not pd.isna(row['Copper']) else 70.0,
                'platelets': float(row['Platelets']),
                'sgot': float(row['SGOT']) if not pd.isna(row['SGOT']) else 110.0,
                'prothrombin': float(row['Prothrombin']),
                'ascites': 'Yes' if row['Ascites'] == 'Y' else 'No',
                'hepatomegaly': 'Yes' if row['Hepatomegaly'] == 'Y' else 'No',
                'spiders': 'Yes' if row['Spiders'] == 'Y' else 'No',
                'edema': 'Severe' if row['Edema'] == 'Y' else ('Slight' if row['Edema'] == 'S' else 'No')
            })
        return records
    return []

@app.post("/api/predict-vision")
async def predict_vision(
    opacity: float = Form(0.45),
    file: UploadFile = File(None)
):
    img_pil = None
    img_name = "Uploaded Scan"

    if file is not None and file.filename:
        contents = await file.read()
        if len(contents) > 0:
            img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
            img_name = file.filename

    if img_pil is None:
        raise HTTPException(status_code=400, detail="No ultrasound file chosen. Please upload an image.")

    # Use original PIL image directly without spatial aspect distortion
    img_processed = img_pil
    input_tensor = preprocess_image(img_processed, is_train=False).unsqueeze(0)

    with torch.no_grad():
        outputs = vision_model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()
        pred_idx = int(np.argmax(probs))
        stages = ['F0', 'F1', 'F2', 'F3', 'F4']
        pred_stage = stages[pred_idx]
        confidence = float(probs[pred_idx] * 100)

    grad_cam = GradCAM(vision_model, vision_model.layer4)
    cam_mask, _ = grad_cam(input_tensor, target_category=pred_idx)
    img_np = np.array(img_processed.resize((224, 224)))
    overlay, _ = overlay_heatmap(img_np, cam_mask, alpha=opacity)

    overlay_pil = Image.fromarray(overlay)
    
    stage_probs = [{'stage': s, 'prob': float(probs[i] * 100)} for i, s in enumerate(stages)]

    return {
        'img_name': img_name,
        'pred_stage': pred_stage,
        'confidence': round(confidence, 1),
        'meta': METAVIR_INFO[pred_stage],
        'stage_probs': stage_probs,
        'original_b64': pil_to_base64(img_processed),
        'overlay_b64': pil_to_base64(overlay_pil)
    }

@app.post("/api/predict-clinical")
def predict_clinical(payload: dict):
    if clinical_model is None:
        raise HTTPException(status_code=500, detail="Clinical model not loaded.")

    edema_map = {"No": 0.0, "Slight": 0.5, "Severe": 1.0}
    input_data = {
        'Age_Years': float(payload.get('age_years', 52.0)),
        'Sex': 1 if payload.get('sex') == "Female" else 0,
        'Ascites': 1 if payload.get('ascites') == "Yes" else 0,
        'Hepatomegaly': 1 if payload.get('hepatomegaly') == "Yes" else 0,
        'Spiders': 1 if payload.get('spiders') == "Yes" else 0,
        'Edema': edema_map.get(payload.get('edema'), 0.0),
        'Bilirubin': float(payload.get('bilirubin', 1.5)),
        'Cholesterol': 300.0,
        'Albumin': float(payload.get('albumin', 3.5)),
        'Copper': float(payload.get('copper', 70.0)),
        'Alk_Phos': 1200.0,
        'SGOT': float(payload.get('sgot', 110.0)),
        'Tryglicerides': 120.0,
        'Platelets': float(payload.get('platelets', 250.0)),
        'Prothrombin': float(payload.get('prothrombin', 10.5)),
        'Stage': float(payload.get('stage', 3))
    }

    df_input = pd.DataFrame([input_data])
    preds_proba = clinical_model.predict_proba(df_input)[0]
    pred_class = int(np.argmax(preds_proba))
    
    risk_score = float((preds_proba[1] + preds_proba[2]) * 100)
    surv_prob = float(preds_proba[0] * 100)

    risk_label = "Low Adverse Risk" if risk_score < 25 else ("Moderate Adverse Risk" if risk_score < 60 else "High Adverse Outcome Risk")
    risk_category = "Low" if risk_score < 25 else ("Moderate" if risk_score < 60 else "High")

    importances = clinical_prep.get('feature_cols', [])
    feat_imps = clinical_model.feature_importances_
    
    drivers = []
    for feat, imp in zip(importances, feat_imps):
        drivers.append({'feature': feat, 'importance': float(imp)})
    drivers = sorted(drivers, key=lambda x: x['importance'], reverse=True)[:8]

    return {
        'risk_score': round(risk_score, 1),
        'survival_prob': round(surv_prob, 1),
        'risk_label': risk_label,
        'risk_category': risk_category,
        'predicted_status': clinical_prep['inv_status_map'][pred_class],
        'top_drivers': drivers
    }

@app.get("/api/metrics")
def get_metrics():
    v_metrics_path = os.path.join(MODELS_DIR, "vision_metrics.json")
    c_metrics_path = os.path.join(MODELS_DIR, "clinical_metrics.json")

    v_data, c_data = {}, {}
    if os.path.exists(v_metrics_path):
        with open(v_metrics_path) as f:
            v_data = json.load(f)
    if os.path.exists(c_metrics_path):
        with open(c_metrics_path) as f:
            c_data = json.load(f)

    return {'vision': v_data, 'clinical': c_data}

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
