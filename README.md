# 🩺 FibroGastro AI

### Multimodal liver fibrosis staging and clinical prognosis

FibroGastro AI is a Streamlit research application that combines ultrasound image analysis with clinical biomarker modeling. It is designed to make model predictions easier to inspect through stage labels, outcome probabilities, metrics, and Grad-CAM visualizations.

> ⚠️ **Research use only:** This project is not a medical device and must not be used for diagnosis, treatment, or other clinical decisions.

## ✨ Features

| Area | What it does |
| --- | --- |
| 🖼️ Ultrasound staging | Uses a ResNet18 classifier to predict METAVIR stages F0-F4. |
| 🔥 Explainability | Uses Grad-CAM to show image regions that influence a vision prediction. |
| 📊 Clinical prognosis | Uses XGBoost to estimate outcomes from cirrhosis biomarkers. |
| 📈 Analytics | Displays model metrics, confusion matrices, and feature importance when available. |

## 📁 Project layout

```text
.
├── app.py                    # Streamlit application
├── auto_crop.py              # Ultrasound image crop helper
├── cirrhosis.csv             # Clinical training data
├── gradcam.py                # Grad-CAM visualization
├── resnet18_model.py         # ResNet18 definition
├── train_clinical_model.py   # XGBoost training pipeline
├── train_vision_model.py     # ResNet18 training pipeline
├── eval_vision_metrics.py    # Vision evaluation and metrics generation
├── test_pipeline.py          # Core pipeline smoke test
├── Dataset/F0-F4/            # Ultrasound images grouped by stage
└── models/                   # Trained models and metrics
```

## 🧰 Requirements

- Python 3.9 or newer
- CPU or CUDA-capable GPU
- Ultrasound images in `Dataset/F0`, `Dataset/F1`, `Dataset/F2`, `Dataset/F3`, and `Dataset/F4`
- Clinical data in `cirrhosis.csv`

## 🚀 Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 🧠 Train the models

Train the clinical XGBoost model:

```powershell
python train_clinical_model.py
```

Train the ResNet18 ultrasound classifier:

```powershell
python train_vision_model.py
```

Evaluate the vision model and regenerate its metrics:

```powershell
python eval_vision_metrics.py
```

Training writes model artifacts under `models/`. Dataset files, CSV files, and binary model artifacts are ignored by Git because they may be large or contain sensitive data. Store them locally or manage them through a separate data or model registry.

## ▶️ Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The clinical prognosis tab requires:

- `models/xgboost_clinical.pkl`
- `models/clinical_preprocessor.pkl`

The vision tab loads `models/resnet18_cirrhosis.pth`. Without trained weights, it falls back to an untrained model.

## 🧪 Test the pipeline

```powershell
python test_pipeline.py
```

The smoke test expects the trained clinical artifacts to exist. It checks the clinical model, vision model, and Grad-CAM overlay shape.

## 🏷️ Output labels

The vision classifier uses the METAVIR labels `F0`, `F1`, `F2`, `F3`, and `F4`:

- `F0`: No fibrosis
- `F1`: Portal fibrosis
- `F2`: Periportal fibrosis
- `F3`: Septal fibrosis
- `F4`: Cirrhosis

The clinical classifier maps the source `Status` field as follows:

- `C`: Survived
- `D`: Deceased
- `CL`: Liver transplant

Predictions are estimates from the supplied training data and should not be interpreted as clinical advice.
