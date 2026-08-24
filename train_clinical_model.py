import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import xgboost as xgb

def train_clinical_pipeline(csv_path, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    print(f"Loading clinical dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean & Preprocess
    df_clean = df.copy()

    # Convert Age from days to years
    if 'Age' in df_clean.columns:
        df_clean['Age_Years'] = (df_clean['Age'] / 365.25).round(1)

    # Encode target 'Status': C=0 (Survived), D=1 (Deceased), CL=2 (Transplant)
    status_map = {'C': 0, 'D': 1, 'CL': 2}
    df_clean['Target'] = df_clean['Status'].map(status_map)

    # Categorical features mapping
    binary_cols = {'Sex': {'M': 0, 'F': 1},
                   'Ascites': {'N': 0, 'Y': 1},
                   'Hepatomegaly': {'N': 0, 'Y': 1},
                   'Spiders': {'N': 0, 'Y': 1},
                   'Drug': {'D-penicillamine': 1, 'Placebo': 0}}
    
    for col, mapping in binary_cols.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)

    # Edema: N=0, S=0.5, Y=1.0
    if 'Edema' in df_clean.columns:
        df_clean['Edema'] = df_clean['Edema'].map({'N': 0.0, 'S': 0.5, 'Y': 1.0})

    # Select numerical & engineered features
    feature_cols = [
        'Age_Years', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema',
        'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
        'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage'
    ]

    # Impute missing values with column median/mode
    medians = {}
    for col in feature_cols:
        if col in df_clean.columns:
            median_val = df_clean[col].median() if df_clean[col].dtype != 'object' else df_clean[col].mode()[0]
            if pd.isna(median_val):
                median_val = 0
            df_clean[col] = df_clean[col].fillna(median_val)
            medians[col] = float(median_val)

    X = df_clean[feature_cols]
    y = df_clean['Target']

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"XGBoost Clinical Model Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Survived (C)', 'Deceased (D)', 'Transplant (CL)']))

    # Feature Importance
    importances = model.feature_importances_
    feat_imp = {col: float(imp) for col, imp in zip(feature_cols, importances)}
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

    # Save artifacts
    preprocessor = {
        'feature_cols': feature_cols,
        'medians': medians,
        'status_map': status_map,
        'inv_status_map': {0: 'Survived (Censored)', 1: 'Deceased', 2: 'Liver Transplant'}
    }

    with open(os.path.join(models_dir, 'xgboost_clinical.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(models_dir, 'clinical_preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

    metrics = {
        'accuracy': float(acc),
        'confusion_matrix': cm,
        'feature_importance': feat_imp,
        'target_names': ['Survived', 'Deceased', 'Transplant']
    }

    with open(os.path.join(models_dir, 'clinical_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    print("Saved XGBoost clinical model and metrics successfully!")
    return model, preprocessor, metrics

if __name__ == '__main__':
    base_dir = r'c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection'
    csv_file = os.path.join(base_dir, 'cirrhosis.csv')
    models_folder = os.path.join(base_dir, 'models')
    train_clinical_pipeline(csv_file, models_folder)
