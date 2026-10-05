import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Data
df = pd.read_csv('cirrhosis.csv')
df['Age_Years'] = (df['Age'] / 365.25).round(1)
features = ['Age_Years', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']

X = df[features]
y = LabelEncoder().fit_transform(df['Status'])

X_imp = SimpleImputer(strategy='median').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imp)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 2. Benchmark 5 Classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF Kernel)': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='mlogloss', random_state=42)
}

print("=== Classification Algorithm Benchmark Results ===")
metrics_list = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    prec = precision_score(y_test, preds, average='weighted', zero_division=0) * 100
    rec = recall_score(y_test, preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0) * 100
    
    print(f"{name:20s} -> Accuracy: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}%")
    metrics_list.append({'Algorithm': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

df_metrics = pd.DataFrame(metrics_list)
print("\nBest Algorithm:", df_metrics.sort_values(by='Accuracy', ascending=False).iloc[0]['Algorithm'])
