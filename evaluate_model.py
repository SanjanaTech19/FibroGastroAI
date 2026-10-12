import os
import sys
import json
import torch
import numpy as np
from PIL import Image

from resnet18_model import build_resnet18
from preprocess import prepare_image
from calibration import calibrate_probabilities, evaluate_prediction_confidence, compute_ordinal_metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR = r"c:\Users\Rajasekar\OneDrive\Desktop\Liver Cirrhosis Detection"
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(MODELS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

def evaluate_on_test_set():
    print("="*80)
    print("TASK 11: COMPREHENSIVE EVALUATION ON UNTOUCHED TEST SET")
    print("="*80)

    manifest_path = os.path.join(MODELS_DIR, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Dataset manifest missing! Please run dataset_builder.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    test_items = manifest['test']
    class_names = manifest['class_names']

    model_path = os.path.join(MODELS_DIR, "resnet18_cirrhosis.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model weights missing! Please run train_vision_model.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []
    sample_logs = []

    for item in test_items:
        fpath = item['path']
        true_label = item['label']
        try:
            tensor, _, _ = prepare_image(fpath, is_train=False)
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor).squeeze(0).cpu().numpy()
                probs = calibrate_probabilities(logits, temperature=1.0)
                pred_label = int(np.argmax(probs))
                
            y_true.append(true_label)
            y_pred.append(pred_label)
            y_probs.append(probs.tolist())

            conf_eval = evaluate_prediction_confidence(probs, class_names)
            is_correct = (pred_label == true_label)
            
            sample_logs.append({
                'filename': item['filename'],
                'true_stage': class_names[true_label],
                'pred_stage': class_names[pred_label],
                'confidence': conf_eval['confidence_score'],
                'status': conf_eval['confidence_status'],
                'is_correct': is_correct
            })
        except Exception as e:
            print(f"Error evaluating test item {fpath}: {e}")

    # Calculate comprehensive metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    ordinal_metrics = compute_ordinal_metrics(y_true, y_pred)

    # Per-class sensitivity & specificity
    cm_arr = np.array(cm)
    per_class_stats = {}
    for i, cname in enumerate(class_names):
        tp = cm_arr[i, i]
        fn = np.sum(cm_arr[i, :]) - tp
        fp = np.sum(cm_arr[:, i]) - tp
        tn = np.sum(cm_arr) - (tp + fn + fp)
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        per_class_stats[cname] = {
            'sensitivity': round(sens * 100, 2),
            'specificity': round(spec * 100, 2),
            'precision': round(report[cname]['precision'] * 100, 2),
            'recall': round(report[cname]['recall'] * 100, 2),
            'f1_score': round(report[cname]['f1-score'] * 100, 2)
        }

    evaluation_report = {
        'overall_metrics': {
            'test_accuracy': round(float(acc * 100), 2),
            'balanced_accuracy': round(float(bal_acc * 100), 2),
            'macro_f1': round(float(macro_f1 * 100), 2),
            'total_test_samples': len(test_items)
        },
        'ordinal_metrics': ordinal_metrics,
        'per_class_metrics': per_class_stats,
        'confusion_matrix': cm,
        'classification_report': report,
        'sample_predictions': sample_logs
    }

    eval_out_path = os.path.join(EVAL_DIR, "test_evaluation_metrics.json")
    with open(eval_out_path, 'w') as f:
        json.dump(evaluation_report, f, indent=4)

    print("\n--- TEST SET PERFORMANCE METRICS ---")
    print(f"Test Accuracy:         {acc*100:.2f}%")
    print(f"Balanced Accuracy:     {bal_acc*100:.2f}%")
    print(f"Macro F1-Score:        {macro_f1*100:.2f}%")
    print(f"+/- 1 Stage Accuracy:  {ordinal_metrics['plus_minus_1_stage_accuracy']}%")
    print(f"F4 Sensitivity:        {ordinal_metrics['f4_cirrhosis_sensitivity']}%")
    print(f"F4 Specificity:        {ordinal_metrics['f4_cirrhosis_specificity']}%")

    print("\nConfusion Matrix:")
    print(np.array(cm))
    print(f"\nSaved test evaluation report to {eval_out_path}")
    print("="*80)

    return evaluation_report

if __name__ == '__main__':
    evaluate_on_test_set()
