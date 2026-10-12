import numpy as np

def calibrate_probabilities(logits, temperature=1.0):
    """
    Applies temperature scaling to raw logits before softmax calculation.
    """
    logits_calibrated = logits / temperature
    exp_logits = np.exp(logits_calibrated - np.max(logits_calibrated))
    probs = exp_logits / np.sum(exp_logits)
    return probs

def evaluate_prediction_confidence(probs, stages=['F0', 'F1', 'F2', 'F3', 'F4'], confidence_threshold=0.45):
    """
    Task 9 & 19: Probability Calibration & Uncertainty Detection.
    If maximum softmax probability is below the confidence threshold (0.45),
    returns an Indeterminate / Low Confidence status.
    """
    pred_idx = int(np.argmax(probs))
    max_prob = float(probs[pred_idx])
    confidence_pct = round(max_prob * 100, 1)

    stage_probs = [{'stage': s, 'prob': round(float(probs[i] * 100), 1)} for i, s in enumerate(stages)]

    if max_prob >= 0.70:
        status = "High Confidence"
        display_label = f"Predicted METAVIR Stage: {stages[pred_idx]}"
        is_reliable = True
    elif max_prob >= confidence_threshold:
        status = "Moderate Confidence"
        display_label = f"Predicted METAVIR Stage: {stages[pred_idx]}"
        is_reliable = True
    else:
        status = "Indeterminate / Low Confidence"
        display_label = "Indeterminate prediction — image quality/model confidence is insufficient."
        is_reliable = False

    return {
        'pred_stage': stages[pred_idx],
        'confidence_score': confidence_pct,
        'confidence_status': status,
        'display_label': display_label,
        'is_reliable': is_reliable,
        'stage_probs': stage_probs,
        'disclaimer': "This is a research prediction and not a clinical diagnosis."
    }

def compute_ordinal_metrics(y_true, y_pred):
    """
    Task 10: METAVIR Ordinal Evaluation Metrics.
    Computes exact accuracy, ±1 stage accuracy, F3/F4 severe detection, and F4 cirrhosis detection.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)

    if n == 0:
        return {}

    exact_acc = float(np.mean(y_true == y_pred) * 100)
    plus_minus_1_acc = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100)

    # Severe stage detection (F3 or F4 -> label >= 3)
    severe_true = (y_true >= 3)
    severe_pred = (y_pred >= 3)
    severe_tp = np.sum((severe_true == 1) & (severe_pred == 1))
    severe_fn = np.sum((severe_true == 1) & (severe_pred == 0))
    severe_fp = np.sum((severe_true == 0) & (severe_pred == 1))
    severe_tn = np.sum((severe_true == 0) & (severe_pred == 0))

    severe_sens = float(severe_tp / (severe_tp + severe_fn) * 100) if (severe_tp + severe_fn) > 0 else 0.0
    severe_spec = float(severe_tn / (severe_tn + severe_fp) * 100) if (severe_tn + severe_fp) > 0 else 0.0

    # F4 Cirrhosis detection (F4 -> label == 4)
    f4_true = (y_true == 4)
    f4_pred = (y_pred == 4)
    f4_tp = np.sum((f4_true == 1) & (f4_pred == 1))
    f4_fn = np.sum((f4_true == 1) & (f4_pred == 0))
    f4_fp = np.sum((f4_true == 0) & (f4_pred == 1))
    f4_tn = np.sum((f4_true == 0) & (f4_pred == 0))

    f4_sens = float(f4_tp / (f4_tp + f4_fn) * 100) if (f4_tp + f4_fn) > 0 else 0.0
    f4_spec = float(f4_tn / (f4_tn + f4_fp) * 100) if (f4_tn + f4_fp) > 0 else 0.0

    return {
        'exact_stage_accuracy': round(exact_acc, 2),
        'plus_minus_1_stage_accuracy': round(plus_minus_1_acc, 2),
        'severe_stage_f3_f4_sensitivity': round(severe_sens, 2),
        'severe_stage_f3_f4_specificity': round(severe_spec, 2),
        'f4_cirrhosis_sensitivity': round(f4_sens, 2),
        'f4_cirrhosis_specificity': round(f4_spec, 2)
    }
