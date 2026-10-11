import numpy as np

def compute_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return int(tp), int(fp), int(fn), int(tn)

def compute_metrics(y_true, y_pred, y_probs):
    tp, fp, fn, tn = compute_confusion_matrix(y_true, y_pred)
    total = len(y_true)
    
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Probabilistic Metrics
    # 1. Brier Score (Mean Squared Error in probability space)
    brier_score = np.mean((y_probs - y_true) ** 2)
    
    # 2. Log-Loss / Binary Cross Entropy
    eps = 1e-12
    p_clipped = np.clip(y_probs, eps, 1.0 - eps)
    log_loss = -np.mean(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))
    
    # 3. Fast ROC-AUC calculation
    roc_auc = compute_roc_auc(y_true, y_probs)
    
    return {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "Brier Score": round(float(brier_score), 4),
        "Log-Loss": round(float(log_loss), 4),
        "Confusion Matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    }

def compute_roc_auc(y_true, y_probs):
    """Computes Area Under ROC Curve via trapezoidal integration."""
    thresholds = np.linspace(0.0, 1.0, 101)
    tpr_list = []
    fpr_list = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
        
    for th in sorted(thresholds, reverse=True):
        y_pred = (y_probs >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr = tp / n_pos
        fpr = fp / n_neg
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    # Trapezoid area
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    area = np.trapezoid(tpr_arr, fpr_arr)
    return max(0.0, min(1.0, float(area)))
