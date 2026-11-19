"""Clinical machine learning evaluation metrics and calibration scoring.
Computes discrimination (AUROC, AUPRC), calibration (Brier, ECE), and diagnostic metrics.
"""

import os
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across uniform confidence bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for samples in this probability bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Computes all required discrimination, calibration, and classification metrics.
    
    Returns:
        Dict containing Accuracy, Precision, Recall (Sensitivity), Specificity,
        F1-score, ROC-AUC, PR-AUC, Brier Score, and Expected Calibration Error.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Probabilistic discrimination
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(y_true.mean())

    # Calibration
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "ece": float(ece),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_dir: str = "results/figures"
) -> Tuple[str, str]:
    """Generates and saves publication-quality ROC and PR curves."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(f"ROC Curve - {model_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    roc_filename = f"roc_{model_name.lower().replace(' ', '_')}.png"
    roc_path = os.path.join(output_dir, roc_filename)
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline_pr = float(y_true.mean())

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="#2ca02c", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})")
    plt.axhline(y=baseline_pr, color="#7f7f7f", linestyle="--", lw=1.5, label=f"Prevalence ({baseline_pr:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title(f"Precision-Recall Curve - {model_name}", fontsize=12, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    pr_filename = f"pr_{model_name.lower().replace(' ', '_')}.png"
    pr_path = os.path.join(output_dir, pr_filename)
    plt.savefig(pr_path, dpi=300)
    plt.close()

    return roc_path, pr_path
