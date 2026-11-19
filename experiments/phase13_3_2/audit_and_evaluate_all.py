"""
Phase 13.3.2 Audit & Metric Extraction Script.
Evaluates all 10 saved Phase 13.3.1 checkpoints on the locked patient-disjoint landmark cohort.
Computes default-threshold metrics, validation-locked threshold metrics, clinical operating points,
ECE, Brier, and probability distribution statistics.
"""

import json
import logging
import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import torch

sys.path.insert(0, os.path.abspath("."))
from src.privacy.formal_dp_engine import ClinicalDPMLPNet, DPHybridQNNNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase13_3_2_Auditor")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:
            in_bin = in_bin | (probs == bin_boundaries[i + 1])
        n_in_bin = np.sum(in_bin)
        if n_in_bin > 0:
            avg_prob = np.mean(probs[in_bin])
            avg_label = np.mean(labels[in_bin])
            ece += (n_in_bin / n) * np.abs(avg_prob - avg_label)
    return float(ece)


def find_validation_thresholds(y_val: np.ndarray, p_val: np.ndarray):
    """Finds optimal thresholds strictly on validation set."""
    fpr, tpr, thresh = roc_curve(y_val, p_val)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    best_youden_tau = float(thresh[best_j_idx])

    # F1 search over candidate thresholds
    cand_thresh = np.unique(np.percentile(p_val, np.linspace(1, 99, 100)))
    best_f1 = -1.0
    best_f1_tau = 0.5
    for t in cand_thresh:
        preds = (p_val >= t).astype(int)
        f = f1_score(y_val, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_f1_tau = float(t)

    return best_youden_tau, best_f1_tau


def compute_operating_points(y_test: np.ndarray, p_test: np.ndarray):
    """Computes Sens@Spec and Spec@Sens clinical operating points."""
    fpr, tpr, _ = roc_curve(y_test, p_test)
    spec = 1.0 - fpr

    sens_spec80 = float(np.max([t for f, t in zip(fpr, tpr) if f <= 0.20]))
    sens_spec90 = float(np.max([t for f, t in zip(fpr, tpr) if f <= 0.10]))
    sens_spec95 = float(np.max([t for f, t in zip(fpr, tpr) if f <= 0.05]))

    spec_sens80 = float(np.max([1.0 - f for f, t in zip(fpr, tpr) if t >= 0.80]))
    spec_sens90 = float(np.max([1.0 - f for f, t in zip(fpr, tpr) if t >= 0.90]))

    return {
        "Sens@Spec80": sens_spec80,
        "Sens@Spec90": sens_spec90,
        "Sens@Spec95": sens_spec95,
        "Spec@Sens80": spec_sens80,
        "Spec@Sens90": spec_sens90
    }


def main():
    os.makedirs("results/phase13_3_2", exist_ok=True)
    os.makedirs("research/phase13_3_2", exist_ok=True)

    X_train = np.load("data/processed_landmark/X_train.npy")
    y_train = np.load("data/processed_landmark/y_train.npy")
    X_val = np.load("data/processed_landmark/X_val.npy")
    y_val = np.load("data/processed_landmark/y_val.npy")
    X_test = np.load("data/processed_landmark/X_test.npy")
    y_test = np.load("data/processed_landmark/y_test.npy")

    logger.info(f"Loaded datasets: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")
    logger.info(f"Deaths: Train={int(y_train.sum())}, Val={int(y_val.sum())}, Test={int(y_test.sum())}")

    models_config = [
        ("Classical_DP_MLP", "1.0", "mlp_eps_1.0.pt", "mlp"),
        ("Classical_DP_MLP", "2.0", "mlp_eps_2.0.pt", "mlp"),
        ("Classical_DP_MLP", "4.0", "mlp_eps_4.0.pt", "mlp"),
        ("Classical_DP_MLP", "8.0", "mlp_eps_8.0.pt", "mlp"),
        ("Classical_DP_MLP", "inf", "mlp_eps_inf.pt", "mlp"),
        ("Hybrid_DP_QNN", "1.0", "qnn_eps_1.0.pt", "qnn"),
        ("Hybrid_DP_QNN", "2.0", "qnn_eps_2.0.pt", "qnn"),
        ("Hybrid_DP_QNN", "4.0", "qnn_eps_4.0.pt", "qnn"),
        ("Hybrid_DP_QNN", "8.0", "qnn_eps_8.0.pt", "qnn"),
        ("Hybrid_DP_QNN", "inf", "qnn_eps_inf.pt", "qnn"),
    ]

    table1_rows = []
    table2_rows = []
    table3_rows = []
    table4_rows = []

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    for arch_name, eps_str, ckpt_name, mtype in models_config:
        ckpt_path = os.path.join("results/phase13_3_1/checkpoints", ckpt_name)
        if not os.path.exists(ckpt_path):
            logger.error(f"Missing checkpoint: {ckpt_path}")
            continue

        if mtype == "mlp":
            model = ClinicalDPMLPNet(input_dim=41, hidden_dims=[64, 32, 16], dropout=0.20)
        else:
            model = DPHybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=1)

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            l_val = model(X_val_t).numpy()
            p_val = 1.0 / (1.0 + np.exp(-l_val))
            p_val = np.clip(p_val, 1e-6, 1.0 - 1e-6)

            l_test = model(X_test_t).numpy()
            p_test = 1.0 / (1.0 + np.exp(-l_test))
            p_test = np.clip(p_test, 1e-6, 1.0 - 1e-6)

        # 1. Default threshold (0.50)
        preds_def = (p_test >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_def, labels=[0, 1]).ravel()
        sens_def = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_def = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec_def = precision_score(y_test, preds_def, zero_division=0)
        rec_def = recall_score(y_test, preds_def, zero_division=0)
        f1_def = f1_score(y_test, preds_def, zero_division=0)
        acc_def = accuracy_score(y_test, preds_def)
        bal_acc_def = balanced_accuracy_score(y_test, preds_def)
        auroc = roc_auc_score(y_test, p_test)
        prauc = average_precision_score(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)
        ece = compute_ece(p_test, y_test, n_bins=10)

        table1_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Threshold": 0.50,
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Sensitivity": round(sens_def, 4),
            "Specificity": round(spec_def, 4),
            "Precision": round(prec_def, 4),
            "Recall": round(rec_def, 4),
            "F1": round(f1_def, 4),
            "Accuracy": round(acc_def, 4),
            "Balanced_Accuracy": round(bal_acc_def, 4),
            "AUROC": round(auroc, 4),
            "AUPRC": round(prauc, 4),
            "PRAUC": round(prauc, 4),
            "Brier": round(brier, 4),
            "ECE": round(ece, 4),
            "Pred_Pos_Count": int(tp + fp),
            "Pred_Pos_Rate": round((tp + fp) / len(y_test), 4)
        })

        # 2. Validation-locked optimal thresholds
        tau_youden, tau_f1 = find_validation_thresholds(y_val, p_val)

        # Evaluate Youden on test set
        preds_y = (p_test >= tau_youden).astype(int)
        tn_y, fp_y, fn_y, tp_y = confusion_matrix(y_test, preds_y, labels=[0, 1]).ravel()
        sens_y = tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else 0.0
        spec_y = tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0.0
        prec_y = precision_score(y_test, preds_y, zero_division=0)
        rec_y = recall_score(y_test, preds_y, zero_division=0)
        f1_y = f1_score(y_test, preds_y, zero_division=0)
        acc_y = accuracy_score(y_test, preds_y)
        bal_acc_y = balanced_accuracy_score(y_test, preds_y)

        table2_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Val_Locked_Youden_Tau": round(tau_youden, 6),
            "TP": int(tp_y),
            "FP": int(fp_y),
            "TN": int(tn_y),
            "FN": int(fn_y),
            "Sensitivity": round(sens_y, 4),
            "Specificity": round(spec_y, 4),
            "Precision": round(prec_y, 4),
            "Recall": round(rec_y, 4),
            "F1": round(f1_y, 4),
            "Accuracy": round(acc_y, 4),
            "Balanced_Accuracy": round(bal_acc_y, 4),
            "Pred_Pos_Count": int(tp_y + fp_y),
            "Pred_Pos_Rate": round((tp_y + fp_y) / len(y_test), 4)
        })

        # 3. Clinical operating points & calibration
        op_points = compute_operating_points(y_test, p_test)
        table3_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Sens@Spec80": round(op_points["Sens@Spec80"], 4),
            "Sens@Spec90": round(op_points["Sens@Spec90"], 4),
            "Sens@Spec95": round(op_points["Sens@Spec95"], 4),
            "Spec@Sens80": round(op_points["Spec@Sens80"], 4),
            "Spec@Sens90": round(op_points["Spec@Sens90"], 4),
            "Brier": round(brier, 4),
            "ECE": round(ece, 4)
        })

        # 4. Probability distribution statistics
        table4_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Min_Prob": round(float(np.min(p_test)), 6),
            "Q25_Prob": round(float(np.percentile(p_test, 25)), 6),
            "Median_Prob": round(float(np.median(p_test)), 6),
            "Mean_Prob": round(float(np.mean(p_test)), 6),
            "Q75_Prob": round(float(np.percentile(p_test, 75)), 6),
            "Max_Prob": round(float(np.max(p_test)), 6),
            "Std_Prob": round(float(np.std(p_test)), 6)
        })

    df1 = pd.DataFrame(table1_rows)
    df2 = pd.DataFrame(table2_rows)
    df3 = pd.DataFrame(table3_rows)
    df4 = pd.DataFrame(table4_rows)

    df1.to_csv("results/phase13_3_2/table1_default_threshold_metrics.csv", index=False)
    df2.to_csv("results/phase13_3_2/table2_validation_locked_threshold_metrics.csv", index=False)
    df3.to_csv("results/phase13_3_2/table3_clinical_operating_points.csv", index=False)
    df4.to_csv("results/phase13_3_2/table4_probability_distributions.csv", index=False)

    logger.info("Saved Table 1, Table 2, Table 3, and Table 4 to results/phase13_3_2/")
    print("\n--- TABLE 1: DEFAULT THRESHOLD 0.50 METRICS ---")
    print(df1[["Model", "Epsilon", "TP", "FP", "TN", "FN", "Sensitivity", "Specificity", "F1", "Balanced_Accuracy", "AUROC", "AUPRC"]])

    print("\n--- TABLE 2: VALIDATION-LOCKED YOUDEN THRESHOLD METRICS ---")
    print(df2[["Model", "Epsilon", "Val_Locked_Youden_Tau", "TP", "FP", "TN", "FN", "Sensitivity", "Specificity", "Precision", "F1", "Balanced_Accuracy"]])

    print("\n--- TABLE 3: CLINICAL OPERATING POINTS ---")
    print(df3)

    print("\n--- TABLE 4: PROBABILITY DISTRIBUTION STATS ---")
    print(df4)


if __name__ == "__main__":
    main()
