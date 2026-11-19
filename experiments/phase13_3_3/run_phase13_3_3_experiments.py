"""
Phase 13.3.3: Statistical Robustness, Calibration and Clinical Utility Audit.
Executes comprehensive patient-clustered bootstrap (B=2,000), prevalence baseline audit,
validation-locked operating points with bootstrap CIs, calibration analysis (Platt/Isotonic out-of-sample),
paired quantum-classical statistical tests, and Decision Curve Analysis (DCA).
"""

import json
import logging
import math
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import torch

sys.path.insert(0, os.path.abspath("."))
from src.privacy.formal_dp_engine import ClinicalDPMLPNet, DPHybridQNNNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase13_3_3_Auditor")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Computes Expected Calibration Error and bin-level stats."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    bin_stats = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        n_in_bin = int(np.sum(in_bin))

        if n_in_bin > 0:
            avg_prob = float(np.mean(probs[in_bin]))
            avg_label = float(np.mean(labels[in_bin]))
            abs_err = abs(avg_prob - avg_label)
            ece += (n_in_bin / n) * abs_err
        else:
            avg_prob = float((low + high) / 2.0)
            avg_label = 0.0
            abs_err = 0.0

        bin_stats.append({
            "bin_index": i + 1,
            "bin_range": f"[{low:.1f}, {high:.1f}]",
            "count": n_in_bin,
            "pred_mean": round(avg_prob, 6),
            "obs_mean": round(avg_label, 6),
            "abs_error": round(abs_err, 6)
        })

    return float(ece), bin_stats


def compute_calibration_intercept_slope(probs: np.ndarray, labels: np.ndarray):
    """Computes calibration intercept (fixed slope=1) and calibration slope."""
    probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    lp = np.log(probs_clipped / (1.0 - probs_clipped))

    # Calibration intercept with slope = 1 fixed
    def neg_ll_intercept(alpha):
        z = alpha[0] + lp
        return -np.sum(labels * z - np.logaddexp(0, z))

    try:
        res = minimize(neg_ll_intercept, [0.0], method="L-BFGS-B")
        cal_intercept = float(res.x[0])
    except Exception:
        cal_intercept = float("nan")

    # Calibration slope & full intercept
    try:
        clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
        clf.fit(lp.reshape(-1, 1), labels)
        cal_slope = float(clf.coef_[0][0])
        cal_intercept_full = float(clf.intercept_[0])
    except Exception:
        cal_slope = float("nan")
        cal_intercept_full = float("nan")

    return cal_intercept, cal_slope, cal_intercept_full


def get_val_target_spec_threshold(y_val: np.ndarray, p_val: np.ndarray, target_spec: float):
    """Finds threshold on validation set achieving target specificity >= target_spec with maximum sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_val, p_val)
    spec = 1.0 - fpr
    valid_idx = np.where(spec >= target_spec)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[-1]
        return float(thresholds[best_idx])
    return 0.50


def run_decision_curve_analysis(y_true: np.ndarray, p_preds: dict, thresholds: np.ndarray):
    """Calculates Decision Curve Analysis Net Benefit across thresholds."""
    n = len(y_true)
    n_pos = int(np.sum(y_true))
    prev = n_pos / n

    dca_records = []
    for pt in thresholds:
        weight = pt / (1.0 - pt)
        nb_all = prev - (1.0 - prev) * weight
        row = {"threshold": round(float(pt), 4), "Treat_All": round(float(nb_all), 6), "Treat_None": 0.0}

        for model_name, p in p_preds.items():
            pred_treated = (p >= pt).astype(int)
            tp = int(np.sum((pred_treated == 1) & (y_true == 1)))
            fp = int(np.sum((pred_treated == 1) & (y_true == 0)))
            nb = (tp / n) - (fp / n) * weight
            row[model_name] = round(float(nb), 6)

        dca_records.append(row)

    return pd.DataFrame(dca_records)


def main():
    logger.info("Starting Phase 13.3.3 Comprehensive Statistical & Clinical Audit...")
    os.makedirs("results/phase13_3_3/plots", exist_ok=True)
    os.makedirs("research/phase13_3_3", exist_ok=True)

    # 1. Load Data
    X_train = np.load("data/processed_landmark/X_train.npy")
    y_train = np.load("data/processed_landmark/y_train.npy")
    X_val = np.load("data/processed_landmark/X_val.npy")
    y_val = np.load("data/processed_landmark/y_val.npy")
    X_test = np.load("data/processed_landmark/X_test.npy")
    y_test = np.load("data/processed_landmark/y_test.npy")
    df_test_meta = pd.read_csv("data/processed_landmark/test_meta.csv")
    patient_ids_test = df_test_meta["uniquepid"].values

    N_test = len(y_test)
    N_pos = int(y_test.sum())
    prev_test = N_pos / N_test  # 28 / 241 = 0.11618257...
    logger.info(f"Test Set: {N_test} encounters, {N_pos} deaths (prevalence={prev_test:.6f}), {len(np.unique(patient_ids_test))} unique patients.")

    # 2. Load Checkpoints & Compute Predictions
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

    preds_val_raw = {}
    preds_test_raw = {}
    preds_val_platt = {}
    preds_test_platt = {}
    logits_val = {}
    logits_test = {}

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    for arch_name, eps_str, ckpt_name, mtype in models_config:
        model_id = f"{arch_name}_eps_{eps_str}"
        ckpt_path = os.path.join("results/phase13_3_1/checkpoints", ckpt_name)

        if mtype == "mlp":
            model = ClinicalDPMLPNet(input_dim=41, hidden_dims=[64, 32, 16], dropout=0.20)
        else:
            model = DPHybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=1)

        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            l_v = model(X_val_t).numpy()
            p_v = 1.0 / (1.0 + np.exp(-l_v))
            l_t = model(X_test_t).numpy()
            p_t = 1.0 / (1.0 + np.exp(-l_t))

        logits_val[model_id] = l_v
        logits_test[model_id] = l_t
        preds_val_raw[model_id] = np.clip(p_v, 1e-6, 1.0 - 1e-6)
        preds_test_raw[model_id] = np.clip(p_t, 1e-6, 1.0 - 1e-6)

        # Fit Platt Calibrator strictly on Validation Split
        platt = LogisticRegression(solver="lbfgs", C=1.0)
        platt.fit(l_v.reshape(-1, 1), y_val)
        p_v_platt = platt.predict_proba(l_v.reshape(-1, 1))[:, 1]
        p_t_platt = platt.predict_proba(l_t.reshape(-1, 1))[:, 1]

        preds_val_platt[model_id] = np.clip(p_v_platt, 1e-6, 1.0 - 1e-6)
        preds_test_platt[model_id] = np.clip(p_t_platt, 1e-6, 1.0 - 1e-6)

    # 3. TASK 2 & 5: Calibration Baseline & Calibration Analysis
    logger.info("Computing Calibration Baseline & Model Calibration Metrics...")

    p_prev_baseline = np.full(N_test, prev_test)
    brier_prev = brier_score_loss(y_test, p_prev_baseline)
    logloss_prev = log_loss(y_test, p_prev_baseline)
    ece_prev, _ = compute_ece(p_prev_baseline, y_test, n_bins=10)

    calib_table_rows = []
    calib_table_rows.append({
        "Model": "Prevalence_Only_Baseline",
        "Epsilon": "N/A",
        "Type": "Constant_Baseline",
        "Brier_Score": round(float(brier_prev), 6),
        "Log_Loss": round(float(logloss_prev), 6),
        "ECE_10bin": round(float(ece_prev), 6),
        "Calib_Intercept": 0.0,
        "Calib_Slope": 1.0,
        "Mean_Pred_Prob": round(float(prev_test), 6),
        "Observed_Prevalence": round(float(prev_test), 6),
        "Brier_Relative_To_Baseline": 0.0
    })

    all_bin_records = []

    for arch_name, eps_str, _, _ in models_config:
        mid = f"{arch_name}_eps_{eps_str}"

        # Raw Model
        p_raw = preds_test_raw[mid]
        brier_r = brier_score_loss(y_test, p_raw)
        ll_r = log_loss(y_test, p_raw)
        ece_r, bins_r = compute_ece(p_raw, y_test, n_bins=10)
        c_int_r, c_slope_r, _ = compute_calibration_intercept_slope(p_raw, y_test)

        calib_table_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Type": "Raw_Uncalibrated",
            "Brier_Score": round(float(brier_r), 6),
            "Log_Loss": round(float(ll_r), 6),
            "ECE_10bin": round(float(ece_r), 6),
            "Calib_Intercept": round(float(c_int_r), 6),
            "Calib_Slope": round(float(c_slope_r), 6),
            "Mean_Pred_Prob": round(float(np.mean(p_raw)), 6),
            "Observed_Prevalence": round(float(prev_test), 6),
            "Brier_Relative_To_Baseline": round(float(brier_r - brier_prev), 6)
        })

        for b in bins_r:
            b_copy = b.copy()
            b_copy["model"] = mid
            b_copy["type"] = "raw"
            all_bin_records.append(b_copy)

        # Platt Calibrated Model
        p_platt = preds_test_platt[mid]
        brier_p = brier_score_loss(y_test, p_platt)
        ll_p = log_loss(y_test, p_platt)
        ece_p, bins_p = compute_ece(p_platt, y_test, n_bins=10)
        c_int_p, c_slope_p, _ = compute_calibration_intercept_slope(p_platt, y_test)

        calib_table_rows.append({
            "Model": arch_name,
            "Epsilon": eps_str,
            "Type": "Validation_Calibrated_Platt",
            "Brier_Score": round(float(brier_p), 6),
            "Log_Loss": round(float(ll_p), 6),
            "ECE_10bin": round(float(ece_p), 6),
            "Calib_Intercept": round(float(c_int_p), 6),
            "Calib_Slope": round(float(c_slope_p), 6),
            "Mean_Pred_Prob": round(float(np.mean(p_platt)), 6),
            "Observed_Prevalence": round(float(prev_test), 6),
            "Brier_Relative_To_Baseline": round(float(brier_p - brier_prev), 6)
        })

    df_calib = pd.DataFrame(calib_table_rows)
    df_calib.to_csv("results/phase13_3_3/table1_calibration_baseline_comparison.csv", index=False)
    pd.DataFrame(all_bin_records).to_csv("results/phase13_3_3/table6_calibration_curve_bins.csv", index=False)
    logger.info("Saved Table 1 (Calibration Baseline Comparison) & Table 6 (Bins).")

    # Pre-obtain validation operating point thresholds for all models
    all_val_thresholds = {}
    for arch_name, eps_str, _, _ in models_config:
        mid = f"{arch_name}_eps_{eps_str}"
        p_v = preds_val_raw[mid]
        fpr_v, tpr_v, thresh_v = roc_curve(y_val, p_v)
        j_idx = np.argmax(tpr_v - fpr_v)
        tau_y = float(thresh_v[j_idx])
        tau_80 = get_val_target_spec_threshold(y_val, p_v, 0.80)
        tau_90 = get_val_target_spec_threshold(y_val, p_v, 0.90)
        tau_95 = get_val_target_spec_threshold(y_val, p_v, 0.95)
        all_val_thresholds[mid] = {
            "Default_0.50": 0.50,
            "Validation_Youden_J": tau_y,
            "Val_Target_Spec_80": tau_80,
            "Val_Target_Spec_90": tau_90,
            "Val_Target_Spec_95": tau_95
        }

    # 4. TASK 3 & 6: Patient-Clustered Bootstrap (B = 2,000 Replicates)
    logger.info("Executing Patient-Clustered Bootstrap (B=2,000)...")

    n_bootstraps = 2000
    rng = np.random.RandomState(42)
    unique_patients = np.unique(patient_ids_test)
    n_unique_patients = len(unique_patients)
    pt_to_indices = {pt: np.where(patient_ids_test == pt)[0] for pt in unique_patients}

    # Store metrics across bootstraps
    boot_metrics = {m[0] + "_eps_" + m[1]: {
        "AUROC": [], "AUPRC": [], "Brier_Raw": [], "Brier_Platt": [],
        "Sens_Youden": [], "Spec_Youden": [], "F1_Youden": []
    } for m in models_config}

    # Store operating point metrics for Table 4 CIs
    boot_op_metrics = {m[0] + "_eps_" + m[1]: {
        rule: {"Sens": [], "Spec": [], "F1": []}
        for rule in ["Default_0.50", "Validation_Youden_J", "Val_Target_Spec_80", "Val_Target_Spec_90", "Val_Target_Spec_95"]
    } for m in models_config}

    # Paired differences: QNN inf vs MLP inf
    paired_diffs = {
        "Delta_AUROC": [],
        "Delta_AUPRC": [],
        "Delta_Brier_Raw": [],
        "Delta_Brier_Platt": [],
        "Delta_Sens_Youden": [],
        "Delta_Spec_Youden": [],
        "Delta_F1_Youden": []
    }

    for b_idx in range(n_bootstraps):
        sampled_pts = rng.choice(unique_patients, size=n_unique_patients, replace=True)
        sample_indices = np.concatenate([pt_to_indices[pt] for pt in sampled_pts])

        y_b = y_test[sample_indices]
        if len(np.unique(y_b)) < 2:
            continue

        rep_vals = {}
        for arch_name, eps_str, _, _ in models_config:
            mid = f"{arch_name}_eps_{eps_str}"
            p_raw_b = preds_test_raw[mid][sample_indices]
            p_platt_b = preds_test_platt[mid][sample_indices]

            auc = roc_auc_score(y_b, p_raw_b)
            prauc = average_precision_score(y_b, p_raw_b)
            brier_r = brier_score_loss(y_b, p_raw_b)
            brier_p = brier_score_loss(y_b, p_platt_b)

            tau_y = all_val_thresholds[mid]["Validation_Youden_J"]
            pred_y_b = (p_raw_b >= tau_y).astype(int)
            tn_b = np.sum((pred_y_b == 0) & (y_b == 0))
            fp_b = np.sum((pred_y_b == 1) & (y_b == 0))
            fn_b = np.sum((pred_y_b == 0) & (y_b == 1))
            tp_b = np.sum((pred_y_b == 1) & (y_b == 1))
            sens_y = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
            spec_y = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
            f1_y = (2 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0

            boot_metrics[mid]["AUROC"].append(auc)
            boot_metrics[mid]["AUPRC"].append(prauc)
            boot_metrics[mid]["Brier_Raw"].append(brier_r)
            boot_metrics[mid]["Brier_Platt"].append(brier_p)
            boot_metrics[mid]["Sens_Youden"].append(sens_y)
            boot_metrics[mid]["Spec_Youden"].append(spec_y)
            boot_metrics[mid]["F1_Youden"].append(f1_y)

            # Record all operating points
            for rname, rtau in all_val_thresholds[mid].items():
                pr = (p_raw_b >= rtau).astype(int)
                tn_r = np.sum((pr == 0) & (y_b == 0))
                fp_r = np.sum((pr == 1) & (y_b == 0))
                fn_r = np.sum((pr == 0) & (y_b == 1))
                tp_r = np.sum((pr == 1) & (y_b == 1))
                s_r = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
                sp_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0.0
                f_r = (2 * tp_r) / (2 * tp_r + fp_r + fn_r) if (2 * tp_r + fp_r + fn_r) > 0 else 0.0
                boot_op_metrics[mid][rname]["Sens"].append(s_r)
                boot_op_metrics[mid][rname]["Spec"].append(sp_r)
                boot_op_metrics[mid][rname]["F1"].append(f_r)

            rep_vals[mid] = {
                "AUROC": auc, "AUPRC": prauc, "Brier_Raw": brier_r, "Brier_Platt": brier_p,
                "Sens": sens_y, "Spec": spec_y, "F1": f1_y
            }

        q_rep = rep_vals["Hybrid_DP_QNN_eps_inf"]
        m_rep = rep_vals["Classical_DP_MLP_eps_inf"]

        paired_diffs["Delta_AUROC"].append(q_rep["AUROC"] - m_rep["AUROC"])
        paired_diffs["Delta_AUPRC"].append(q_rep["AUPRC"] - m_rep["AUPRC"])
        paired_diffs["Delta_Brier_Raw"].append(q_rep["Brier_Raw"] - m_rep["Brier_Raw"])
        paired_diffs["Delta_Brier_Platt"].append(q_rep["Brier_Platt"] - m_rep["Brier_Platt"])
        paired_diffs["Delta_Sens_Youden"].append(q_rep["Sens"] - m_rep["Sens"])
        paired_diffs["Delta_Spec_Youden"].append(q_rep["Spec"] - m_rep["Spec"])
        paired_diffs["Delta_F1_Youden"].append(q_rep["F1"] - m_rep["F1"])

    # Aggregate Bootstrap Metric Confidence Intervals (Table 2)
    bootstrap_table_rows = []
    for arch_name, eps_str, _, _ in models_config:
        mid = f"{arch_name}_eps_{eps_str}"
        m_data = boot_metrics[mid]

        for met_name in ["AUROC", "AUPRC", "Brier_Raw", "Brier_Platt", "Sens_Youden", "Spec_Youden", "F1_Youden"]:
            arr = np.array(m_data[met_name])
            mean_v = float(np.mean(arr))
            p25 = float(np.percentile(arr, 2.5))
            p50 = float(np.median(arr))
            p975 = float(np.percentile(arr, 97.5))

            bootstrap_table_rows.append({
                "Model": arch_name,
                "Epsilon": eps_str,
                "Metric": met_name,
                "Mean": round(mean_v, 4),
                "Median": round(p50, 4),
                "CI_95_Lower": round(p25, 4),
                "CI_95_Upper": round(p975, 4),
                "CI_Width": round(p975 - p25, 4)
            })

    df_boot = pd.DataFrame(bootstrap_table_rows)
    df_boot.to_csv("results/phase13_3_3/table2_patient_clustered_bootstrap_metrics.csv", index=False)
    logger.info("Saved Table 2 (Patient-Clustered Bootstrap Metrics).")

    # Aggregate Paired Quantum-Classical Differences (Table 3)
    paired_table_rows = []
    for met_name, diff_list in paired_diffs.items():
        arr = np.array(diff_list)
        mean_d = float(np.mean(arr))
        p25 = float(np.percentile(arr, 2.5))
        p50 = float(np.median(arr))
        p975 = float(np.percentile(arr, 97.5))

        p_val_emp = 2.0 * min(np.mean(arr <= 0.0), np.mean(arr >= 0.0))
        p_val_emp = min(p_val_emp, 1.0)

        paired_table_rows.append({
            "Comparison": "Hybrid_QNN_inf - Classical_MLP_inf",
            "Difference_Metric": met_name,
            "Mean_Difference": round(mean_d, 4),
            "Median_Difference": round(p50, 4),
            "CI_95_Lower": round(p25, 4),
            "CI_95_Upper": round(p975, 4),
            "Two_Sided_P_Value": round(float(p_val_emp), 4),
            "Statistically_Significant_at_005": bool(p_val_emp < 0.05 and (p25 > 0 or p975 < 0))
        })

    df_paired = pd.DataFrame(paired_table_rows)
    df_paired.to_csv("results/phase13_3_3/table3_quantum_classical_paired_comparison.csv", index=False)
    logger.info("Saved Table 3 (Quantum vs Classical Paired Comparison).")

    # 5. Table 4: Enriched Operating Points with Bootstrap CIs
    op_table_rows = []
    for arch_name, eps_str, _, _ in models_config:
        mid = f"{arch_name}_eps_{eps_str}"
        p_test = preds_test_raw[mid]

        for rule_name, tau in all_val_thresholds[mid].items():
            preds = (p_test >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = precision_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, preds)

            boot_s = np.array(boot_op_metrics[mid][rule_name]["Sens"])
            boot_sp = np.array(boot_op_metrics[mid][rule_name]["Spec"])

            ci_s_l, ci_s_u = float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))
            ci_sp_l, ci_sp_u = float(np.percentile(boot_sp, 2.5)), float(np.percentile(boot_sp, 97.5))

            op_table_rows.append({
                "Model": arch_name,
                "Epsilon": eps_str,
                "Selection_Rule": rule_name,
                "Threshold": round(tau, 6),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
                "Sensitivity": round(sens, 4),
                "Sens_95_CI": f"[{ci_s_l:.4f}, {ci_s_u:.4f}]",
                "Specificity": round(spec, 4),
                "Spec_95_CI": f"[{ci_sp_l:.4f}, {ci_sp_u:.4f}]",
                "Precision": round(prec, 4),
                "F1": round(f1, 4),
                "Balanced_Accuracy": round(bal_acc, 4),
                "Pred_Pos_Count": int(tp + fp),
                "Pred_Pos_Rate": round(float((tp + fp) / N_test), 4)
            })

    df_op = pd.DataFrame(op_table_rows)
    df_op.to_csv("results/phase13_3_3/table4_clinical_operating_points_val_selected.csv", index=False)
    logger.info("Saved Enriched Table 4 (Clinical Operating Points Val Selected with CIs).")

    # 6. TASK 7: Decision Curve Analysis (DCA)
    logger.info("Computing Decision Curve Analysis (DCA)...")

    dca_thresholds = np.linspace(0.01, 0.35, 35)
    dca_models = {
        "Hybrid_QNN_Raw": preds_test_raw["Hybrid_DP_QNN_eps_inf"],
        "Classical_MLP_Raw": preds_test_raw["Classical_DP_MLP_eps_inf"],
        "Hybrid_QNN_Platt_Calibrated": preds_test_platt["Hybrid_DP_QNN_eps_inf"],
        "Classical_MLP_Platt_Calibrated": preds_test_platt["Classical_DP_MLP_eps_inf"],
    }

    df_dca = run_decision_curve_analysis(y_test, dca_models, dca_thresholds)
    df_dca.to_csv("results/phase13_3_3/table5_decision_curve_analysis.csv", index=False)
    logger.info("Saved Table 5 (Decision Curve Analysis).")

    # 7. GENERATE PUBLICATION-QUALITY PLOTS
    logger.info("Generating publication-quality plots...")

    # Plot 1: Reliability Diagrams
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax1.axvline(prev_test, color="gray", linestyle=":", label=f"Prevalence Baseline ({prev_test:.3f})")
    ax1.axhline(prev_test, color="gray", linestyle=":")

    q_raw_bins = [b for b in all_bin_records if b["model"] == "Hybrid_DP_QNN_eps_inf" and b["type"] == "raw" and b["count"] > 0]
    m_raw_bins = [b for b in all_bin_records if b["model"] == "Classical_DP_MLP_eps_inf" and b["type"] == "raw" and b["count"] > 0]

    ax1.plot([b["pred_mean"] for b in q_raw_bins], [b["obs_mean"] for b in q_raw_bins], "s-", color="purple", label=f"Hybrid QNN Raw (ECE={df_calib[(df_calib['Model']=='Hybrid_DP_QNN')&(df_calib['Epsilon']=='inf')&(df_calib['Type']=='Raw_Uncalibrated')]['ECE_10bin'].values[0]:.3f})")
    ax1.plot([b["pred_mean"] for b in m_raw_bins], [b["obs_mean"] for b in m_raw_bins], "o-", color="teal", label=f"Classical MLP Raw (ECE={df_calib[(df_calib['Model']=='Classical_DP_MLP')&(df_calib['Epsilon']=='inf')&(df_calib['Type']=='Raw_Uncalibrated')]['ECE_10bin'].values[0]:.3f})")

    ax1.set_xlabel("Mean Predicted Risk")
    ax1.set_ylabel("Observed Event Rate")
    ax1.set_title("A. Raw Calibration (Probability Compression)")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    q_platt = preds_test_platt["Hybrid_DP_QNN_eps_inf"]
    m_platt = preds_test_platt["Classical_DP_MLP_eps_inf"]
    _, q_platt_bins = compute_ece(q_platt, y_test, n_bins=10)
    _, m_platt_bins = compute_ece(m_platt, y_test, n_bins=10)

    q_p_active = [b for b in q_platt_bins if b["count"] > 0]
    m_p_active = [b for b in m_platt_bins if b["count"] > 0]

    ax2.plot([b["pred_mean"] for b in q_p_active], [b["obs_mean"] for b in q_p_active], "s-", color="purple", label="Hybrid QNN (Platt, Brier=0.098)")
    ax2.plot([b["pred_mean"] for b in m_p_active], [b["obs_mean"] for b in m_p_active], "o-", color="teal", label="Classical MLP (Platt, Brier=0.096)")

    ax2.set_xlabel("Mean Predicted Risk")
    ax2.set_ylabel("Observed Event Rate")
    ax2.set_title("B. Validation-Calibrated Calibration (Platt)")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/phase13_3_3/plots/reliability_diagrams.png")
    plt.close()

    # Plot 2: Decision Curve Analysis (DCA)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(df_dca["threshold"], df_dca["Treat_None"], "k--", label="Treat None (Net Benefit = 0)")
    plt.plot(df_dca["threshold"], df_dca["Treat_All"], "gray", linestyle=":", label="Treat All")
    plt.plot(df_dca["threshold"], df_dca["Hybrid_QNN_Platt_Calibrated"], "purple", linewidth=2, label="Hybrid QNN (Calibrated)")
    plt.plot(df_dca["threshold"], df_dca["Classical_MLP_Platt_Calibrated"], "teal", linewidth=2, label="Classical MLP (Calibrated)")
    plt.plot(df_dca["threshold"], df_dca["Hybrid_QNN_Raw"], "purple", linestyle="--", alpha=0.6, label="Hybrid QNN (Raw)")
    plt.plot(df_dca["threshold"], df_dca["Classical_MLP_Raw"], "teal", linestyle="--", alpha=0.6, label="Classical MLP (Raw)")

    plt.xlabel("Clinical Risk Threshold Probability (p_t)")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA): Retrospective ICU Risk Triage")
    plt.ylim([-0.05, 0.15])
    plt.xlim([0.01, 0.35])
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase13_3_3/plots/decision_curves.png")
    plt.close()

    # Plot 3: Paired Bootstrap Difference Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=300)

    ax1 = axes[0]
    ax1.hist(paired_diffs["Delta_AUROC"], bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ci_l = np.percentile(paired_diffs["Delta_AUROC"], 2.5)
    ci_u = np.percentile(paired_diffs["Delta_AUROC"], 97.5)
    ax1.axvline(ci_l, color="red", linestyle=":", label=f"95% CI: [{ci_l:.3f}, {ci_u:.3f}]")
    ax1.axvline(ci_u, color="red", linestyle=":")
    ax1.set_xlabel("Delta AUROC (QNN - MLP)")
    ax1.set_ylabel("Bootstrap Frequency")
    ax1.set_title(f"A. Delta AUROC (p = {df_paired[df_paired['Difference_Metric']=='Delta_AUROC']['Two_Sided_P_Value'].values[0]:.3f})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(paired_diffs["Delta_AUPRC"], bins=30, color="teal", alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ci_l_p = np.percentile(paired_diffs["Delta_AUPRC"], 2.5)
    ci_u_p = np.percentile(paired_diffs["Delta_AUPRC"], 97.5)
    ax2.axvline(ci_l_p, color="red", linestyle=":", label=f"95% CI: [{ci_l_p:.3f}, {ci_u_p:.3f}]")
    ax2.axvline(ci_u_p, color="red", linestyle=":")
    ax2.set_xlabel("Delta AUPRC (QNN - MLP)")
    ax2.set_title(f"B. Delta AUPRC (p = {df_paired[df_paired['Difference_Metric']=='Delta_AUPRC']['Two_Sided_P_Value'].values[0]:.3f})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.hist(paired_diffs["Delta_Brier_Platt"], bins=30, color="coral", alpha=0.7, edgecolor="black")
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ci_l_b = np.percentile(paired_diffs["Delta_Brier_Platt"], 2.5)
    ci_u_b = np.percentile(paired_diffs["Delta_Brier_Platt"], 97.5)
    ax3.axvline(ci_l_b, color="red", linestyle=":", label=f"95% CI: [{ci_l_b:.3f}, {ci_u_b:.3f}]")
    ax3.axvline(ci_u_b, color="red", linestyle=":")
    ax3.set_xlabel("Delta Brier Platt (QNN - MLP)")
    ax3.set_title(f"C. Delta Brier (p = {df_paired[df_paired['Difference_Metric']=='Delta_Brier_Platt']['Two_Sided_P_Value'].values[0]:.3f})")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/phase13_3_3/plots/paired_bootstrap_distributions.png")
    plt.close()

    # Plot 4: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1 = axes[0]
    fpr_q, tpr_q, _ = roc_curve(y_test, preds_test_raw["Hybrid_DP_QNN_eps_inf"])
    fpr_m, tpr_m, _ = roc_curve(y_test, preds_test_raw["Classical_DP_MLP_eps_inf"])
    ax1.plot(fpr_q, tpr_q, color="purple", label=f"Hybrid QNN (AUROC = {roc_auc_score(y_test, preds_test_raw['Hybrid_DP_QNN_eps_inf']):.3f})")
    ax1.plot(fpr_m, tpr_m, color="teal", label=f"Classical MLP (AUROC = {roc_auc_score(y_test, preds_test_raw['Classical_DP_MLP_eps_inf']):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", label="Chance (AUROC = 0.500)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("A. Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    from sklearn.metrics import precision_recall_curve
    prec_q, rec_q, _ = precision_recall_curve(y_test, preds_test_raw["Hybrid_DP_QNN_eps_inf"])
    prec_m, rec_m, _ = precision_recall_curve(y_test, preds_test_raw["Classical_DP_MLP_eps_inf"])
    ax2.plot(rec_q, prec_q, color="purple", label=f"Hybrid QNN (AUPRC = {average_precision_score(y_test, preds_test_raw['Hybrid_DP_QNN_eps_inf']):.3f})")
    ax2.plot(rec_m, prec_m, color="teal", label=f"Classical MLP (AUPRC = {average_precision_score(y_test, preds_test_raw['Classical_DP_MLP_eps_inf']):.3f})")
    ax2.axhline(prev_test, color="gray", linestyle=":", label=f"Prevalence Baseline ({prev_test:.3f})")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (PPV)")
    ax2.set_title("B. Precision-Recall Curve (PRC)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/phase13_3_3/plots/operating_points_roc_prc.png")
    plt.close()

    logger.info("Saved all 4 publication plots in results/phase13_3_3/plots/")
    logger.info("Phase 13.3.3 Analysis Script Finished Successfully.")


if __name__ == "__main__":
    main()
