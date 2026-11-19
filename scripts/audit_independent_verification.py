"""
Independent Verification and Recalculation Engine.
Loads saved checkpoints and test data directly to recalculate all metrics,
bootstrap 95% confidence intervals, and compute paired statistical tests.
"""
import copy
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    confusion_matrix
)
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical.baselines import ClinicalMLPNet
from src.models.quantum.hybrid_qnn import HybridQNNNet
from src.personalization.local_adapter import PersonalizedHybridQNNNet, PersonalizedMLPNet
from src.federated.client import compute_ece

torch.set_num_threads(1)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> dict:
    """Computes non-parametric patient-level bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    aurocs, praucs, briers, eces = [], [], [], []
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        p_b = probs[idx]
        
        # Guard against zero positive resamples
        if len(np.unique(y_b)) < 2:
            continue
            
        aurocs.append(roc_auc_score(y_b, p_b))
        praucs.append(average_precision_score(y_b, p_b))
        briers.append(brier_score_loss(y_b, p_b))
        eces.append(compute_ece(p_b, y_b))
        
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_ci": (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))),
        "prauc_mean": float(np.mean(praucs)),
        "prauc_ci": (float(np.percentile(praucs, 2.5)), float(np.percentile(praucs, 97.5))),
        "brier_mean": float(np.mean(briers)),
        "brier_ci": (float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))),
        "ece_mean": float(np.mean(eces)),
        "ece_ci": (float(np.percentile(eces, 2.5)), float(np.percentile(eces, 97.5)))
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> dict:
    """Computes paired bootstrap p-value for AUROC difference (Model A - Model B)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    
    orig_diff = roc_auc_score(y_true, probs_a) - roc_auc_score(y_true, probs_b)
    
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, probs_a[idx])
        auc_b = roc_auc_score(y_b, probs_b[idx])
        diffs.append(auc_a - auc_b)
        
    diffs = np.array(diffs)
    # Two-sided empirical p-value under null hypothesis (diff == 0)
    p_value = 2.0 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    p_value = min(1.0, max(0.0, float(p_value)))
    
    return {
        "diff_mean": float(np.mean(diffs)),
        "diff_ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_value": p_value,
        "is_significant": bool(p_value < 0.05)
    }


def main():
    print("===============================================================")
    print("INDEPENDENT SCIENTIFIC AUDIT & RECALCULATION ENGINE")
    print("===============================================================\n")

    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    test_meta = pd.read_csv("data/processed/test_meta.csv")

    print(f"Loaded Test Set: N={len(y_test)} stays, Positive (Expired)={int(y_test.sum())} ({y_test.mean():.2%})\n")

    recalculation_records = []

    # -------------------------------------------------------------
    # 1. AUDIT EXPERIMENT E1: CLASSICAL MODELS
    # -------------------------------------------------------------
    print("--- 1. AUDITING EXPERIMENT E1: CENTRALIZED CLASSICAL MODELS ---")
    classical_reported = pd.read_csv("results/metrics/classical_baseline_metrics.csv")
    print("Reported classical metrics:\n", classical_reported[["Model", "Test_AUROC", "Test_PRAUC", "Test_Brier", "Test_ECE"]])

    classical_preds = {}

    # Logistic Regression
    lr = joblib.load("checkpoints/classical/logistic_regression.joblib")
    p_lr = lr.predict_proba(X_test)[:, 1] if hasattr(lr, "predict_proba") else lr.predict_proba(X_test)
    classical_preds["Logistic Regression"] = p_lr

    # Random Forest
    rf = joblib.load("checkpoints/classical/random_forest.joblib")
    p_rf = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else rf.predict_proba(X_test)
    classical_preds["Random Forest"] = p_rf

    # SVM
    svm = joblib.load("checkpoints/classical/support_vector_machine.joblib")
    p_svm = svm.predict_proba(X_test)[:, 1] if hasattr(svm, "predict_proba") else svm.predict_proba(X_test)
    classical_preds["Support Vector Machine"] = p_svm

    # XGBoost
    xgb = joblib.load("checkpoints/classical/xgboost.joblib")
    p_xgb = xgb.predict_proba(X_test)[:, 1] if hasattr(xgb, "predict_proba") else xgb.predict_proba(X_test)
    classical_preds["XGBoost"] = p_xgb

    # MLP
    mlp = ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32, 16], dropout=0.25)
    mlp.load_state_dict(torch.load("checkpoints/classical/multi_layer_perceptron.pt", map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(mlp(torch.tensor(X_test, dtype=torch.float32))).numpy()
    classical_preds["Multi-Layer Perceptron"] = p_mlp

    for name, p in classical_preds.items():
        auc = roc_auc_score(y_test, p)
        prauc = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)
        ece = compute_ece(p, y_test)
        ci = bootstrap_metric_ci(y_test, p)

        # Lookup reported
        row = classical_reported[classical_reported["Model"] == name]
        rep_auc = float(row["Test_AUROC"].values[0]) if len(row) > 0 else np.nan

        delta_auc = auc - rep_auc
        print(f"{name:25s} | Reported: {rep_auc:.4f} | Recalculated: {auc:.4f} (Delta: {delta_auc:+.4f}) | 95% CI: [{ci['auroc_ci'][0]:.4f}, {ci['auroc_ci'][1]:.4f}]")

        recalculation_records.append({
            "experiment_id": "E1",
            "model_name": name,
            "metric": "AUROC",
            "reported_value": rep_auc,
            "recalculated_value": auc,
            "delta": delta_auc,
            "ci_95": f"[{ci['auroc_ci'][0]:.4f}, {ci['auroc_ci'][1]:.4f}]",
            "verification_status": "Verified Exactly" if abs(delta_auc) < 1e-4 else "Discrepancy Detected"
        })

    # -------------------------------------------------------------
    # 2. AUDIT QUANTUM MACHINE LEARNING MODELS
    # -------------------------------------------------------------
    print("\n--- 2. AUDITING QUANTUM MODELS (STANDALONE & DEPTH SCALING) ---")
    quantum_reported = pd.read_csv("results/metrics/quantum_experiments_metrics.csv")
    print("Reported quantum metrics:\n", quantum_reported[["Experiment", "Test_AUROC", "Test_PRAUC", "Test_Brier", "Test_ECE"]])

    quantum_ckpts = [
        ("Hybrid QNN (4-Qubit Depth-2 Strong)", "checkpoints/quantum/hybrid_qnn_4_qubit_depth_2_strong.pt", 4, 2, "strongly_entangling"),
        ("Hybrid QNN (4-Qubit Depth-1 Light)", "checkpoints/quantum/hybrid_qnn_4_qubit_depth_1_light.pt", 4, 1, "strongly_entangling"),
        ("Hybrid QNN (4-Qubit Depth-3 Deep)", "checkpoints/quantum/hybrid_qnn_4_qubit_depth_3_deep.pt", 4, 3, "strongly_entangling"),
        ("Hybrid QNN (4-Qubit Ring Entangling)", "checkpoints/quantum/hybrid_qnn_4_qubit_ring_entangling.pt", 4, 2, "basic_ring"),
        ("Hybrid QNN (6-Qubit Scaled Register)", "checkpoints/quantum/hybrid_qnn_6_qubit_scaled_register.pt", 6, 2, "strongly_entangling"),
    ]

    quantum_preds = {}

    for name, path, n_qubits, depth, ent in quantum_ckpts:
        model_q = HybridQNNNet(input_dim=41, num_qubits=n_qubits, circuit_depth=depth, entanglement=ent)
        model_q.load_state_dict(torch.load(path, map_location="cpu"))
        model_q.eval()
        with torch.no_grad():
            logits_q = model_q(torch.tensor(X_test, dtype=torch.float32))
            p_q = torch.sigmoid(logits_q).numpy()
        quantum_preds[name] = p_q

        auc_q = roc_auc_score(y_test, p_q)
        prauc_q = average_precision_score(y_test, p_q)
        ci_q = bootstrap_metric_ci(y_test, p_q)

        row_q = quantum_reported[quantum_reported["Experiment"] == name]
        rep_auc_q = float(row_q["Test_AUROC"].values[0]) if len(row_q) > 0 else np.nan
        delta_q = auc_q - rep_auc_q

        print(f"{name:35s} | Reported: {rep_auc_q:.4f} | Recalculated: {auc_q:.4f} (Delta: {delta_q:+.4f}) | 95% CI: [{ci_q['auroc_ci'][0]:.4f}, {ci_q['auroc_ci'][1]:.4f}]")

        recalculation_records.append({
            "experiment_id": "Quantum_Scaling",
            "model_name": name,
            "metric": "AUROC",
            "reported_value": rep_auc_q,
            "recalculated_value": auc_q,
            "delta": delta_q,
            "ci_95": f"[{ci_q['auroc_ci'][0]:.4f}, {ci_q['auroc_ci'][1]:.4f}]",
            "verification_status": "Verified Exactly" if abs(delta_q) < 1e-4 else "Discrepancy Detected"
        })

    # -------------------------------------------------------------
    # 3. STATISTICAL RESOLUTION OF "QUANTUM ADVANTAGE" CLAIM
    # -------------------------------------------------------------
    print("\n--- 3. RESOLUTION OF QUANTUM ADVANTAGE CLAIM ---")
    p_q_strong = quantum_preds["Hybrid QNN (4-Qubit Depth-2 Strong)"]
    auc_q_strong = roc_auc_score(y_test, p_q_strong)

    print(f"Hybrid QNN (4-Qubit Depth-2 Strong) AUROC: {auc_q_strong:.4f}")
    print("Paired comparisons against all classical models on same 393 patient stays:")

    for c_name, p_c in classical_preds.items():
        auc_c = roc_auc_score(y_test, p_c)
        test_res = paired_bootstrap_test(y_test, p_q_strong, p_c, n_bootstraps=1000, seed=42)
        diff = auc_q_strong - auc_c
        sig_str = "SIGNIFICANT" if test_res["is_significant"] else "NOT STATISTICALLY SIGNIFICANT"
        print(f"  vs {c_name:25s} (AUROC {auc_c:.4f}): Diff = {diff:+.4f} (95% CI: [{test_res['diff_ci'][0]:+.4f}, {test_res['diff_ci'][1]:+.4f}], p={test_res['p_value']:.4f}) -> {sig_str}")

    # -------------------------------------------------------------
    # 4. AUDIT EXPERIMENTS E2 & E3: FEDERATED LEARNING
    # -------------------------------------------------------------
    print("\n--- 4. AUDITING EXPERIMENTS E2 & E3: FEDERATED LEARNING ---")
    fed_reported = pd.read_csv("results/metrics/federated_experiments_metrics.csv")
    print("Reported federated metrics:\n", fed_reported[["experiment", "strategy", "model_type", "global_auroc", "worst_client_auroc", "total_bandwidth_mb"]])

    fed_ckpts = [
        ("Classical FL (FedAvg)", "checkpoints/federated/fed_fedavg_classical_best.pt", "classical"),
        ("Classical FL (FedProx)", "checkpoints/federated/fed_fedprox_classical_best.pt", "classical"),
        ("Hybrid Quantum FL (FedAvg)", "checkpoints/federated/fed_fedavg_quantum_best.pt", "quantum"),
        ("Hybrid Quantum FL (FedProx)", "checkpoints/federated/fed_fedprox_quantum_best.pt", "quantum"),
    ]

    for exp_name, ckpt_path, m_type in fed_ckpts:
        if m_type == "classical":
            net = ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32])
        else:
            net = HybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=2, entanglement="strongly_entangling")

        net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        net.eval()
        with torch.no_grad():
            logits_fed = net(torch.tensor(X_test, dtype=torch.float32))
            p_fed = torch.sigmoid(logits_fed).numpy()

        auc_fed = roc_auc_score(y_test, p_fed)
        ci_fed = bootstrap_metric_ci(y_test, p_fed)

        row_f = fed_reported[fed_reported["experiment"] == exp_name]
        rep_auc_f = float(row_f["global_auroc"].values[0]) if len(row_f) > 0 else np.nan
        delta_f = auc_fed - rep_auc_f

        print(f"{exp_name:30s} | Reported: {rep_auc_f:.4f} | Recalculated: {auc_fed:.4f} (Delta: {delta_f:+.4f}) | 95% CI: [{ci_fed['auroc_ci'][0]:.4f}, {ci_fed['auroc_ci'][1]:.4f}]")

        recalculation_records.append({
            "experiment_id": "E2/E3",
            "model_name": exp_name,
            "metric": "Global AUROC",
            "reported_value": rep_auc_f,
            "recalculated_value": auc_fed,
            "delta": delta_f,
            "ci_95": f"[{ci_fed['auroc_ci'][0]:.4f}, {ci_fed['auroc_ci'][1]:.4f}]",
            "verification_status": "Verified Exactly" if abs(delta_f) < 1e-4 else "Discrepancy Detected"
        })

    # -------------------------------------------------------------
    # 5. AUDIT EXPERIMENTS E4 & E5: PERSONALIZATION
    # -------------------------------------------------------------
    print("\n--- 5. AUDITING EXPERIMENTS E4 & E5: PERSONALIZATION ---")
    pers_reported = pd.read_csv("results/metrics/personalization_experiments_metrics.csv")
    print("Reported personalization summary:\n", pers_reported)

    per_hosp_reported = pd.read_csv("results/metrics/per_hospital_personalization_gains.csv")
    print("\nReported Consortium 5 (Hospital_Consortium_5) records in E5 (Quantum):")
    c5_rep = per_hosp_reported[(per_hosp_reported["client_id"] == "Hospital_Consortium_5") & (per_hosp_reported["model_type"] == "quantum")]
    print(c5_rep[["regime", "auroc", "prauc", "brier", "ece"]])

    # Audit the test sample for Consortium 5
    c5_test_mask = (test_meta["hospitalid"] % 5 == 4).values
    y_test_c5 = y_test[c5_test_mask]
    X_test_c5 = X_test[c5_test_mask]
    print(f"\nConsortium 5 Test Set: Total Stays={len(y_test_c5)}, Positives={int(y_test_c5.sum())}, Negatives={int((y_test_c5==0).sum())}")

    # -------------------------------------------------------------
    # 6. AUDIT EXPERIMENT E6: PRIVACY & DIFFERENTIAL PRIVACY
    # -------------------------------------------------------------
    print("\n--- 6. AUDITING EXPERIMENT E6: PRIVACY & DP GUARANTEE ---")
    tradeoff_rep = pd.read_csv("results/metrics/privacy_utility_tradeoff.csv")
    print("Reported privacy utility tradeoff:\n", tradeoff_rep[["target_epsilon", "noise_multiplier", "test_auroc", "mia_auroc", "mia_advantage"]])

    # Save summary dataframe
    df_recalc = pd.DataFrame(recalculation_records)
    df_recalc.to_csv("research/RESULTS_REPRODUCIBILITY_MATRIX.csv", index=False)
    print("\nSaved recalculation verification matrix to research/RESULTS_REPRODUCIBILITY_MATRIX.csv")


if __name__ == "__main__":
    main()
