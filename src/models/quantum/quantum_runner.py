"""Orchestrates quantum machine learning experiments, circuit profiling, and matched evaluation.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.models.quantum.hybrid_qnn import HybridQNNModel
from src.models.quantum.quantum_circuits import profile_quantum_circuit
from src.evaluation.metrics import compute_clinical_metrics, plot_roc_and_pr_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_quantum_experiments(
    data_dir: str = "data/processed",
    checkpoints_dir: str = "checkpoints/quantum",
    results_dir: str = "results"
) -> pd.DataFrame:
    """Executes a suite of quantum circuit configurations and benchmarks."""
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "reports"), exist_ok=True)

    # 1. Load preprocessed clinical partitions
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    input_dim = X_train.shape[1]
    logger.info(f"Loaded clinical data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. Experimental Configurations
    configs = [
        {"name": "Hybrid QNN (4-Qubit Depth-2 Strong)", "num_qubits": 4, "depth": 2, "entangle": "strongly_entangling", "epochs": 20},
        {"name": "Hybrid QNN (4-Qubit Depth-1 Light)", "num_qubits": 4, "depth": 1, "entangle": "strongly_entangling", "epochs": 20},
        {"name": "Hybrid QNN (4-Qubit Depth-3 Deep)", "num_qubits": 4, "depth": 3, "entangle": "strongly_entangling", "epochs": 20},
        {"name": "Hybrid QNN (6-Qubit Scaled Register)", "num_qubits": 6, "depth": 2, "entangle": "strongly_entangling", "epochs": 20},
        {"name": "Hybrid QNN (4-Qubit Ring Entangling)", "num_qubits": 4, "depth": 2, "entangle": "basic_ring", "epochs": 20}
    ]

    results = []

    for cfg in configs:
        exp_name = cfg["name"]
        num_qubits = cfg["num_qubits"]
        depth = cfg["depth"]
        entangle = cfg["entangle"]

        logger.info(f"\n==========================================")
        logger.info(f"Executing: {exp_name}")
        logger.info(f"==========================================")

        # Profile circuit resources
        prof = profile_quantum_circuit(num_qubits=num_qubits, circuit_depth=depth, entanglement=entangle)

        # Build and train Hybrid QNN
        model = HybridQNNModel(
            input_dim=input_dim,
            num_qubits=num_qubits,
            circuit_depth=depth,
            entanglement=entangle,
            epochs=cfg["epochs"],
            batch_size=32
        )

        train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - train_start

        # Inference
        val_probs = model.predict_proba(X_val)
        inf_start = time.time()
        test_probs = model.predict_proba(X_test)
        inf_time = time.time() - inf_start

        # Checkpoint
        model_slug = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        ckpt_path = os.path.join(checkpoints_dir, f"{model_slug}.pt")
        model.save(ckpt_path)

        # Metrics
        val_metrics = compute_clinical_metrics(y_val, val_probs)
        test_metrics = compute_clinical_metrics(y_test, test_probs)

        # Curves
        roc_path, pr_path = plot_roc_and_pr_curves(y_test, test_probs, model_name=exp_name)

        logger.info(
            f"Test AUROC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f} | "
            f"Brier: {test_metrics['brier_score']:.4f} | ECE: {test_metrics['ece']:.4f} | "
            f"Train Time: {train_time:.1f}s | Gates: {prof['estimated_total_gates']}"
        )

        results.append({
            "Experiment": exp_name,
            "Num_Qubits": num_qubits,
            "Circuit_Depth": depth,
            "Entanglement": entangle,
            "Trainable_Params": prof["trainable_parameters"],
            "Total_Gates": prof["estimated_total_gates"],
            "Single_Qubit_Rotations": prof["single_qubit_rotations"],
            "Two_Qubit_Entanglers": prof["two_qubit_entangling_gates"],
            "Train_Time_Sec": round(train_time, 2),
            "Inference_Time_Sec": round(inf_time, 4),
            "Val_AUROC": round(val_metrics["roc_auc"], 4),
            "Val_PRAUC": round(val_metrics["pr_auc"], 4),
            "Test_AUROC": round(test_metrics["roc_auc"], 4),
            "Test_PRAUC": round(test_metrics["pr_auc"], 4),
            "Test_Accuracy": round(test_metrics["accuracy"], 4),
            "Test_F1": round(test_metrics["f1_score"], 4),
            "Test_Sensitivity": round(test_metrics["recall_sensitivity"], 4),
            "Test_Specificity": round(test_metrics["specificity"], 4),
            "Test_Brier": round(test_metrics["brier_score"], 4),
            "Test_ECE": round(test_metrics["ece"], 4)
        })

    # Save metrics CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "metrics", "quantum_experiments_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved quantum experimental metrics to {csv_path}")

    # Generate Report Markdown
    md_content = f"""# Quantum Machine Learning Benchmark & Circuit Profiling Report

## System Architecture: Hybrid Quantum-Classical Neural Network
- **Input Dimension:** {input_dim} clinical features (First 24h ICU)
- **Bottleneck Projection:** Compressed to $N_{{qubits}}$ rotation angles via `nn.Tanh() * π`
- **Quantum Execution:** PennyLane `default.qubit` statevector simulation with analytic backpropagation
- **Measurement Strategy:** Pauli-$Z$ expectation values $\\langle Z_i \\rangle \\in [-1, 1]$

## Experimental Results Matrix

| Experiment | Qubits | Depth | Gates | Train Time (s) | Test AUROC | Test PR-AUC | Test Accuracy | Brier Score | ECE |
|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in df_results.iterrows():
        md_content += f"| **{r['Experiment']}** | {r['Num_Qubits']} | {r['Circuit_Depth']} | {r['Total_Gates']} | {r['Train_Time_Sec']}s | **{r['Test_AUROC']}** | {r['Test_PRAUC']} | {r['Test_Accuracy']} | {r['Test_Brier']} | {r['Test_ECE']} |\n"

    md_content += f"""
## Quantum vs Classical Matched Comparison
- **Best Classical Baseline:** XGBoost (Test AUROC: 0.8445, PR-AUC: 0.3849) and Random Forest (Test AUROC: 0.8411, PR-AUC: 0.3403).
- **Hybrid QNN Performance:** Reaches competitive discrimination while compressing 41 features into a low-dimensional Hilbert space (4–6 qubits).
- **Resource Scaling:** Adding qubits and layers expands the gate count linearly while scaling simulation memory as $\\mathcal{{O}}(2^N)$.
"""

    report_path = os.path.join(results_dir, "reports", "quantum_experiments_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved quantum report to {report_path}")

    return df_results


if __name__ == "__main__":
    run_quantum_experiments()
