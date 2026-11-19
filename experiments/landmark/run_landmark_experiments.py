"""
Phase 13.1 Landmark Experiment Suite.
Retrains and re-evaluates all models on the audited 24-hour landmark clinical cohort:
1. Centralized Classical Baselines (Logistic Regression, Linear SVM, Random Forest, MLP, XGBoost).
2. Centralized Hybrid QNN (4 qubits, depth 2 strongly entangling).
3. Federated Learning (FedAvg and FedProx for Classical MLP and Hybrid QNN).
4. Formal DP-SGD (Classical MLP and Hybrid QNN across epsilon in {1.0, 2.0, 4.0, 8.0, inf}).
5. Hospital Personalization (Global vs Fine-Tuned).
6. Statistical Validation: Full patient-level bootstrap (B=1000) 95% Confidence Intervals.
"""

import copy
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.svm import LinearSVC
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.classical.baselines import ClinicalMLPNet
from src.models.quantum.hybrid_qnn import HybridQNNNet
from src.privacy.formal_dp_engine import (
    ClinicalDPMLPNet,
    DPHybridQNNNet,
    FormalOpacusMLPTrainer,
    FormalPerSampleDPQNNTrainer
)
from src.federated.advanced_partitioner import partition_dirichlet_clients
from src.federated.communication import ClientUpdateMessage
from src.federated.strategies import FedAvgStrategy, FedProxStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase13_1_LandmarkExperiments")


# ---------------------------------------------------------------------------
# PATIENT-LEVEL BOOTSTRAP STATISTICAL VALIDATION
# ---------------------------------------------------------------------------

def compute_metrics_with_bootstrap(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    patient_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Computes clinical metrics and patient-level bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    y_probs = np.clip(y_probs, 1e-6, 1.0 - 1e-6)
    y_preds = (y_probs >= 0.5).astype(int)

    # Point estimates
    auc = roc_auc_score(y_true, y_probs)
    prc = average_precision_score(y_true, y_probs)
    brier = brier_score_loss(y_true, y_probs)
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    
    # Specificity
    tn = np.sum((y_true == 0) & (y_preds == 0))
    fp = np.sum((y_true == 0) & (y_preds == 1))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Group by patient for patient-level bootstrap
    unique_pts = np.unique(patient_ids)
    n_pts = len(unique_pts)
    pt_to_indices = {pt: np.where(patient_ids == pt)[0] for pt in unique_pts}

    boot_aucs, boot_prcs, boot_briers = [], [], []
    boot_accs, boot_precs, boot_recs, boot_specs, boot_f1s = [], [], [], [], []

    for _ in range(n_bootstraps):
        sampled_pts = rng.choice(unique_pts, size=n_pts, replace=True)
        sample_indices = np.concatenate([pt_to_indices[pt] for pt in sampled_pts])

        b_true = y_true[sample_indices]
        b_probs = y_probs[sample_indices]
        b_preds = (b_probs >= 0.5).astype(int)

        if len(np.unique(b_true)) < 2:
            continue

        boot_aucs.append(roc_auc_score(b_true, b_probs))
        boot_prcs.append(average_precision_score(b_true, b_probs))
        boot_briers.append(brier_score_loss(b_true, b_probs))
        boot_accs.append(accuracy_score(b_true, b_preds))
        boot_precs.append(precision_score(b_true, b_preds, zero_division=0))
        boot_recs.append(recall_score(b_true, b_preds, zero_division=0))
        boot_f1s.append(f1_score(b_true, b_preds, zero_division=0))

        b_tn = np.sum((b_true == 0) & (b_preds == 0))
        b_fp = np.sum((b_true == 0) & (b_preds == 1))
        boot_specs.append(float(b_tn / (b_tn + b_fp)) if (b_tn + b_fp) > 0 else 0.0)

    def get_ci(arr: List[float]) -> Tuple[float, float]:
        if len(arr) == 0:
            return (0.0, 0.0)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    auc_ci = get_ci(boot_aucs)
    prc_ci = get_ci(boot_prcs)
    brier_ci = get_ci(boot_briers)
    acc_ci = get_ci(boot_accs)
    prec_ci = get_ci(boot_precs)
    rec_ci = get_ci(boot_recs)
    spec_ci = get_ci(boot_specs)
    f1_ci = get_ci(boot_f1s)

    return {
        "N_total": len(y_true),
        "N_pos": int(np.sum(y_true == 1)),
        "N_neg": int(np.sum(y_true == 0)),
        "Prevalence": float(np.mean(y_true)),
        "AUROC": float(auc),
        "AUROC_CI_Lower": auc_ci[0],
        "AUROC_CI_Upper": auc_ci[1],
        "AUPRC": float(prc),
        "AUPRC_CI_Lower": prc_ci[0],
        "AUPRC_CI_Upper": prc_ci[1],
        "Brier": float(brier),
        "Brier_CI_Lower": brier_ci[0],
        "Brier_CI_Upper": brier_ci[1],
        "Accuracy": float(acc),
        "Accuracy_CI_Lower": acc_ci[0],
        "Accuracy_CI_Upper": acc_ci[1],
        "Sensitivity": float(rec),
        "Sensitivity_CI_Lower": rec_ci[0],
        "Sensitivity_CI_Upper": rec_ci[1],
        "Specificity": float(spec),
        "Specificity_CI_Lower": spec_ci[0],
        "Specificity_CI_Upper": spec_ci[1],
        "Precision": float(prec),
        "Precision_CI_Lower": prec_ci[0],
        "Precision_CI_Upper": prec_ci[1],
        "Recall": float(rec),
        "Recall_CI_Lower": rec_ci[0],
        "Recall_CI_Upper": rec_ci[1],
        "F1": float(f1),
        "F1_CI_Lower": f1_ci[0],
        "F1_CI_Upper": f1_ci[1]
    }


# ---------------------------------------------------------------------------
# EXPERIMENT EXECUTION ENGINE
# ---------------------------------------------------------------------------

class LandmarkExperimentSuite:
    """Orchestrates comprehensive retraining and re-evaluation on landmark cohort."""

    def __init__(
        self,
        data_dir: str = "data/processed_landmark",
        results_dir: str = "results/phase13_1"
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.metrics_dir = os.path.join(results_dir, "metrics")
        self.checkpoints_dir = os.path.join(results_dir, "checkpoints")

        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        logger.info(f"Loading landmark dataset from {self.data_dir}...")
        self.X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        self.y_train = np.load(os.path.join(self.data_dir, "y_train.npy"))
        self.X_val = np.load(os.path.join(self.data_dir, "X_val.npy"))
        self.y_val = np.load(os.path.join(self.data_dir, "y_val.npy"))
        self.X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))

        self.train_meta = pd.read_csv(os.path.join(self.data_dir, "train_meta.csv"))
        self.val_meta = pd.read_csv(os.path.join(self.data_dir, "val_meta.csv"))
        self.test_meta = pd.read_csv(os.path.join(self.data_dir, "test_meta.csv"))
        self.test_pts = self.test_meta["uniquepid"].values

        logger.info(
            f"Landmark Cohort Loaded: Train={len(self.X_train)} (deaths={self.y_train.sum()}), "
            f"Val={len(self.X_val)} (deaths={self.y_val.sum()}), "
            f"Test={len(self.X_test)} (deaths={self.y_test.sum()}). Features={self.X_train.shape[1]}"
        )

    def run_centralized_baselines(self) -> pd.DataFrame:
        """Trains and evaluates centralized classical models and Hybrid QNN."""
        logger.info("=== Running Centralized Baselines on Landmark Cohort ===")
        records = []

        # 1. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        lr.fit(self.X_train, self.y_train)
        probs_lr = lr.predict_proba(self.X_test)[:, 1]
        m_lr = compute_metrics_with_bootstrap(self.y_test, probs_lr, self.test_pts)
        m_lr["Model"] = "Logistic Regression"
        m_lr["Paradigm"] = "Centralized Classical"
        records.append(m_lr)
        joblib.dump(lr, os.path.join(self.checkpoints_dir, "logistic_regression.joblib"))

        # 2. Linear SVM with Platt Calibration
        logger.info("Training Linear SVM...")
        svm_base = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
        svm = CalibratedClassifierCV(estimator=svm_base, cv=3)
        svm.fit(self.X_train, self.y_train)
        probs_svm = svm.predict_proba(self.X_test)[:, 1]
        m_svm = compute_metrics_with_bootstrap(self.y_test, probs_svm, self.test_pts)
        m_svm["Model"] = "Linear SVM"
        m_svm["Paradigm"] = "Centralized Classical"
        records.append(m_svm)
        joblib.dump(svm, os.path.join(self.checkpoints_dir, "linear_svm.joblib"))

        # 3. Random Forest
        logger.info("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        probs_rf = rf.predict_proba(self.X_test)[:, 1]
        m_rf = compute_metrics_with_bootstrap(self.y_test, probs_rf, self.test_pts)
        m_rf["Model"] = "Random Forest"
        m_rf["Paradigm"] = "Centralized Classical"
        records.append(m_rf)
        joblib.dump(rf, os.path.join(self.checkpoints_dir, "random_forest.joblib"))

        # 4. Multi-Layer Perceptron (Tabular PyTorch)
        logger.info("Training PyTorch Clinical MLP...")
        mlp = ClinicalMLPNet(
            input_dim=self.X_train.shape[1],
            hidden_dims=[64, 32, 16],
            dropout=0.20
        )
        pos_weight = torch.tensor([float((self.y_train == 0).sum() / max(1, (self.y_train == 1).sum()))])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.005)

        train_ds = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32)
        )
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        mlp.train()
        for epoch in range(25):
            for bx, by in train_dl:
                optimizer.zero_grad()
                out = mlp(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        mlp.eval()
        with torch.no_grad():
            x_test_t = torch.tensor(self.X_test, dtype=torch.float32)
            probs_mlp = torch.sigmoid(mlp(x_test_t)).numpy()

        m_mlp = compute_metrics_with_bootstrap(self.y_test, probs_mlp, self.test_pts)
        m_mlp["Model"] = "Clinical MLP"
        m_mlp["Paradigm"] = "Centralized Classical"
        records.append(m_mlp)
        torch.save(mlp.state_dict(), os.path.join(self.checkpoints_dir, "clinical_mlp.pt"))

        # 5. XGBoost
        logger.info("Training XGBoost...")
        scale_pos = (self.y_train == 0).sum() / max(1, (self.y_train == 1).sum())
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        probs_xgb = xgb_model.predict_proba(self.X_test)[:, 1]
        m_xgb = compute_metrics_with_bootstrap(self.y_test, probs_xgb, self.test_pts)
        m_xgb["Model"] = "XGBoost"
        m_xgb["Paradigm"] = "Centralized Classical"
        records.append(m_xgb)
        joblib.dump(xgb_model, os.path.join(self.checkpoints_dir, "xgboost.joblib"))

        # 6. Centralized Hybrid QNN
        logger.info("Training Centralized Hybrid QNN (4 qubits, depth 2)...")
        qnn = HybridQNNNet(
            input_dim=self.X_train.shape[1],
            num_qubits=4,
            circuit_depth=2,
            entanglement="strongly_entangling"
        )
        q_opt = torch.optim.Adam(qnn.parameters(), lr=0.005)
        q_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        qnn.train()
        for epoch in range(12):
            for bx, by in q_dl:
                q_opt.zero_grad()
                out = qnn(bx)
                loss = criterion(out, by)
                loss.backward()
                q_opt.step()

        qnn.eval()
        with torch.no_grad():
            probs_qnn = torch.sigmoid(qnn(x_test_t)).numpy()

        m_qnn = compute_metrics_with_bootstrap(self.y_test, probs_qnn, self.test_pts)
        m_qnn["Model"] = "Hybrid QNN (4-Qubit, Depth 2)"
        m_qnn["Paradigm"] = "Centralized Quantum"
        records.append(m_qnn)
        torch.save(qnn.state_dict(), os.path.join(self.checkpoints_dir, "hybrid_qnn.pt"))

        df_cent = pd.DataFrame(records)
        df_cent.to_csv(os.path.join(self.metrics_dir, "centralized_metrics.csv"), index=False)
        logger.info(f"Centralized baselines completed. Saved to {self.metrics_dir}/centralized_metrics.csv")
        return df_cent

    def run_federated_experiments(self) -> pd.DataFrame:
        """Executes FedAvg and FedProx across Dirichlet hospital clients."""
        logger.info("=== Running Federated Learning on Landmark Cohort ===")
        input_dim = self.X_train.shape[1]
        x_test_t = torch.tensor(self.X_test, dtype=torch.float32)

        # Partition landmark train set into 5 hospital clients (alpha=0.5)
        clients, _, _ = partition_dirichlet_clients(
            model_factory=lambda: ClinicalMLPNet(input_dim=input_dim, hidden_dims=[64, 32, 16]),
            data_dir=self.data_dir,
            num_clients=5,
            alpha=0.5,
            seed=42
        )

        rounds = 10
        local_epochs = 2
        batch_size = 32
        records = []

        # A. Classical FedAvg
        logger.info("[Federated] Running Classical FedAvg...")
        global_mlp_fedavg = ClinicalMLPNet(input_dim=input_dim, hidden_dims=[64, 32, 16])
        fedavg_strat = FedAvgStrategy()

        for r in range(rounds):
            client_updates = []
            for c_id, client in clients.items():
                local_m = copy.deepcopy(global_mlp_fedavg)
                opt = torch.optim.Adam(local_m.parameters(), lr=0.005)
                ds = TensorDataset(
                    torch.tensor(client.X_train, dtype=torch.float32),
                    torch.tensor(client.y_train, dtype=torch.float32)
                )
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
                pos_w = torch.tensor([float((client.y_train == 0).sum() / max(1, (client.y_train == 1).sum()))])
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                local_m.train()
                for _ in range(local_epochs):
                    for bx, by in dl:
                        opt.zero_grad()
                        loss = crit(local_m(bx), by)
                        loss.backward()
                        opt.step()

                client_updates.append(ClientUpdateMessage(
                    client_id=c_id,
                    round_idx=r,
                    num_samples=len(client.X_train),
                    model_state_dict=copy.deepcopy(local_m.state_dict()),
                    train_loss=0.0,
                    local_val_metrics={},
                    local_test_metrics={}
                ))

            # Aggregate
            agg_weights = fedavg_strat.aggregate(client_updates)
            global_mlp_fedavg.load_state_dict(agg_weights)

        global_mlp_fedavg.eval()
        with torch.no_grad():
            probs_fedavg = torch.sigmoid(global_mlp_fedavg(x_test_t)).numpy()

        m_fedavg = compute_metrics_with_bootstrap(self.y_test, probs_fedavg, self.test_pts)
        m_fedavg["Model"] = "FedAvg Classical MLP"
        m_fedavg["Paradigm"] = "Federated Classical"
        records.append(m_fedavg)
        torch.save(global_mlp_fedavg.state_dict(), os.path.join(self.checkpoints_dir, "fedavg_mlp.pt"))

        # B. Classical FedProx (mu = 0.01)
        logger.info("[Federated] Running Classical FedProx (mu=0.01)...")
        global_mlp_fedprox = ClinicalMLPNet(input_dim=input_dim, hidden_dims=[64, 32, 16])
        fedprox_strat = FedProxStrategy(mu=0.01)

        for r in range(rounds):
            client_updates = []
            global_params = [p.detach().clone() for p in global_mlp_fedprox.parameters()]

            for c_id, client in clients.items():
                local_m = copy.deepcopy(global_mlp_fedprox)
                opt = torch.optim.Adam(local_m.parameters(), lr=0.005)
                ds = TensorDataset(
                    torch.tensor(client.X_train, dtype=torch.float32),
                    torch.tensor(client.y_train, dtype=torch.float32)
                )
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
                pos_w = torch.tensor([float((client.y_train == 0).sum() / max(1, (client.y_train == 1).sum()))])
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                local_m.train()
                for _ in range(local_epochs):
                    for bx, by in dl:
                        opt.zero_grad()
                        loss = crit(local_m(bx), by)
                        prox = sum(torch.sum((p - gp) ** 2) for p, gp in zip(local_m.parameters(), global_params))
                        total_loss = loss + 0.01 / 2.0 * prox
                        total_loss.backward()
                        opt.step()

                client_updates.append(ClientUpdateMessage(
                    client_id=c_id,
                    round_idx=r,
                    num_samples=len(client.X_train),
                    model_state_dict=copy.deepcopy(local_m.state_dict()),
                    train_loss=0.0,
                    local_val_metrics={},
                    local_test_metrics={}
                ))

            agg_weights = fedprox_strat.aggregate(client_updates)
            global_mlp_fedprox.load_state_dict(agg_weights)

        global_mlp_fedprox.eval()
        with torch.no_grad():
            probs_fedprox = torch.sigmoid(global_mlp_fedprox(x_test_t)).numpy()

        m_fedprox = compute_metrics_with_bootstrap(self.y_test, probs_fedprox, self.test_pts)
        m_fedprox["Model"] = "FedProx Classical MLP"
        m_fedprox["Paradigm"] = "Federated Classical"
        records.append(m_fedprox)
        torch.save(global_mlp_fedprox.state_dict(), os.path.join(self.checkpoints_dir, "fedprox_mlp.pt"))

        # C. Hybrid QNN FedAvg
        logger.info("[Federated] Running Hybrid QNN FedAvg...")
        global_qnn_fedavg = HybridQNNNet(input_dim=input_dim, num_qubits=4, circuit_depth=2)

        for r in range(rounds):
            client_updates = []
            for c_id, client in clients.items():
                local_m = copy.deepcopy(global_qnn_fedavg)
                opt = torch.optim.Adam(local_m.parameters(), lr=0.005)
                ds = TensorDataset(
                    torch.tensor(client.X_train, dtype=torch.float32),
                    torch.tensor(client.y_train, dtype=torch.float32)
                )
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
                pos_w = torch.tensor([float((client.y_train == 0).sum() / max(1, (client.y_train == 1).sum()))])
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                local_m.train()
                for _ in range(local_epochs):
                    for bx, by in dl:
                        opt.zero_grad()
                        loss = crit(local_m(bx), by)
                        loss.backward()
                        opt.step()

                client_updates.append(ClientUpdateMessage(
                    client_id=c_id,
                    round_idx=r,
                    num_samples=len(client.X_train),
                    model_state_dict=copy.deepcopy(local_m.state_dict()),
                    train_loss=0.0,
                    local_val_metrics={},
                    local_test_metrics={}
                ))

            agg_weights = fedavg_strat.aggregate(client_updates)
            global_qnn_fedavg.load_state_dict(agg_weights)

        global_qnn_fedavg.eval()
        with torch.no_grad():
            probs_qnn_fedavg = torch.sigmoid(global_qnn_fedavg(x_test_t)).numpy()

        m_qnn_fedavg = compute_metrics_with_bootstrap(self.y_test, probs_qnn_fedavg, self.test_pts)
        m_qnn_fedavg["Model"] = "FedAvg Hybrid QNN"
        m_qnn_fedavg["Paradigm"] = "Federated Quantum"
        records.append(m_qnn_fedavg)
        torch.save(global_qnn_fedavg.state_dict(), os.path.join(self.checkpoints_dir, "fedavg_qnn.pt"))

        # D. Hybrid QNN FedProx (mu = 0.01)
        logger.info("[Federated] Running Hybrid QNN FedProx (mu=0.01)...")
        global_qnn_fedprox = HybridQNNNet(input_dim=input_dim, num_qubits=4, circuit_depth=2)

        for r in range(rounds):
            client_updates = []
            global_params = [p.detach().clone() for p in global_qnn_fedprox.parameters()]

            for c_id, client in clients.items():
                local_m = copy.deepcopy(global_qnn_fedprox)
                opt = torch.optim.Adam(local_m.parameters(), lr=0.005)
                ds = TensorDataset(
                    torch.tensor(client.X_train, dtype=torch.float32),
                    torch.tensor(client.y_train, dtype=torch.float32)
                )
                dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
                pos_w = torch.tensor([float((client.y_train == 0).sum() / max(1, (client.y_train == 1).sum()))])
                crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

                local_m.train()
                for _ in range(local_epochs):
                    for bx, by in dl:
                        opt.zero_grad()
                        loss = crit(local_m(bx), by)
                        prox = sum(torch.sum((p - gp) ** 2) for p, gp in zip(local_m.parameters(), global_params))
                        total_loss = loss + 0.01 / 2.0 * prox
                        total_loss.backward()
                        opt.step()

                client_updates.append(ClientUpdateMessage(
                    client_id=c_id,
                    round_idx=r,
                    num_samples=len(client.X_train),
                    model_state_dict=copy.deepcopy(local_m.state_dict()),
                    train_loss=0.0,
                    local_val_metrics={},
                    local_test_metrics={}
                ))

            agg_weights = fedprox_strat.aggregate(client_updates)
            global_qnn_fedprox.load_state_dict(agg_weights)

        global_qnn_fedprox.eval()
        with torch.no_grad():
            probs_qnn_fedprox = torch.sigmoid(global_qnn_fedprox(x_test_t)).numpy()

        m_qnn_fedprox = compute_metrics_with_bootstrap(self.y_test, probs_qnn_fedprox, self.test_pts)
        m_qnn_fedprox["Model"] = "FedProx Hybrid QNN"
        m_qnn_fedprox["Paradigm"] = "Federated Quantum"
        records.append(m_qnn_fedprox)
        torch.save(global_qnn_fedprox.state_dict(), os.path.join(self.checkpoints_dir, "fedprox_qnn.pt"))

        df_fed = pd.DataFrame(records)
        df_fed.to_csv(os.path.join(self.metrics_dir, "federated_metrics.csv"), index=False)
        logger.info(f"Federated experiments completed. Saved to {self.metrics_dir}/federated_metrics.csv")
        return df_fed

    def run_privacy_utility_experiments(self) -> pd.DataFrame:
        """Evaluates formal DP-SGD across epsilon in {1, 2, 4, 8, inf} on landmark cohort."""
        logger.info("=== Running Formal DP-SGD on Landmark Cohort ===")
        budgets = [1.0, 2.0, 4.0, 8.0, float("inf")]
        delta = 1e-5
        input_dim = self.X_train.shape[1]
        x_test_t = torch.tensor(self.X_test, dtype=torch.float32)

        records = []

        # 1. Classical MLP with Opacus PrivacyEngine (RDPAccountant, Poisson subsampling)
        mlp_trainer = FormalOpacusMLPTrainer(
            input_dim=input_dim,
            hidden_dims=[64, 32, 16],
            lr=0.005,
            max_grad_norm=1.0,
            epochs=15,
            batch_size=64,
            target_delta=delta
        )

        for eps in budgets:
            eps_lbl = "inf" if math.isinf(eps) else f"{eps:.1f}"
            logger.info(f"[Formal DP] Classical MLP training with target epsilon={eps_lbl}...")
            model_dp, ach_eps, ach_delta = mlp_trainer.train(
                X_train=self.X_train,
                y_train=self.y_train,
                target_epsilon=eps
            )
            model_dp.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model_dp(x_test_t)).numpy()

            m = compute_metrics_with_bootstrap(self.y_test, probs, self.test_pts)
            m["Model"] = f"DP-MLP (eps={eps_lbl})"
            m["Base_Model"] = "Classical MLP"
            m["Target_Epsilon"] = eps_lbl
            m["Achieved_Epsilon"] = ach_eps
            m["Delta"] = ach_delta
            m["Sampling_Mechanism"] = "Poisson Subsampling (UniformWithReplacementSampler)"
            m["Accountant"] = "Rényi Differential Privacy (RDPAccountant)"
            m["Adjacency"] = "Add/Remove-One-Record (Sensitivity Delta_2 = C = 1.0)"
            records.append(m)

        # 2. Hybrid QNN with FormalPerSampleDPQNNTrainer (Poisson subsampling, RDPAccountant)
        qnn_trainer = FormalPerSampleDPQNNTrainer(
            input_dim=input_dim,
            num_qubits=4,
            circuit_depth=2,
            lr=0.005,
            max_grad_norm=1.0,
            epochs=5,
            batch_size=32,
            target_delta=delta
        )

        for eps in budgets:
            eps_lbl = "inf" if math.isinf(eps) else f"{eps:.1f}"
            logger.info(f"[Formal DP] Hybrid QNN training with target epsilon={eps_lbl}...")
            qnn_dp, ach_eps, ach_delta = qnn_trainer.train(
                X_train=self.X_train,
                y_train=self.y_train,
                target_epsilon=eps
            )
            qnn_dp.eval()
            with torch.no_grad():
                probs = torch.sigmoid(qnn_dp(x_test_t)).numpy()

            m = compute_metrics_with_bootstrap(self.y_test, probs, self.test_pts)
            m["Model"] = f"DP-QNN (eps={eps_lbl})"
            m["Base_Model"] = "Hybrid QNN"
            m["Target_Epsilon"] = eps_lbl
            m["Achieved_Epsilon"] = ach_eps
            m["Delta"] = ach_delta
            m["Sampling_Mechanism"] = "Poisson Subsampling (UniformWithReplacementSampler)"
            m["Accountant"] = "Rényi Differential Privacy (RDPAccountant)"
            m["Adjacency"] = "Add/Remove-One-Record (Sensitivity Delta_2 = C = 1.0)"
            records.append(m)

        df_dp = pd.DataFrame(records)
        df_dp.to_csv(os.path.join(self.metrics_dir, "privacy_utility_metrics.csv"), index=False)
        logger.info(f"Privacy-utility experiments completed. Saved to {self.metrics_dir}/privacy_utility_metrics.csv")
        return df_dp

    def run_personalization_experiments(self) -> pd.DataFrame:
        """Evaluates hospital-specific fine-tuning personalization vs global model."""
        logger.info("=== Running Hospital Personalization Experiments on Landmark Cohort ===")
        input_dim = self.X_train.shape[1]

        # Hospital client partition
        clients, _, _ = partition_dirichlet_clients(
            model_factory=lambda: ClinicalMLPNet(input_dim=input_dim, hidden_dims=[64, 32, 16]),
            data_dir=self.data_dir,
            num_clients=5,
            alpha=0.5,
            seed=42
        )

        # Pre-trained global FedAvg model
        fedavg_model = ClinicalMLPNet(input_dim=input_dim, hidden_dims=[64, 32, 16])
        fedavg_path = os.path.join(self.checkpoints_dir, "fedavg_mlp.pt")
        if os.path.exists(fedavg_path):
            fedavg_model.load_state_dict(torch.load(fedavg_path))

        records = []
        # Evaluate global model vs personalized fine-tuned model on test set
        fedavg_model.eval()
        with torch.no_grad():
            x_test_t = torch.tensor(self.X_test, dtype=torch.float32)
            probs_global = torch.sigmoid(fedavg_model(x_test_t)).numpy()

        m_global = compute_metrics_with_bootstrap(self.y_test, probs_global, self.test_pts)
        m_global["Strategy"] = "Global Federated (No Personalization)"
        records.append(m_global)

        # Personalize by fine-tuning on client 0 data (representative hospital) for 3 epochs with low lr
        local_pers = copy.deepcopy(fedavg_model)
        c0 = clients["client_0"]
        c0_ds = TensorDataset(
            torch.tensor(c0.X_train, dtype=torch.float32),
            torch.tensor(c0.y_train, dtype=torch.float32)
        )
        c0_dl = DataLoader(c0_ds, batch_size=32, shuffle=True)
        pos_w = torch.tensor([float((c0.y_train == 0).sum() / max(1, (c0.y_train == 1).sum()))])
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(local_pers.parameters(), lr=0.001)

        local_pers.train()
        for _ in range(3):
            for bx, by in c0_dl:
                opt.zero_grad()
                loss = crit(local_pers(bx), by)
                loss.backward()
                opt.step()

        local_pers.eval()
        with torch.no_grad():
            probs_pers = torch.sigmoid(local_pers(x_test_t)).numpy()

        m_pers = compute_metrics_with_bootstrap(self.y_test, probs_pers, self.test_pts)
        m_pers["Strategy"] = "Local Fine-Tuned Personalization"
        records.append(m_pers)

        df_pers = pd.DataFrame(records)
        df_pers.to_csv(os.path.join(self.metrics_dir, "personalization_metrics.csv"), index=False)
        logger.info(f"Personalization experiments completed. Saved to {self.metrics_dir}/personalization_metrics.csv")
        return df_pers

    def run_all(self):
        """Runs the complete landmark experimental protocol."""
        t0 = time.time()
        cent_path = os.path.join(self.metrics_dir, "centralized_metrics.csv")
        if os.path.exists(cent_path):
            logger.info("Loading existing verified centralized metrics...")
            df_cent = pd.read_csv(cent_path)
        else:
            df_cent = self.run_centralized_baselines()
        df_fed = self.run_federated_experiments()
        df_dp = self.run_privacy_utility_experiments()
        df_pers = self.run_personalization_experiments()

        # Combine all experiments into unified master table
        all_dfs = []
        for df, name in [
            (df_cent, "Centralized"),
            (df_fed, "Federated"),
            (df_dp, "Privacy-Utility"),
            (df_pers, "Personalization")
        ]:
            d = df.copy()
            d["Experiment_Family"] = name
            all_dfs.append(d)

        df_all = pd.concat(all_dfs, ignore_index=True)
        master_path = os.path.join(self.metrics_dir, "all_landmark_experiments.csv")
        df_all.to_csv(master_path, index=False)
        logger.info(f"Master landmark experiment suite complete in {time.time() - t0:.1f}s. Saved to {master_path}")


if __name__ == "__main__":
    suite = LandmarkExperimentSuite()
    suite.run_all()
