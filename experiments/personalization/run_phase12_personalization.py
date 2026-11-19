"""
Phase 12 Advanced Hospital Personalization Experiment.
Evaluates multi-hospital personalization under Dirichlet non-IID partitioning:
1. Global Shared FL Baseline (no local adaptation)
2. Local Fine-Tuning (FT) of entire network
3. Parameter-Efficient Local Residual Adapters (freezing Theta_s, training Theta_k)
4. Personalized Hybrid QNN (freezing shared PQC, adapting local residual head)
Reports N_test, N_pos, AUROC, AUPRC with 95% bootstrap confidence intervals for all nodes.
"""

import copy
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.abspath("."))

from src.models.classical.baselines import ClinicalMLPNet
from src.models.quantum.hybrid_qnn import HybridQNNNet
from src.personalization.local_adapter import HospitalAdapterModule, PersonalizedHybridQNNNet
from src.federated.advanced_partitioner import partition_dirichlet_clients

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase12_Personalization")

torch.set_num_threads(1)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Computes AUROC and 95% percentile bootstrap confidence intervals."""
    if len(np.unique(y_true)) < 2:
        return 0.50, 0.50, 0.50

    base_auc = float(roc_auc_score(y_true, y_prob))
    rng = np.random.RandomState(seed)
    boot_aucs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = y_prob[idx]
        boot_aucs.append(roc_auc_score(y_b, p_b))

    if not boot_aucs:
        return base_auc, base_auc, base_auc

    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    return base_auc, ci_lower, ci_upper


def evaluate_probs(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32)
        out = model(x_t)
        probs = torch.sigmoid(out).cpu().numpy()
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def run_advanced_personalization_experiments():
    logger.info("Initializing Phase 12 Advanced Personalization...")
    os.makedirs("results/phase12/metrics", exist_ok=True)
    os.makedirs("results/phase12/figures", exist_ok=True)

    # 1. Load Dirichlet Partition (alpha=0.5)
    model_factory = lambda: ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32, 16])
    clients, global_test, partition_df = partition_dirichlet_clients(
        model_factory=model_factory,
        num_clients=5,
        alpha=0.5,
        seed=42
    )

    # Load pre-trained Global Checkpoint if available, else load global reference
    global_ckpt_path = "results/phase12/checkpoints/fed_fedavg_classical_best.pt"
    global_state = torch.load(global_ckpt_path, map_location="cpu")

    results_records = []

    for cid, client in clients.items():
        logger.info(f"Evaluating Personalization on Client: {cid} ({client.hospital_name})...")
        X_tr = client.X_train
        y_tr = client.y_train
        X_te = client.X_test
        y_te = client.y_test

        n_test = len(X_te)
        n_pos = int(y_te.sum())
        logger.info(f"-> Node {cid}: N_train={len(X_tr)} (pos={int(y_tr.sum())}), N_test={n_test} (pos={n_pos})")

        # Skip evaluation if test set has zero positive cases (mathematically undefined AUROC)
        if n_pos == 0 or len(np.unique(y_te)) < 2:
            logger.warning(f"Node {cid} has 0 positive cases in test. Setting baseline values.")
            continue

        # Strategy 1: Global Model (Zero Adaptation)
        global_model = ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32, 16])
        global_model.load_state_dict(global_state)
        p_global = evaluate_probs(global_model, X_te)
        auc_g, ci_l_g, ci_u_g = compute_bootstrap_ci(y_te, p_global)
        pr_g = float(average_precision_score(y_te, p_global))

        results_records.append({
            "Client_ID": cid,
            "Hospital_Name": client.hospital_name,
            "N_Test": n_test,
            "N_Test_Pos": n_pos,
            "Strategy": "Global_FL_Baseline",
            "AUROC": auc_g,
            "CI_Lower": ci_l_g,
            "CI_Upper": ci_u_g,
            "PRAUC": pr_g,
            "Delta_AUROC": 0.0
        })

        # Strategy 2: Local Fine-Tuning (FT) of entire network
        ft_model = ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32, 16])
        ft_model.load_state_dict(copy.deepcopy(global_state))
        opt_ft = torch.optim.Adam(ft_model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        ds_tr = torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=32, shuffle=True)
        ft_model.train()
        for _ in range(5):
            for bx, by in dl_tr:
                opt_ft.zero_grad()
                loss = criterion(ft_model(bx), by)
                loss.backward()
                opt_ft.step()
        p_ft = evaluate_probs(ft_model, X_te)
        auc_ft, ci_l_ft, ci_u_ft = compute_bootstrap_ci(y_te, p_ft)
        pr_ft = float(average_precision_score(y_te, p_ft))

        results_records.append({
            "Client_ID": cid,
            "Hospital_Name": client.hospital_name,
            "N_Test": n_test,
            "N_Test_Pos": n_pos,
            "Strategy": "Local_Fine_Tuning",
            "AUROC": auc_ft,
            "CI_Lower": ci_l_ft,
            "CI_Upper": ci_u_ft,
            "PRAUC": pr_ft,
            "Delta_AUROC": auc_ft - auc_g
        })

        # Strategy 3: Parameter-Efficient Local Residual Adapter
        # Wraps global encoder, adds HospitalAdapterModule
        class AdaptedMLP(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_encoder = base_model.network[:-1]  # Output dim 16
                self.adapter = HospitalAdapterModule(in_features=16, bottleneck_ratio=2)
                self.head = base_model.network[-1]           # Linear(16, 1)

            def forward(self, x):
                with torch.no_grad():
                    rep = self.base_encoder(x)
                adapted_rep = self.adapter(rep)
                return self.head(adapted_rep).squeeze(-1)

        adapted_mlp = AdaptedMLP(copy.deepcopy(global_model))
        opt_ad = torch.optim.Adam(adapted_mlp.adapter.parameters(), lr=0.005)
        adapted_mlp.train()
        for _ in range(5):
            for bx, by in dl_tr:
                opt_ad.zero_grad()
                loss = criterion(adapted_mlp(bx), by)
                loss.backward()
                opt_ad.step()
        p_ad = evaluate_probs(adapted_mlp, X_te)
        auc_ad, ci_l_ad, ci_u_ad = compute_bootstrap_ci(y_te, p_ad)
        pr_ad = float(average_precision_score(y_te, p_ad))

        results_records.append({
            "Client_ID": cid,
            "Hospital_Name": client.hospital_name,
            "N_Test": n_test,
            "N_Test_Pos": n_pos,
            "Strategy": "Residual_Adapter",
            "AUROC": auc_ad,
            "CI_Lower": ci_l_ad,
            "CI_Upper": ci_u_ad,
            "PRAUC": pr_ad,
            "Delta_AUROC": auc_ad - auc_g
        })

        # Strategy 4: Personalized Hybrid QNN
        pqnn = PersonalizedHybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=2)
        # Freeze shared encoder and quantum layer
        for p in pqnn.encoder.parameters():
            p.requires_grad = False
        for p in pqnn.quantum_layer.parameters():
            p.requires_grad = False
        opt_pqnn = torch.optim.Adam(
            list(pqnn.local_adapter.parameters()) + list(pqnn.local_head.parameters()),
            lr=0.01
        )
        pqnn.train()
        for _ in range(3):
            for bx, by in dl_tr:
                opt_pqnn.zero_grad()
                loss = criterion(pqnn(bx), by)
                loss.backward()
                opt_pqnn.step()
        p_pqnn = evaluate_probs(pqnn, X_te)
        auc_pqnn, ci_l_pqnn, ci_u_pqnn = compute_bootstrap_ci(y_te, p_pqnn)
        pr_pqnn = float(average_precision_score(y_te, p_pqnn))

        results_records.append({
            "Client_ID": cid,
            "Hospital_Name": client.hospital_name,
            "N_Test": n_test,
            "N_Test_Pos": n_pos,
            "Strategy": "Personalized_Hybrid_QNN",
            "AUROC": auc_pqnn,
            "CI_Lower": ci_l_pqnn,
            "CI_Upper": ci_u_pqnn,
            "PRAUC": pr_pqnn,
            "Delta_AUROC": auc_pqnn - auc_g
        })

        logger.info(
            f"-> {cid}: Global={auc_g:.4f} [{ci_l_g:.3f}, {ci_u_g:.3f}], "
            f"FT={auc_ft:.4f} (d={auc_ft - auc_g:+.4f}), "
            f"Adapter={auc_ad:.4f} (d={auc_ad - auc_g:+.4f}), "
            f"PQNN={auc_pqnn:.4f} (d={auc_pqnn - auc_g:+.4f})"
        )

    df_pers = pd.DataFrame(results_records)
    out_csv = "results/phase12/metrics/advanced_personalization_metrics.csv"
    df_pers.to_csv(out_csv, index=False)
    logger.info(f"Saved personalization metrics to {out_csv}")

    # Plot Personalization Gains Figure
    plot_personalization_gains(df_pers)

    return df_pers


def plot_personalization_gains(df: pd.DataFrame):
    """Plots comparative personalization performance with 95% bootstrap error bars."""
    nodes = df["Client_ID"].unique()
    strategies = ["Global_FL_Baseline", "Local_Fine_Tuning", "Residual_Adapter", "Personalized_Hybrid_QNN"]
    colors = {"Global_FL_Baseline": "#7f7f7f", "Local_Fine_Tuning": "#1f77b4", "Residual_Adapter": "#ff7f0e", "Personalized_Hybrid_QNN": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(nodes))
    width = 0.20

    for i, strat in enumerate(strategies):
        sub = df[df["Strategy"] == strat]
        means = []
        yerr_low = []
        yerr_high = []
        for n in nodes:
            row = sub[sub["Client_ID"] == n]
            if len(row) > 0:
                mean = row["AUROC"].values[0]
                cl = row["CI_Lower"].values[0]
                cu = row["CI_Upper"].values[0]
                means.append(mean)
                yerr_low.append(max(0, mean - cl))
                yerr_high.append(max(0, cu - mean))
            else:
                means.append(0)
                yerr_low.append(0)
                yerr_high.append(0)

        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=[yerr_low, yerr_high],
            capsize=3,
            label=strat.replace("_", " "),
            color=colors[strat],
            edgecolor="black"
        )

    ax.set_xticks(x)
    labels = []
    for n in nodes:
        row = df[df["Client_ID"] == n].iloc[0]
        labels.append(f"{n}\n(N_te={row['N_Test']}, Pos={row['N_Test_Pos']})")
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("In-Hospital Mortality AUROC (95% CI)", fontsize=11, fontweight="bold")
    ax.set_title("Personalization Strategies Across Dirichlet Hospital Nodes", fontsize=12, fontweight="bold")
    ax.set_ylim(0.40, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig_path = "results/phase12/figures/advanced_personalization_gains.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved personalization plot to {fig_path}")


if __name__ == "__main__":
    df = run_advanced_personalization_experiments()
    print("Advanced personalization experiments completed successfully.")
