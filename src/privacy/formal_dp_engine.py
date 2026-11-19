"""
Formal Differential Privacy Engine for Classical and Hybrid Quantum Models.
Implements mathematically rigorous per-sample DP-SGD with:
1. Per-sample gradient computation and L2 clipping (norm <= C)
2. Calibrated Gaussian noise addition (N(0, sigma^2 C^2 I))
3. Formal Rényi Differential Privacy (RDP) accounting via Opacus RDPAccountant
4. Native Opacus PrivacyEngine integration for Classical ClinicalDPMLPNet
5. Exact micro-batched per-sample gradient clipping engine for HybridQNNNet
"""

import copy
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from src.models.quantum.quantum_circuits import create_pqc_qnode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DP-COMPATIBLE MODEL ARCHITECTURES (REPLACING BATCHNORM WITH GROUPNORM)
# ---------------------------------------------------------------------------

class ClinicalDPMLPNet(nn.Module):
    """Deep tabular MLP compatible with per-sample DP-SGD (GroupNorm instead of BatchNorm)."""

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.20
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            # GroupNorm is strictly per-sample and does not mix batch statistics
            layers.append(nn.GroupNorm(num_groups=2, num_channels=h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class DPHybridQNNNet(nn.Module):
    """Hybrid Quantum-Classical Neural Network compatible with per-sample DP-SGD."""

    def __init__(
        self,
        input_dim: int = 41,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        entanglement: str = "strongly_entangling",
        device_name: str = "default.qubit"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        # Classical encoder with GroupNorm (no cross-sample leakage)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.GroupNorm(num_groups=2, num_channels=16),
            nn.ReLU(),
            nn.Linear(16, num_qubits),
            nn.Tanh()
        )

        # PennyLane Parameterized Quantum Circuit (PQC)
        circuit, weight_shapes, _ = create_pqc_qnode(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            entanglement=entanglement,
            device_name=device_name
        )
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical classification head
        self.head = nn.Sequential(
            nn.Linear(num_qubits * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        angles = enc * np.pi
        q_out = self.quantum_layer(angles)
        comb = torch.cat([enc, q_out], dim=-1)
        logits = self.head(comb).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# 2. PRIVACY CALIBRATION AND ACCOUNTING
# ---------------------------------------------------------------------------

def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    total_steps: int
) -> float:
    """Computes exact noise multiplier sigma for target (epsilon, delta) via Opacus."""
    if math.isinf(target_epsilon) or target_epsilon >= 100.0:
        return 0.0

    try:
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=total_steps
        )
        return float(sigma)
    except Exception:
        # Fallback binary search using step
        low = 0.01
        high = 50.0
        for _ in range(40):
            mid = (low + high) / 2.0
            acc = RDPAccountant()
            for _ in range(total_steps):
                acc.step(noise_multiplier=mid, sample_rate=sample_rate)
            cur_eps = acc.get_epsilon(delta=target_delta)
            if cur_eps > target_epsilon:
                low = mid
            else:
                high = mid
            if abs(high - low) < 1e-3:
                break
        return float(high)


# ---------------------------------------------------------------------------
# 3. CLASSICAL OPACUS DP-SGD TRAINER
# ---------------------------------------------------------------------------

class FormalOpacusMLPTrainer:
    """Formal DP-SGD training for Classical MLP using Opacus PrivacyEngine."""

    def __init__(
        self,
        input_dim: int = 41,
        hidden_dims: List[int] = [64, 32, 16],
        lr: float = 0.005,
        max_grad_norm: float = 1.0,
        epochs: int = 20,
        batch_size: int = 64,
        target_delta: float = 1e-5,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_delta = target_delta
        self.device = torch.device(device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_epsilon: float,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[nn.Module, float, float]:
        """Trains ClinicalDPMLPNet with formal DP-SGD guarantees.
        
        Returns:
            Tuple of (trained_model, achieved_epsilon, achieved_delta)
        """
        model = ClinicalDPMLPNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        pos_weight = torch.tensor([float((y_train == 0).sum() / max(1, (y_train == 1).sum()))]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if math.isinf(target_epsilon) or target_epsilon >= 100.0:
            # Non-private baseline
            for epoch in range(self.epochs):
                model.train()
                for bx, by in data_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    # Standard gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                    optimizer.step()
            return model, float("inf"), 0.0

        privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=False)
        model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=self.target_delta,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm
        )

        # Exact expected batch size under Poisson subsampling: E[|B|] = q * N
        # Aligns with Option A mathematical expectation (replaces Opacus integer truncation)
        exact_expected_batch_size = float(len(data_loader.dataset) * data_loader.sample_rate)
        optimizer.expected_batch_size = exact_expected_batch_size

        total_batches = 0
        for epoch in range(self.epochs):
            model.train()
            for bx, by in data_loader:
                total_batches += 1
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        achieved_eps = privacy_engine.get_epsilon(delta=self.target_delta)
        # Unwrap module from GradSampleModule if needed
        unwrapped_model = getattr(model, "_module", model)
        
        self.last_run_stats = {
            "total_sampler_iterations": total_batches,
            "optimizer_updates": total_batches,
            "accountant_steps": total_batches,
            "expected_steps": self.epochs * len(data_loader),
            "achieved_epsilon": float(achieved_eps),
            "achieved_delta": self.target_delta,
            "expected_batch_size": getattr(optimizer, "expected_batch_size", exact_expected_batch_size),
            "nominal_batch_size": self.batch_size,
            "sample_rate": getattr(data_loader, "sample_rate", (self.batch_size / len(X_train))),
            "clipping_bound": self.max_grad_norm,
        }
        return unwrapped_model, float(achieved_eps), self.target_delta


# ---------------------------------------------------------------------------
# 4. EXACT PER-SAMPLE DP-SGD ENGINE FOR HYBRID QNN
# ---------------------------------------------------------------------------

class FormalPerSampleDPQNNTrainer:
    """Exact per-sample DP-SGD training for Hybrid QNN.
    
    Computes genuine per-sample gradients g_i = nabla ell(f(x_i), y_i),
    clips each sample gradient norm ||g_i||_2 <= C,
    accumulates sum_i clip(g_i),
    injects calibrated Gaussian noise N(0, sigma^2 C^2 I),
    and formally bounds (eps, delta) via Opacus RDPAccountant.
    """

    def __init__(
        self,
        input_dim: int = 41,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        lr: float = 0.01,
        max_grad_norm: float = 1.0,
        epochs: int = 15,
        batch_size: int = 32,
        target_delta: float = 1e-5,
        device: str = "cpu"
    ):
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_delta = target_delta
        self.device = torch.device(device)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_epsilon: float,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[nn.Module, float, float]:
        """Trains DPHybridQNNNet with exact per-sample DP-SGD."""
        model = DPHybridQNNNet(
            input_dim=self.input_dim,
            num_qubits=self.num_qubits,
            circuit_depth=self.circuit_depth
        ).to(self.device)

        pos_weight = torch.tensor([float((y_train == 0).sum() / max(1, (y_train == 1).sum()))]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        N = len(X_train)
        steps_per_epoch = int(math.ceil(N / self.batch_size))
        total_steps = steps_per_epoch * self.epochs
        sample_rate = 1.0 / steps_per_epoch
        # Exact expected batch size under Poisson subsampling with inclusion rate q: E[|B|] = q * N
        # Aligns with Option A mathematical expectation (replaces nominal batch_size divisor)
        expected_batch_size = float(N * sample_rate)

        if math.isinf(target_epsilon) or target_epsilon >= 100.0:
            sigma = 0.0
            acc = None
        else:
            sigma = calibrate_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=self.target_delta,
                sample_rate=sample_rate,
                total_steps=total_steps
            )
            acc = RDPAccountant()

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        # Safe collate function to handle empty Poisson draws without IndexError
        def safe_collate(batch):
            if len(batch) == 0:
                return torch.empty((0, self.input_dim), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
            return torch.utils.data.dataloader.default_collate(batch)

        # Use UniformWithReplacementSampler for genuine Poisson subsampling matching Opacus RDPAccountant
        sampler = UniformWithReplacementSampler(num_samples=N, sample_rate=sample_rate)
        data_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=safe_collate)

        params = [p for p in model.parameters() if p.requires_grad]

        total_sampler_iterations = 0
        non_empty_batches = 0
        empty_batches = 0
        optimizer_updates = 0
        accountant_steps = 0

        for epoch in range(self.epochs):
            model.train()
            for bx, by in data_loader:
                total_sampler_iterations += 1
                B = len(bx)
                optimizer.zero_grad()
                accum_grads = [torch.zeros_like(p) for p in params]

                if B > 0:
                    non_empty_batches += 1
                    bx, by = bx.to(self.device), by.to(self.device)
                    # Exact per-sample gradient computation and L2 clipping (record-level sensitivity C)
                    for i in range(B):
                        xi = bx[i:i+1]
                        yi = by[i:i+1]
                        out_i = model(xi)
                        loss_i = criterion(out_i, yi)
                        sample_grads = torch.autograd.grad(loss_i, params, retain_graph=False)

                        # Compute individual sample L2 gradient norm
                        sample_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in sample_grads)).item()
                        clip_factor = min(1.0, self.max_grad_norm / (sample_norm + 1e-8))

                        # Accumulate clipped gradients
                        for acc_g, g in zip(accum_grads, sample_grads):
                            acc_g.add_(g * clip_factor)
                else:
                    empty_batches += 1

                # If private training, inject noise and normalize by fixed data-independent expected batch size
                if sigma > 0.0:
                    for acc_g, p in zip(accum_grads, params):
                        # Add Gaussian noise calibrated to sensitivity Delta_2 = C
                        noise = torch.randn_like(acc_g) * (sigma * self.max_grad_norm)
                        # Fixed expected batch size normalization (preserves exact RDP post-processing)
                        p.grad = (acc_g + noise) / expected_batch_size
                    if acc is not None:
                        acc.step(noise_multiplier=sigma, sample_rate=sample_rate)
                        accountant_steps += 1
                    optimizer.step()
                    optimizer_updates += 1
                else:
                    # Non-private baseline
                    if B > 0:
                        for acc_g, p in zip(accum_grads, params):
                            p.grad = acc_g / float(B)
                        optimizer.step()
                        optimizer_updates += 1

        if acc is not None:
            achieved_eps = float(acc.get_epsilon(delta=self.target_delta))
        else:
            achieved_eps = float("inf")

        self.last_run_stats = {
            "total_sampler_iterations": total_sampler_iterations,
            "non_empty_batches": non_empty_batches,
            "empty_batches": empty_batches,
            "optimizer_updates": optimizer_updates,
            "accountant_steps": accountant_steps,
            "expected_steps": total_steps,
            "achieved_epsilon": achieved_eps,
            "achieved_delta": self.target_delta,
            "expected_batch_size": expected_batch_size,
            "nominal_batch_size": self.batch_size,
            "sample_rate": sample_rate,
            "sigma": sigma,
            "clipping_bound": self.max_grad_norm,
        }

        return model, achieved_eps, self.target_delta


# ---------------------------------------------------------------------------
# 5. EVALUATION AND EMPIRICAL PRIVACY LEAKAGE (MIA RISK)
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu"
) -> Dict[str, float]:
    """Computes comprehensive clinical prediction metrics."""
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(dev)
        logits = model(x_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    preds = (probs >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_test, probs))
    prauc = float(average_precision_score(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))
    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    return {
        "AUROC": auroc,
        "PRAUC": prauc,
        "Brier": brier,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }


def compute_empirical_privacy_leakage(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_size: int = 300,
    device: str = "cpu"
) -> Dict[str, float]:
    """Calculates empirical Membership Inference Attack (MIA) risk.
    
    Uses negative binary cross-entropy (confidence/loss difference) as
    membership score. If the model memorized training samples, members
    have significantly lower loss than non-members.
    MIA AUC = 0.50 denotes perfect privacy (random guessing).
    MIA AUC >= 0.70 denotes significant privacy leakage.
    """
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    n_sample = min(sample_size, len(X_train), len(X_test))
    idx_tr = np.random.choice(len(X_train), n_sample, replace=False)
    idx_te = np.random.choice(len(X_test), n_sample, replace=False)

    with torch.no_grad():
        x_tr = torch.tensor(X_train[idx_tr], dtype=torch.float32).to(dev)
        y_tr = y_train[idx_tr]
        l_tr = model(x_tr).cpu().numpy()
        p_tr = 1.0 / (1.0 + np.exp(-l_tr))
        p_tr = np.clip(p_tr, 1e-6, 1.0 - 1e-6)
        # Member loss (negative cross entropy as membership score)
        loss_tr = -(y_tr * np.log(p_tr) + (1 - y_tr) * np.log(1 - p_tr))
        score_member = -loss_tr  # Lower loss = higher membership likelihood

        x_te = torch.tensor(X_test[idx_te], dtype=torch.float32).to(dev)
        y_te = y_test[idx_te]
        l_te = model(x_te).cpu().numpy()
        p_te = 1.0 / (1.0 + np.exp(-l_te))
        p_te = np.clip(p_te, 1e-6, 1.0 - 1e-6)
        loss_te = -(y_te * np.log(p_te) + (1 - y_te) * np.log(1 - p_te))
        score_non_member = -loss_te

    labels = np.concatenate([np.ones(n_sample), np.zeros(n_sample)])
    scores = np.concatenate([score_member, score_non_member])

    mia_auc = float(roc_auc_score(labels, scores))
    # Threshold at median score
    median_thresh = float(np.median(scores))
    mia_preds = (scores >= median_thresh).astype(int)
    mia_adv = float(abs(np.mean(mia_preds[labels == 1]) - np.mean(mia_preds[labels == 0])))

    return {
        "MIA_AUC": mia_auc,
        "MIA_Advantage": mia_adv
    }
