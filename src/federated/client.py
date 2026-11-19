"""
Hospital Federated Client Module.
Simulates an independent hospital site or consortium holding private clinical records.
Raw patient records never leave the local client instance.
"""
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, recall_score

from src.federated.communication import ClientRegistrationMessage, ClientUpdateMessage

logger = logging.getLogger(__name__)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


class HospitalClient:
    """Represents a single hospital node participating in federated learning."""

    def __init__(
        self,
        client_id: str,
        hospital_name: str,
        hospital_ids: List[int],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_factory: Callable[[], nn.Module],
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.hospital_name = hospital_name
        self.hospital_ids = hospital_ids

        # Local private data partitions
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self.device = torch.device(device)
        self.model = model_factory().to(self.device)

        # Local class weight
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        self.pos_weight_val = float(n_neg / max(n_pos, 1))

        logger.info(
            f"Initialized Client {client_id} ({hospital_name}): "
            f"Train={len(X_train)} (pos={n_pos}), Val={len(X_val)}, Test={len(X_test)}"
        )

    def register(self) -> ClientRegistrationMessage:
        """Returns registration metadata for coordinator."""
        mortality_rate = float(np.mean(self.y_train)) if len(self.y_train) > 0 else 0.0
        return ClientRegistrationMessage(
            client_id=self.client_id,
            hospital_name=self.hospital_name,
            hospital_ids=self.hospital_ids,
            num_train_samples=len(self.X_train),
            num_val_samples=len(self.X_val),
            num_test_samples=len(self.X_test),
            mortality_rate=mortality_rate
        )

    def set_parameters(self, state_dict: Dict[str, torch.Tensor]):
        """Loads parameters received from coordinator into local model."""
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns current local model parameters."""
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def train(
        self,
        epochs: int = 3,
        lr: float = 0.005,
        batch_size: int = 32,
        strategy: str = "FedAvg",
        mu: float = 0.01,
        global_state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """Executes local gradient steps on local private records."""
        self.model.train()
        
        # Loss function with local class weighting
        pos_weight = torch.tensor([self.pos_weight_val], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Pre-convert tensors
        X_t = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(self.y_train, dtype=torch.float32).to(self.device)
        n_samples = len(self.X_train)

        # Prepare global state for FedProx proximal regularization
        global_tensors = {}
        if strategy == "FedProx" and global_state_dict is not None:
            for name, param in self.model.named_parameters():
                if name in global_state_dict:
                    global_tensors[name] = global_state_dict[name].to(self.device)

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx: start_idx + batch_size]
                bx = X_t[batch_idx]
                by = y_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)

                # FedProx proximal penalty: (mu / 2) * ||theta - theta_global||^2
                if strategy == "FedProx" and global_tensors:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_tensors:
                            prox_loss += torch.sum((param - global_tensors[name]) ** 2)
                    loss += (mu / 2.0) * prox_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

        avg_loss = total_loss / max(steps, 1)
        return float(avg_loss)

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """Evaluates local model on local validation or test split."""
        self.model.eval()
        if split == "val":
            X = self.X_val
            y = self.y_val
        elif split == "test":
            X = self.X_test
            y = self.y_test
        else:
            X = self.X_train
            y = self.y_train

        if len(y) == 0 or len(np.unique(y)) < 2:
            # Fallback for empty or single-class client slice
            return {
                "auroc": 0.5,
                "prauc": float(np.mean(y)) if len(y) > 0 else 0.0,
                "accuracy": 0.5,
                "brier": 0.25,
                "ece": 0.0,
                "sensitivity": 0.0,
                "specificity": 1.0
            }

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        try:
            auroc = roc_auc_score(y, probs)
            prauc = average_precision_score(y, probs)
        except Exception:
            auroc = 0.5
            prauc = float(np.mean(y))

        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        sens = recall_score(y, preds, zero_division=0)
        
        # Specificity
        neg_mask = (y == 0)
        spec = float(np.mean(preds[neg_mask] == 0)) if np.any(neg_mask) else 1.0

        brier = brier_score_loss(y, probs)
        ece = compute_ece(probs, y)

        return {
            "auroc": float(auroc),
            "prauc": float(prauc),
            "accuracy": float(acc),
            "brier": float(brier),
            "ece": float(ece),
            "sensitivity": float(sens),
            "specificity": float(spec)
        }

    def generate_update(
        self,
        round_idx: int,
        train_loss: float
    ) -> ClientUpdateMessage:
        """Packages trained weights and local evaluation metrics for coordinator."""
        val_metrics = self.evaluate(split="val")
        test_metrics = self.evaluate(split="test")

        return ClientUpdateMessage(
            client_id=self.client_id,
            round_idx=round_idx,
            num_samples=len(self.X_train),
            model_state_dict=self.get_parameters(),
            train_loss=train_loss,
            local_val_metrics=val_metrics,
            local_test_metrics=test_metrics
        )
