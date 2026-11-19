"""
Federated Coordinator (Server) Module.
Coordinates rounds, client selection, parameter broadcast, update aggregation,
and global/per-client performance monitoring.
"""
import copy
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score, recall_score

from src.federated.client import HospitalClient, compute_ece
from src.federated.communication import CommunicationTracker, GlobalBroadcastMessage, ClientUpdateMessage
from src.federated.federated_config import FederatedConfig
from src.federated.strategies import BaseFederatedStrategy, FedAvgStrategy, FedProxStrategy

logger = logging.getLogger(__name__)


class FederatedCoordinator:
    """Central server orchestrating federated learning rounds."""

    def __init__(
        self,
        config: FederatedConfig,
        model_factory: Callable[[], nn.Module],
        global_test_data: Tuple[np.ndarray, np.ndarray],
        strategy: Optional[BaseFederatedStrategy] = None,
        device: str = "cpu"
    ):
        self.config = config
        self.device = torch.device(device)
        self.global_model = model_factory().to(self.device)
        self.X_global_test, self.y_global_test = global_test_data

        if strategy is None:
            if config.strategy == "FedProx":
                self.strategy = FedProxStrategy(mu=config.mu)
            else:
                self.strategy = FedAvgStrategy()
        else:
            self.strategy = strategy

        self.clients: Dict[str, HospitalClient] = {}
        self.comm_tracker = CommunicationTracker()
        self.history: List[Dict[str, Any]] = []

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)

    def register_client(self, client: HospitalClient):
        """Registers a participating hospital client."""
        reg_info = client.register()
        self.clients[client.client_id] = client
        logger.info(
            f"Coordinator registered Client: {reg_info.client_id} "
            f"({reg_info.hospital_name}) with {reg_info.num_train_samples} samples."
        )

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluates global model on centralized test holdout."""
        self.global_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(self.X_global_test, dtype=torch.float32).to(self.device)
            logits = self.global_model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        y = self.y_global_test
        try:
            auroc = roc_auc_score(y, probs)
            prauc = average_precision_score(y, probs)
        except Exception:
            auroc = 0.5
            prauc = float(np.mean(y))

        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        sens = recall_score(y, preds, zero_division=0)
        neg_mask = (y == 0)
        spec = float(np.mean(preds[neg_mask] == 0)) if np.any(neg_mask) else 1.0
        brier = brier_score_loss(y, probs)
        ece = compute_ece(probs, y)

        return {
            "auroc": float(auroc),
            "prauc": float(prauc),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "brier": float(brier),
            "ece": float(ece)
        }

    def run_round(self, round_idx: int) -> Dict[str, Any]:
        """Executes a single communication round across registered hospital clients."""
        round_start_time = time.time()
        client_ids = list(self.clients.keys())

        # 1. Client selection & dropout simulation
        selected_client_ids = self.strategy.select_clients(
            client_ids,
            fraction=self.config.client_selection_ratio,
            dropout_rate=self.config.dropout_rate
        )

        global_state = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}
        broadcast_msg = GlobalBroadcastMessage(
            round_idx=round_idx,
            global_parameters=global_state,
            strategy_config={"strategy": self.config.strategy, "mu": self.config.mu}
        )
        self.comm_tracker.record_downlink(round_idx, broadcast_msg, len(selected_client_ids))

        # 2. Local Training on private hospital data
        client_updates: List[ClientUpdateMessage] = []
        for cid in selected_client_ids:
            client = self.clients[cid]
            # Synchronize client model with current global parameters
            client.set_parameters(global_state)

            # Local training step
            loss = client.train(
                epochs=self.config.local_epochs,
                lr=self.config.local_lr,
                batch_size=self.config.local_batch_size,
                strategy=self.config.strategy,
                mu=self.config.mu,
                global_state_dict=global_state
            )

            # Generate privacy-preserving update (model weights only, zero raw patient data)
            update = client.generate_update(round_idx=round_idx, train_loss=loss)
            self.comm_tracker.record_uplink(round_idx, update)
            client_updates.append(update)

        # 3. Model Aggregation
        aggregated_weights = self.strategy.aggregate(client_updates, global_state)
        self.global_model.load_state_dict(aggregated_weights)

        # 4. Global Evaluation
        global_eval = self.evaluate_global_model()
        round_duration = time.time() - round_start_time

        # 5. Per-Client Evaluation Statistics
        client_test_aurocs = [u.local_test_metrics["auroc"] for u in client_updates]
        avg_client_auroc = float(np.mean(client_test_aurocs)) if client_test_aurocs else 0.5
        min_client_auroc = float(np.min(client_test_aurocs)) if client_test_aurocs else 0.5
        max_client_auroc = float(np.max(client_test_aurocs)) if client_test_aurocs else 0.5
        avg_train_loss = float(np.mean([u.train_loss for u in client_updates]))

        round_record = {
            "round": round_idx,
            "strategy": self.config.strategy,
            "model_type": self.config.model_type,
            "participating_clients": len(selected_client_ids),
            "train_loss": avg_train_loss,
            "global_auroc": global_eval["auroc"],
            "global_prauc": global_eval["prauc"],
            "global_accuracy": global_eval["accuracy"],
            "global_sensitivity": global_eval["sensitivity"],
            "global_specificity": global_eval["specificity"],
            "global_brier": global_eval["brier"],
            "global_ece": global_eval["ece"],
            "avg_client_auroc": avg_client_auroc,
            "worst_client_auroc": min_client_auroc,
            "best_client_auroc": max_client_auroc,
            "round_duration_sec": round_duration
        }

        self.history.append(round_record)
        logger.info(
            f"Round {round_idx:02d}/{self.config.num_rounds:02d} [{self.config.strategy}-{self.config.model_type}] - "
            f"Loss: {avg_train_loss:.4f} | Global AUROC: {global_eval['auroc']:.4f} | "
            f"PR-AUC: {global_eval['prauc']:.4f} | Worst-Client AUROC: {min_client_auroc:.4f} "
            f"({round_duration:.2f}s)"
        )
        return round_record

    def train_federated(self) -> pd.DataFrame:
        """Executes full multi-round federated training run."""
        best_auroc = 0.0
        best_checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"fed_{self.config.strategy.lower()}_{self.config.model_type}_best.pt"
        )

        logger.info(
            f"Starting Federated Learning: {self.config.num_rounds} rounds, "
            f"{len(self.clients)} clients, strategy={self.config.strategy}, model={self.config.model_type}"
        )

        for r in range(1, self.config.num_rounds + 1):
            record = self.run_round(r)
            if record["global_auroc"] > best_auroc:
                best_auroc = record["global_auroc"]
                torch.save(self.global_model.state_dict(), best_checkpoint_path)

        history_df = pd.DataFrame(self.history)
        logger.info(f"Federated training complete. Best Global AUROC: {best_auroc:.4f}")
        return history_df
