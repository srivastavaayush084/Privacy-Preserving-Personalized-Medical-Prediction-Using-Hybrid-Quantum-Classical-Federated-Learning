"""Hybrid Quantum-Classical Neural Network (HQNN) module.
Connects classical feature encoder, PennyLane PQC layer, and classical prediction head.
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.quantum.quantum_circuits import create_pqc_qnode

logger = logging.getLogger(__name__)


class HybridQNNNet(nn.Module):
    """End-to-end PyTorch hybrid quantum-classical neural network architecture."""

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

        # 1. Classical Feature Bottleneck Encoder
        # Compresses 41 clinical inputs down to n_qubits angles normalized in [-pi, pi]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, num_qubits),
            nn.Tanh()  # Tanh bounds values in [-1, 1], multiplied by pi for angle embedding
        )

        # 2. Quantum Layer (PennyLane PQC via TorchLayer)
        circuit, weight_shapes, _ = create_pqc_qnode(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            entanglement=entanglement,
            device_name=device_name
        )
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # 3. Classical Prediction Head
        # Concatenates quantum expectation values (n_qubits) with classical bottleneck features (n_qubits)
        self.head = nn.Sequential(
            nn.Linear(num_qubits * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encoding
        encoded_feats = self.encoder(x)  # shape (B, num_qubits)
        angle_inputs = encoded_feats * np.pi

        # Quantum PQC execution
        q_out = self.quantum_layer(angle_inputs)  # shape (B, num_qubits)

        # Joint prediction head
        combined = torch.cat([encoded_feats, q_out], dim=-1)
        logits = self.head(combined).squeeze(-1)
        return logits


class HybridQNNModel:
    """Wrapper for training, evaluation, checkpointing, and prediction of Hybrid QNN."""

    def __init__(
        self,
        input_dim: int = 41,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        entanglement: str = "strongly_entangling",
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.entanglement = entanglement
        self.input_dim = input_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = HybridQNNNet(
            input_dim=input_dim,
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            entanglement=entanglement
        ).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n_samples = len(X_train)

        start_time = time.time()
        self.model.train()
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx: start_idx + self.batch_size]
                batch_x = X_t[batch_idx]
                batch_y = y_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            if (epoch + 1) % 5 == 0:
                logger.info(f"Hybrid QNN Epoch {epoch+1}/{self.epochs} - Loss: {total_loss / n_samples:.4f}")

        self.training_time_seconds = time.time() - start_time
        logger.info(f"Hybrid QNN training completed in {self.training_time_seconds:.2f} seconds.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            # Predict in batches to maintain predictable memory
            all_probs = []
            for start_idx in range(0, len(X), self.batch_size):
                batch_x = X_t[start_idx: start_idx + self.batch_size]
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
            probs = np.concatenate(all_probs)
        self.inference_time_seconds = time.time() - start_time
        return probs

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self
