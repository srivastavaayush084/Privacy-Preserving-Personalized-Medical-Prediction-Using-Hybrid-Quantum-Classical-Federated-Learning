"""
Hospital-specific Adapter Architectures for Federated Personalization.
Enforces strict parameter partitioning:
- Shared Global Representation Parameters (Theta_s): transmitted and aggregated across hospitals.
- Local Hospital-Specific Parameters (Theta_k): private to each hospital site, never shared.
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml

from src.models.quantum.quantum_circuits import create_pqc_qnode

logger = logging.getLogger(__name__)


class HospitalAdapterModule(nn.Module):
    """Lightweight residual adapter for hospital-specific feature conditioning."""

    def __init__(self, in_features: int, bottleneck_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        bottleneck_dim = max(4, in_features // bottleneck_ratio)
        self.adapter = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, in_features)
        )
        self.scaling = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual shortcut preserves shared representations while adapting
        return x + self.scaling * self.adapter(x)


class PersonalizedHybridQNNNet(nn.Module):
    """Hybrid Quantum-Classical Neural Network with Hospital-Specific Adapters.
    
    Shared Parameters (Theta_s):
        - encoder: 41 -> 16 -> num_qubits (Classical dimensionality compression)
        - quantum_layer: PennyLane Parameterized Quantum Circuit (PQC)
        
    Local Parameters (Theta_k):
        - local_adapter: Residual adapter mapping (2 * num_qubits)
        - local_head: Hospital-specific prediction head
    """

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
        self.entanglement = entanglement

        # 1. SHARED: Classical Feature Encoder (Theta_s)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_qubits),
            nn.Tanh()
        )

        # 2. SHARED: Quantum PQC Layer (Theta_s)
        circuit, weight_shapes, _ = create_pqc_qnode(
            num_qubits=num_qubits,
            circuit_depth=circuit_depth,
            entanglement=entanglement,
            device_name=device_name
        )
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Representation dimension = classical bottleneck + quantum expectation values
        rep_dim = num_qubits * 2

        # 3. LOCAL: Hospital-Specific Adapter (Theta_k)
        self.local_adapter = HospitalAdapterModule(in_features=rep_dim)

        # 4. LOCAL: Hospital-Specific Prediction Head (Theta_k)
        self.local_head = nn.Sequential(
            nn.Linear(rep_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def extract_shared_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes joint classical-quantum representation using shared parameters."""
        encoded_feats = self.encoder(x)
        angle_inputs = encoded_feats * np.pi
        q_out = self.quantum_layer(angle_inputs)
        combined = torch.cat([encoded_feats, q_out], dim=-1)
        return combined

    def forward(self, x: torch.Tensor, use_adapter: bool = True) -> torch.Tensor:
        features = self.extract_shared_features(x)
        if use_adapter:
            adapted_features = self.local_adapter(features)
        else:
            adapted_features = features
        logits = self.local_head(adapted_features).squeeze(-1)
        return logits

    def get_shared_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns only the shared parameters (Theta_s) for federated aggregation."""
        shared_state = {}
        for k, v in self.state_dict().items():
            if k.startswith("encoder.") or k.startswith("quantum_layer."):
                shared_state[k] = v.cpu().clone()
        return shared_state

    def load_shared_state_dict(self, shared_state: Dict[str, torch.Tensor]):
        """Loads aggregated shared parameters while strictly preserving local adapter weights."""
        current_state = self.state_dict()
        for k, v in shared_state.items():
            if k in current_state and (k.startswith("encoder.") or k.startswith("quantum_layer.")):
                current_state[k].copy_(v)

    def freeze_shared_parameters(self, freeze: bool = True):
        """Freezes shared parameters during local hospital personalization."""
        for p in self.encoder.parameters():
            p.requires_grad = not freeze
        for p in self.quantum_layer.parameters():
            p.requires_grad = not freeze


class PersonalizedMLPNet(nn.Module):
    """Classical MLP with parameter separation for hospital personalization."""

    def __init__(self, input_dim: int = 41, hidden_dims: list = [64, 32], dropout: float = 0.25):
        super().__init__()
        # 1. SHARED: Feature representation layers (Theta_s)
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        rep_dim = hidden_dims[1]

        # 2. LOCAL: Hospital-Specific Adapter (Theta_k)
        self.local_adapter = HospitalAdapterModule(in_features=rep_dim)

        # 3. LOCAL: Hospital-Specific Prediction Head (Theta_k)
        self.local_head = nn.Sequential(
            nn.Linear(rep_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor, use_adapter: bool = True) -> torch.Tensor:
        features = self.shared_encoder(x)
        if use_adapter:
            adapted = self.local_adapter(features)
        else:
            adapted = features
        logits = self.local_head(adapted).squeeze(-1)
        return logits

    def get_shared_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v.cpu().clone()
            for k, v in self.state_dict().items()
            if k.startswith("shared_encoder.")
        }

    def load_shared_state_dict(self, shared_state: Dict[str, torch.Tensor]):
        current_state = self.state_dict()
        for k, v in shared_state.items():
            if k in current_state and k.startswith("shared_encoder."):
                current_state[k].copy_(v)

    def freeze_shared_parameters(self, freeze: bool = True):
        for p in self.shared_encoder.parameters():
            p.requires_grad = not freeze
