"""
Configuration dataclasses for federated learning experiments.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FederatedConfig:
    """Federated learning execution configuration."""
    num_rounds: int = 10
    num_clients: int = 5
    client_selection_ratio: float = 1.0
    local_epochs: int = 3
    local_lr: float = 0.005
    local_batch_size: int = 32
    strategy: str = "FedAvg"  # 'FedAvg', 'FedProx'
    mu: float = 0.01          # Proximal term coefficient for FedProx
    dropout_rate: float = 0.0 # Client dropout rate per round
    model_type: str = "classical"  # 'classical' (MLP) or 'quantum' (HybridQNNNet)
    num_qubits: int = 4
    circuit_depth: int = 2
    entanglement: str = "strongly_entangling"
    seed: int = 42
    output_dir: str = "results/metrics"
    checkpoint_dir: str = "checkpoints/federated"
