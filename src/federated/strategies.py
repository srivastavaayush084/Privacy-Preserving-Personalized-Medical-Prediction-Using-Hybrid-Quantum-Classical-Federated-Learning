"""
Federated optimization strategies (FedAvg, FedProx, Robust Trimmed FedAvg).
"""
import abc
import random
from typing import Dict, List, Optional
import torch

from src.federated.aggregation import fedavg_aggregate, trimmed_mean_aggregate
from src.federated.communication import ClientUpdateMessage


class BaseFederatedStrategy(abc.ABC):
    """Abstract base class for federated optimization strategies."""

    def __init__(self, name: str):
        self.name = name

    def select_clients(
        self,
        client_ids: List[str],
        fraction: float = 1.0,
        dropout_rate: float = 0.0
    ) -> List[str]:
        """Selects clients for the current communication round and simulates network dropout."""
        k = max(1, int(len(client_ids) * fraction))
        selected = random.sample(client_ids, k)

        # Simulate intermittent client dropout
        active_clients = [c for c in selected if random.random() >= dropout_rate]
        if not active_clients:
            # Enforce at least 1 client communicates
            active_clients = [random.choice(selected)]

        return active_clients

    @abc.abstractmethod
    def aggregate(
        self,
        client_updates: List[ClientUpdateMessage],
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregates local updates into a new global state dict."""
        pass


class FedAvgStrategy(BaseFederatedStrategy):
    """Federated Averaging (McMahan et al., 2017)."""

    def __init__(self):
        super().__init__(name="FedAvg")

    def aggregate(
        self,
        client_updates: List[ClientUpdateMessage],
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        return fedavg_aggregate(client_updates, global_model_state)


class FedProxStrategy(BaseFederatedStrategy):
    """Federated Proximal Optimization (Li et al., 2020) for non-IID data."""

    def __init__(self, mu: float = 0.01):
        super().__init__(name="FedProx")
        self.mu = mu

    def aggregate(
        self,
        client_updates: List[ClientUpdateMessage],
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        return fedavg_aggregate(client_updates, global_model_state)


class RobustFedAvgStrategy(BaseFederatedStrategy):
    """Trimmed Mean Federated Averaging for Byzantine and outlier tolerance."""

    def __init__(self, trim_fraction: float = 0.1):
        super().__init__(name="RobustFedAvg")
        self.trim_fraction = trim_fraction

    def aggregate(
        self,
        client_updates: List[ClientUpdateMessage],
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        return trimmed_mean_aggregate(client_updates, self.trim_fraction)
