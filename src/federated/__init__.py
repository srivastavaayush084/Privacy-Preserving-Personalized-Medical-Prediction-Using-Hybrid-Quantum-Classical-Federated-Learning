"""
Federated learning package for privacy-preserving medical prediction.
"""
from src.federated.federated_config import FederatedConfig
from src.federated.client import HospitalClient
from src.federated.server import FederatedCoordinator
from src.federated.strategies import FedAvgStrategy, FedProxStrategy

__all__ = [
    "FederatedConfig",
    "HospitalClient",
    "FederatedCoordinator",
    "FedAvgStrategy",
    "FedProxStrategy"
]
