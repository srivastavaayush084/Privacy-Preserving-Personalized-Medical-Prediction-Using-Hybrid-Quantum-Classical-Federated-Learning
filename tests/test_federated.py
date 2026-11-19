"""
Unit tests for Federated Learning components (Client, Server, Aggregation, Strategies).
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.classical.baselines import ClinicalMLPNet
from src.federated.client import HospitalClient
from src.federated.communication import ClientUpdateMessage
from src.federated.aggregation import fedavg_aggregate, trimmed_mean_aggregate
from src.federated.federated_config import FederatedConfig
from src.federated.server import FederatedCoordinator
from src.federated.strategies import FedAvgStrategy, FedProxStrategy


def test_hospital_client_train_and_evaluate():
    """Verify local hospital client training and metric evaluation."""
    n_samples = 40
    n_features = 41
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    client = HospitalClient(
        client_id="Client-Test",
        hospital_name="Test Hospital",
        hospital_ids=[101],
        X_train=X[:20],
        y_train=y[:20],
        X_val=X[20:30],
        y_val=y[20:30],
        X_test=X[30:],
        y_test=y[30:],
        model_factory=lambda: ClinicalMLPNet(input_dim=n_features, hidden_dims=[16, 8]),
        device="cpu"
    )

    loss = client.train(epochs=1, lr=0.01, batch_size=10, strategy="FedAvg")
    assert isinstance(loss, float)
    assert loss > 0.0

    metrics = client.evaluate(split="test")
    assert "auroc" in metrics
    assert "prauc" in metrics
    assert "brier" in metrics
    assert "ece" in metrics


def test_fedavg_aggregation():
    """Verify FedAvg parameter mathematical weighting."""
    state1 = {"w": torch.tensor([2.0, 4.0])}
    state2 = {"w": torch.tensor([6.0, 8.0])}

    u1 = ClientUpdateMessage(
        client_id="c1", round_idx=1, num_samples=100,
        model_state_dict=state1, train_loss=0.5,
        local_val_metrics={}, local_test_metrics={"auroc": 0.8}
    )
    u2 = ClientUpdateMessage(
        client_id="c2", round_idx=1, num_samples=300,
        model_state_dict=state2, train_loss=0.4,
        local_val_metrics={}, local_test_metrics={"auroc": 0.85}
    )

    agg = fedavg_aggregate([u1, u2])
    # Expected: (100*2 + 300*6)/400 = (200 + 1800)/400 = 5.0
    # Expected: (100*4 + 300*8)/400 = (400 + 2400)/400 = 7.0
    expected = torch.tensor([5.0, 7.0])
    assert torch.allclose(agg["w"], expected)


def test_trimmed_mean_aggregation():
    """Verify robust trimmed mean rejects extreme outliers."""
    # 3 clients: c1=10.0, c2=10.5, c3=1000.0 (adversarial/corrupt)
    u1 = ClientUpdateMessage("c1", 1, 10, {"w": torch.tensor([10.0])}, 0.1, {}, {})
    u2 = ClientUpdateMessage("c2", 1, 10, {"w": torch.tensor([10.5])}, 0.1, {}, {})
    u3 = ClientUpdateMessage("c3", 1, 10, {"w": torch.tensor([1000.0])}, 0.1, {}, {})

    # Trim fraction 0.33 trims 1 from each end
    agg = trimmed_mean_aggregate([u1, u2, u3], trim_fraction=0.33)
    assert torch.allclose(agg["w"], torch.tensor([10.5]))


def test_federated_coordinator_single_round():
    """Verify coordinator coordinates a full communication round."""
    n_features = 41
    config = FederatedConfig(num_rounds=1, num_clients=2, local_epochs=1, local_batch_size=8)
    
    X_dummy = np.random.randn(20, n_features)
    y_dummy = np.random.choice([0, 1], size=20, p=[0.8, 0.2])

    c1 = HospitalClient(
        "C1", "Hospital 1", [1],
        X_train=X_dummy[:10], y_train=y_dummy[:10],
        X_val=X_dummy[:5], y_val=y_dummy[:5],
        X_test=X_dummy[10:], y_test=y_dummy[10:],
        model_factory=lambda: ClinicalMLPNet(input_dim=n_features, hidden_dims=[8, 4])
    )
    c2 = HospitalClient(
        "C2", "Hospital 2", [2],
        X_train=X_dummy[:10], y_train=y_dummy[:10],
        X_val=X_dummy[:5], y_val=y_dummy[:5],
        X_test=X_dummy[10:], y_test=y_dummy[10:],
        model_factory=lambda: ClinicalMLPNet(input_dim=n_features, hidden_dims=[8, 4])
    )

    coordinator = FederatedCoordinator(
        config=config,
        model_factory=lambda: ClinicalMLPNet(input_dim=n_features, hidden_dims=[8, 4]),
        global_test_data=(X_dummy, y_dummy)
    )
    coordinator.register_client(c1)
    coordinator.register_client(c2)

    record = coordinator.run_round(1)
    assert record["round"] == 1
    assert "global_auroc" in record
    assert "train_loss" in record
