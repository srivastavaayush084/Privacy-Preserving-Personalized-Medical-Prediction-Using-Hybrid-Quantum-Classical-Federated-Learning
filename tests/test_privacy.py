"""
Unit tests for Privacy, Differential Privacy, Secure Aggregation, and Privacy Attacks.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.classical.baselines import ClinicalMLPNet
from src.privacy.privacy_accountant import RDPAccountant
from src.privacy.differential_privacy import DPGradientEngine
from src.privacy.secure_aggregation import SecureAggregationProtocol
from src.privacy.privacy_attacks import MembershipInferenceAttack, GradientInversionAttack


def test_rdp_accountant():
    """Verify RDP privacy loss accumulation and epsilon calculation."""
    accountant = RDPAccountant()
    # Step with noise multiplier 1.0, sampling rate 0.05
    accountant.step(noise_multiplier=1.0, sample_rate=0.05, steps=10)
    eps = accountant.get_epsilon(target_delta=1e-5)
    assert eps > 0.0
    assert eps < 100.0


def test_dp_gradient_engine():
    """Verify gradient clipping and noise injection."""
    model = ClinicalMLPNet(input_dim=10, hidden_dims=[8, 4])
    engine = DPGradientEngine(clip_norm=0.5, noise_multiplier=0.1, target_delta=1e-5)

    dummy_x = torch.randn(4, 10)
    dummy_y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    criterion = nn.BCEWithLogitsLoss()

    out = model(dummy_x)
    loss = criterion(out, dummy_y)
    loss.backward()

    eps = engine.clip_and_perturb_gradients(model, batch_size=4, dataset_size=20)
    assert eps > 0.0

    # Ensure gradients are bounded
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += float(p.grad.data.norm(2).item()) ** 2
    total_norm = total_norm ** 0.5
    # Clipped norm + noise should remain bounded
    assert total_norm < 10.0


def test_secure_aggregation_zero_sum():
    """Verify pairwise masks sum to zero and aggregate perfectly."""
    client_ids = ["Client_A", "Client_B", "Client_C", "Client_D"]
    param_shapes = {"layer1": (4, 4), "bias": (4,)}

    secagg = SecureAggregationProtocol(seed=42)
    masks = secagg.generate_pairwise_masks(client_ids, param_shapes)

    # Sum of all masks must be 0
    total_mask = torch.zeros(param_shapes["layer1"])
    for cid in client_ids:
        total_mask += masks[cid]["layer1"]

    assert torch.allclose(total_mask, torch.zeros_like(total_mask), atol=1e-5)

    # Test masked updates aggregation
    raw_updates = [
        {"layer1": torch.ones(param_shapes["layer1"]) * i}
        for i in range(1, 5)
    ]
    # Raw unmasked average
    expected_avg = sum(u["layer1"] for u in raw_updates) / 4.0

    # Mask updates
    masked_updates = [
        secagg.mask_client_update(raw_updates[idx], masks[client_ids[idx]])
        for idx in range(4)
    ]
    # Aggregate masked updates
    aggregated = secagg.aggregate_masked_updates(masked_updates)

    assert torch.allclose(aggregated["layer1"], expected_avg, atol=1e-5)


def test_membership_inference_attack():
    """Verify MIA feature extraction and shadow attack."""
    model = ClinicalMLPNet(input_dim=10, hidden_dims=[8, 4])
    mia = MembershipInferenceAttack(random_state=42)

    X_member = np.random.randn(20, 10)
    y_member = np.random.choice([0, 1], size=20)
    X_nonmember = np.random.randn(20, 10)
    y_nonmember = np.random.choice([0, 1], size=20)

    metrics = mia.evaluate(model, X_member, y_member, X_nonmember, y_nonmember)
    assert "mia_auroc" in metrics
    assert "mia_accuracy" in metrics
    assert "mia_advantage" in metrics
    assert 0.0 <= metrics["mia_auroc"] <= 1.0


def test_gradient_inversion_attack_runs():
    """Verify gradient inversion optimization executes."""
    model = ClinicalMLPNet(input_dim=6, hidden_dims=[4])
    attacker = GradientInversionAttack(max_iter=5, lr=0.1)

    true_x = torch.randn(6)
    true_y = torch.tensor(1.0)
    criterion = nn.BCEWithLogitsLoss()

    reconstructed, metrics = attacker.attack(model, true_x, true_y, criterion)
    assert reconstructed.shape == true_x.shape
    assert "reconstruction_mse" in metrics
    assert "cosine_similarity" in metrics
