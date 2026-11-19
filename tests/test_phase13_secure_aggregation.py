"""
Phase 13 Secure Aggregation Independent Verification Unit Tests.
Verifies Shamir Secret Sharing polynomial mechanics, Lagrange interpolation over F_p,
dropout unmasking reconciliation, threshold breach aborts, and numerical precision.
"""
import pytest
import numpy as np
import torch

from src.privacy.secure_aggregation_v2 import (
    SecureAggregationV2,
    ShamirSecretSharingSimulator
)


def test_shamir_reconstruction_threshold_and_exactness():
    """Verify that Shamir secret sharing reconstructs exactly with >= t shares and fails with < t."""
    sss = ShamirSecretSharingSimulator(threshold=3, num_clients=5, prime=2147483647)
    rng = np.random.RandomState(42)
    secret = 123456789
    
    shares = sss.split_secret(secret, rng)
    assert len(shares) == 5
    
    # 3 shares: must reconstruct exact secret
    three_shares = {1: shares[1], 3: shares[3], 5: shares[5]}
    rec_secret = sss.reconstruct_secret(three_shares)
    assert rec_secret == secret, f"Reconstructed secret {rec_secret} != original {secret}"
    
    # 2 shares: must raise ValueError
    two_shares = {1: shares[1], 2: shares[2]}
    with pytest.raises(ValueError, match="Need at least 3 shares"):
        sss.reconstruct_secret(two_shares)


def test_secure_aggregation_dropout_resilience_and_exactness():
    """Verify that SecureAggregationV2 achieves exact unmasked mean under 0, 1, and 2 dropouts."""
    client_ids = [f"hospital_{i}" for i in range(1, 6)]
    sec_agg = SecureAggregationV2(client_ids=client_ids, threshold_ratio=0.60, random_seed=42)
    sec_agg.setup_keys_and_shares()
    
    torch.manual_seed(42)
    raw_updates = {
        cid: {"layer1": torch.randn(10, 5) * 0.5, "bias": torch.randn(10) * 0.1}
        for cid in client_ids
    }
    
    # 1. Zero dropouts
    masked_0 = sec_agg.mask_client_updates(raw_updates, dropped_client_ids=set())
    agg_0 = sec_agg.server_aggregate_with_dropout_resolution(
        masked_updates=masked_0,
        all_client_ids=client_ids,
        dropped_client_ids=set()
    )
    gt_0 = {
        "layer1": torch.stack([raw_updates[c]["layer1"] for c in client_ids]).mean(dim=0),
        "bias": torch.stack([raw_updates[c]["bias"] for c in client_ids]).mean(dim=0)
    }
    assert torch.allclose(agg_0["layer1"], gt_0["layer1"], atol=1e-5)
    
    # 2. One dropout (hospital_5)
    dropped_1 = {"hospital_5"}
    surviving_1 = [c for c in client_ids if c not in dropped_1]
    masked_1 = sec_agg.mask_client_updates(raw_updates, dropped_client_ids=dropped_1)
    agg_1 = sec_agg.server_aggregate_with_dropout_resolution(
        masked_updates=masked_1,
        all_client_ids=client_ids,
        dropped_client_ids=dropped_1
    )
    gt_1 = {
        "layer1": torch.stack([raw_updates[c]["layer1"] for c in surviving_1]).mean(dim=0),
        "bias": torch.stack([raw_updates[c]["bias"] for c in surviving_1]).mean(dim=0)
    }
    assert torch.allclose(agg_1["layer1"], gt_1["layer1"], atol=1e-5)
    
    # 3. Two dropouts (hospital_4, hospital_5) -> exactly at threshold 3
    dropped_2 = {"hospital_4", "hospital_5"}
    surviving_2 = [c for c in client_ids if c not in dropped_2]
    masked_2 = sec_agg.mask_client_updates(raw_updates, dropped_client_ids=dropped_2)
    agg_2 = sec_agg.server_aggregate_with_dropout_resolution(
        masked_updates=masked_2,
        all_client_ids=client_ids,
        dropped_client_ids=dropped_2
    )
    gt_2 = {
        "layer1": torch.stack([raw_updates[c]["layer1"] for c in surviving_2]).mean(dim=0),
        "bias": torch.stack([raw_updates[c]["bias"] for c in surviving_2]).mean(dim=0)
    }
    assert torch.allclose(agg_2["layer1"], gt_2["layer1"], atol=1e-5)


def test_secure_aggregation_threshold_breach_abort():
    """Verify that protocol safely aborts when surviving clients fall below threshold."""
    client_ids = [f"hospital_{i}" for i in range(1, 6)]
    sec_agg = SecureAggregationV2(client_ids=client_ids, threshold_ratio=0.60, random_seed=42)
    sec_agg.setup_keys_and_shares()
    
    raw_updates = {
        cid: {"layer1": torch.randn(5, 5)}
        for cid in client_ids
    }
    dropped_3 = {"hospital_3", "hospital_4", "hospital_5"}  # 3 drops -> 2 surviving < 3
    masked = sec_agg.mask_client_updates(raw_updates, dropped_client_ids=dropped_3)
    
    with pytest.raises(RuntimeError, match="below threshold"):
        sec_agg.server_aggregate_with_dropout_resolution(
            masked_updates=masked,
            all_client_ids=client_ids,
            dropped_client_ids=dropped_3
        )
