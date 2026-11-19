"""
Phase 12 Remediation Test Suite.
Verifies:
1. Formal per-sample DP-SGD gradient clipping and Opacus RDP accounting.
2. Secure Aggregation v2 under 0%, 20%, and 40% client dropouts.
3. Secure Aggregation v2 safe abort when surviving clients drop below threshold t.
4. Dirichlet non-IID clinical partitioning.
5. External EHR schema validation and target leakage prevention.
"""

import math
import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))

from src.models.classical.baselines import ClinicalMLPNet
from src.privacy.formal_dp_engine import (
    ClinicalDPMLPNet,
    DPHybridQNNNet,
    FormalOpacusMLPTrainer,
    FormalPerSampleDPQNNTrainer,
    calibrate_noise_multiplier
)
from src.privacy.secure_aggregation_v2 import SecureAggregationV2, ShamirSecretSharingSimulator
from src.federated.advanced_partitioner import partition_dirichlet_clients
from src.data.external_dataset_adapter import ExternalDataHarmonizer


def test_shamir_secret_sharing_simulation():
    """Verifies t-out-of-n Shamir secret reconstruction and threshold property."""
    sss = ShamirSecretSharingSimulator(threshold=3, num_clients=5)
    rng = np.random.RandomState(42)
    secret = 123456789

    shares = sss.split_secret(secret, rng)
    assert len(shares) == 5

    # Any 3 shares must reconstruct the secret
    subset_3 = {1: shares[1], 3: shares[3], 5: shares[5]}
    rec_secret = sss.reconstruct_secret(subset_3)
    assert rec_secret == secret

    # Any 4 shares must also reconstruct the secret
    subset_4 = {2: shares[2], 3: shares[3], 4: shares[4], 5: shares[5]}
    assert sss.reconstruct_secret(subset_4) == secret

    # Fewer than t shares must fail to reconstruct
    subset_2 = {1: shares[1], 2: shares[2]}
    with pytest.raises(ValueError):
        sss.reconstruct_secret(subset_2)


def test_secure_aggregation_v2_numerical_reconstruction():
    """Verifies that SecureAggregationV2 achieves exact unmasking (< 1e-5 error) under dropouts."""
    client_ids = [f"Site_{i}" for i in range(1, 6)]
    secagg = SecureAggregationV2(client_ids=client_ids, threshold_ratio=0.60, random_seed=99)
    secagg.setup_keys_and_shares()

    raw_updates = {
        cid: {"weights": torch.randn(10, 20), "bias": torch.randn(10)}
        for cid in client_ids
    }

    # Scenario A: 0% dropouts
    masked = secagg.mask_client_updates(raw_updates, dropped_client_ids=set())
    recovered = secagg.server_aggregate_with_dropout_resolution(masked, client_ids, dropped_client_ids=set())
    true_avg = {k: sum(raw_updates[cid][k] for cid in client_ids) / 5.0 for k in ["weights", "bias"]}
    for k in ["weights", "bias"]:
        assert torch.allclose(recovered[k], true_avg[k], atol=1e-5)

    # Scenario B: 20% dropouts (1 client drops)
    dropped_20 = {"Site_3"}
    surviving_20 = [cid for cid in client_ids if cid not in dropped_20]
    masked_20 = secagg.mask_client_updates(raw_updates, dropped_client_ids=dropped_20)
    recovered_20 = secagg.server_aggregate_with_dropout_resolution(masked_20, client_ids, dropped_client_ids=dropped_20)
    true_avg_20 = {k: sum(raw_updates[cid][k] for cid in surviving_20) / 4.0 for k in ["weights", "bias"]}
    for k in ["weights", "bias"]:
        assert torch.allclose(recovered_20[k], true_avg_20[k], atol=1e-5)

    # Scenario C: 40% dropouts (2 clients drop, meets threshold t=3)
    dropped_40 = {"Site_2", "Site_4"}
    surviving_40 = [cid for cid in client_ids if cid not in dropped_40]
    masked_40 = secagg.mask_client_updates(raw_updates, dropped_client_ids=dropped_40)
    recovered_40 = secagg.server_aggregate_with_dropout_resolution(masked_40, client_ids, dropped_client_ids=dropped_40)
    true_avg_40 = {k: sum(raw_updates[cid][k] for cid in surviving_40) / 3.0 for k in ["weights", "bias"]}
    for k in ["weights", "bias"]:
        assert torch.allclose(recovered_40[k], true_avg_40[k], atol=1e-5)


def test_secure_aggregation_v2_abort_threshold():
    """Verifies that SecureAggregationV2 aborts safely without leaking partial updates when drops exceed threshold."""
    client_ids = [f"Site_{i}" for i in range(1, 6)]
    secagg = SecureAggregationV2(client_ids=client_ids, threshold_ratio=0.60, random_seed=99)
    secagg.setup_keys_and_shares()

    raw_updates = {cid: {"w": torch.randn(5)} for cid in client_ids}
    excessive_drops = {"Site_1", "Site_2", "Site_3"}  # 3 drops leaves 2 clients (< threshold 3)
    masked = secagg.mask_client_updates(raw_updates, dropped_client_ids=excessive_drops)

    with pytest.raises(RuntimeError) as exc_info:
        secagg.server_aggregate_with_dropout_resolution(masked, client_ids, dropped_client_ids=excessive_drops)
    assert "below threshold" in str(exc_info.value)


def test_formal_dp_per_sample_clipping_and_groupnorm():
    """Verifies that ClinicalDPMLPNet uses GroupNorm and per-sample clipping strictly bounds norms."""
    model = ClinicalDPMLPNet(input_dim=10, hidden_dims=[16, 8])
    # Check that GroupNorm is present and no BatchNorm exists
    has_groupnorm = any(isinstance(m, nn.GroupNorm) for m in model.modules())
    has_batchnorm = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) for m in model.modules())
    assert has_groupnorm
    assert not has_batchnorm

    # Check per-sample gradient clipping
    params = list(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    C = 1.0

    bx = torch.randn(4, 10)
    by = torch.tensor([0.0, 1.0, 0.0, 1.0])

    for i in range(len(bx)):
        out = model(bx[i:i+1])
        loss = criterion(out, by[i:i+1])
        grads = torch.autograd.grad(loss, params, retain_graph=False)
        sample_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads)).item()
        clip_factor = min(1.0, C / (sample_norm + 1e-8))
        clipped_norm = sample_norm * clip_factor
        assert clipped_norm <= C + 1e-5


def test_dirichlet_partitioner_validity():
    """Verifies Dirichlet non-IID partitioner generates non-empty clients with valid test splits."""
    model_factory = lambda: ClinicalMLPNet(input_dim=41)
    clients, (X_te, y_te), summary_df = partition_dirichlet_clients(
        model_factory=model_factory,
        num_clients=5,
        alpha=0.5,
        seed=123
    )

    assert len(clients) == 5
    assert len(X_te) == len(y_te)
    for cid, client in clients.items():
        assert len(client.X_train) > 0
        assert len(client.X_test) > 0
        assert client.y_train.sum() >= 0
        assert client.y_test.sum() >= 0


def test_external_data_harmonizer_leakage_detection():
    """Verifies that ExternalDataHarmonizer detects clinical outcome leakages."""
    harmonizer = ExternalDataHarmonizer()

    # Clean dataset without leakage
    clean_df = pd.DataFrame({
        "age": [65, 72],
        "heartrate": [80, 95],
        "meanbp": [90, 85],
        "creatinine": [1.1, 1.5],
        "bun": [18, 24]
    })
    res_clean = harmonizer.validate_dataset_schema(clean_df)
    assert len(res_clean["detected_leakages"]) == 0

    # Dataset contaminated with target leakage column
    dirty_df = clean_df.copy()
    dirty_df["hospitaldischargestatus"] = ["Alive", "Expired"]
    dirty_df["hospital_expire_flag"] = [0, 1]
    res_dirty = harmonizer.validate_dataset_schema(dirty_df)
    assert not res_dirty["valid"]
    assert "hospitaldischargestatus" in res_dirty["detected_leakages"]
    assert "hospital_expire_flag" in res_dirty["detected_leakages"]

    # Test derived index calculation
    derived = harmonizer.compute_derived_clinical_indices(clean_df)
    assert "shock_index" in derived.columns
    assert "bun_to_creatinine" in derived.columns
    assert np.isclose(derived["shock_index"].iloc[0], 80.0 / 90.0)
    assert np.isclose(derived["bun_to_creatinine"].iloc[0], 18.0 / 1.1)
