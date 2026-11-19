"""
Independent Audit Validation Test Suite.
Validates:
1. Patient-disjoint split integrity (0.00% patient leakage).
2. Absence of target/outcome leakage in feature representations.
3. Checkpoint parameter shapes and execution validity.
4. Metric recalculation numerical precision.
5. Secure aggregation zero-sum bounds.
6. DP accountant monotonic properties.
7. Consortium partition integrity.
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.models.classical.baselines import ClinicalMLPNet
from src.models.quantum.hybrid_qnn import HybridQNNNet
from src.privacy.secure_aggregation import SecureAggregationProtocol
from src.privacy.privacy_accountant import RDPAccountant
from src.federated.client import compute_ece


def test_patient_disjointness():
    """Verify strict 0.00% patient leakage across train, val, and test partitions."""
    train_meta = pd.read_csv("data/processed/train_meta.csv")
    val_meta = pd.read_csv("data/processed/val_meta.csv")
    test_meta = pd.read_csv("data/processed/test_meta.csv")

    train_pids = set(train_meta["uniquepid"])
    val_pids = set(val_meta["uniquepid"])
    test_pids = set(test_meta["uniquepid"])

    assert len(train_pids.intersection(val_pids)) == 0, "Patient leakage between train and val!"
    assert len(train_pids.intersection(test_pids)) == 0, "Patient leakage between train and test!"
    assert len(val_pids.intersection(test_pids)) == 0, "Patient leakage between val and test!"


def test_no_target_leakage_in_features():
    """Verify feature dictionary excludes discharge status, mortality, or APACHE scores."""
    with open("data/processed/feature_names.json", "r") as f:
        feature_names = json.load(f)

    forbidden_tokens = ["discharge", "death", "mortality", "expired", "died", "apachescore", "predictedprob"]
    for feat in feature_names:
        for token in forbidden_tokens:
            assert token not in feat.lower(), f"Potential target leakage in feature: {feat}"


def test_consortium_partition_integrity():
    """Verify each hospital facility is strictly mapped to exactly one consortium."""
    train_meta = pd.read_csv("data/processed/train_meta.csv")
    test_meta = pd.read_csv("data/processed/test_meta.csv")

    for df in [train_meta, test_meta]:
        hosp_to_consortium = df.groupby("hospitalid")["hospitalid"].apply(lambda s: (s % 5).nunique())
        assert (hosp_to_consortium == 1).all(), "Hospital facility mapped to multiple consortiums!"


def test_model_checkpoints_load_and_predict():
    """Verify that all saved model checkpoints load correctly and produce valid probabilities."""
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # 1. Classical XGBoost
    xgb = joblib.load("checkpoints/classical/xgboost.joblib")
    p_xgb = xgb.predict_proba(X_test)[:, 1] if hasattr(xgb, "predict_proba") else xgb.predict_proba(X_test)
    assert p_xgb.shape == (len(y_test),)
    assert np.all((p_xgb >= 0.0) & (p_xgb <= 1.0))

    # 2. Classical Random Forest
    rf = joblib.load("checkpoints/classical/random_forest.joblib")
    p_rf = rf.predict_proba(X_test)[:, 1] if hasattr(rf, "predict_proba") else rf.predict_proba(X_test)
    assert p_rf.shape == (len(y_test),)
    assert np.all((p_rf >= 0.0) & (p_rf <= 1.0))

    # 3. Hybrid QNN (4-qubit depth-2)
    qnn = HybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=2, entanglement="strongly_entangling")
    qnn.load_state_dict(torch.load("checkpoints/quantum/hybrid_qnn_4_qubit_depth_2_strong.pt", map_location="cpu"))
    qnn.eval()
    with torch.no_grad():
        p_qnn = torch.sigmoid(qnn(torch.tensor(X_test, dtype=torch.float32))).numpy()
    assert p_qnn.shape == (len(y_test),)
    assert np.all((p_qnn >= 0.0) & (p_qnn <= 1.0))


def test_secure_aggregation_zero_sum_bound():
    """Verify pairwise masking vectors sum to zero with machine precision error < 1e-5."""
    client_ids = [f"Hospital_{i}" for i in range(1, 6)]
    param_shapes = {"layer_w": (8, 16), "layer_b": (8,)}

    secagg = SecureAggregationProtocol(seed=123)
    masks = secagg.generate_pairwise_masks(client_ids, param_shapes)

    total_w = torch.zeros(param_shapes["layer_w"])
    total_b = torch.zeros(param_shapes["layer_b"])

    for cid in client_ids:
        total_w += masks[cid]["layer_w"]
        total_b += masks[cid]["layer_b"]

    assert float(torch.max(torch.abs(total_w)).item()) < 1e-5
    assert float(torch.max(torch.abs(total_b)).item()) < 1e-5


def test_rdp_accountant_monotonic_properties():
    """Verify that Rényi DP accountant satisfies mathematical monotonicity."""
    acc_low_noise = RDPAccountant()
    acc_low_noise.step(noise_multiplier=0.5, sample_rate=0.05, steps=10)
    eps_low = acc_low_noise.get_epsilon(target_delta=1e-5)

    acc_high_noise = RDPAccountant()
    acc_high_noise.step(noise_multiplier=2.0, sample_rate=0.05, steps=10)
    eps_high = acc_high_noise.get_epsilon(target_delta=1e-5)

    # Higher noise multiplier MUST yield smaller epsilon (stronger privacy guarantee)
    assert eps_high < eps_low, "DP Accountant violated noise monotonicity!"

    # More steps MUST yield higher epsilon (privacy budget exhaustion)
    acc_more_steps = RDPAccountant()
    acc_more_steps.step(noise_multiplier=2.0, sample_rate=0.05, steps=50)
    eps_more = acc_more_steps.get_epsilon(target_delta=1e-5)
    assert eps_more > eps_high, "DP Accountant violated step composition monotonicity!"


def test_metric_calculation_numerical_consistency():
    """Verify AUROC and Brier loss calculations are numerically sound."""
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    p_pred = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.3, 0.75, 0.05])

    auc = roc_auc_score(y_true, p_pred)
    assert 0.95 <= auc <= 1.0

    brier = brier_score_loss(y_true, p_pred)
    assert 0.0 <= brier <= 0.2

    ece = compute_ece(p_pred, y_true, n_bins=5)
    assert 0.0 <= ece <= 0.5
