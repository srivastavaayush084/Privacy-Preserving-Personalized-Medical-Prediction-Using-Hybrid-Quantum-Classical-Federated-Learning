"""
Unit tests for Phase 13.3.1: Critical DP Expected-Batch-Size and Sampling-Rate Consistency.
Verifies exact Poisson sampling rate q, exact float expected batch size B_exp = q*N,
and mathematical sensitivity post-processing invariance for both QNN and MLP.
"""

import math
import numpy as np
import pytest
import torch
from opacus.accountants import RDPAccountant

from src.privacy.formal_dp_engine import (
    FormalOpacusMLPTrainer,
    FormalPerSampleDPQNNTrainer,
    calibrate_noise_multiplier
)


def test_qnn_expected_batch_size_float_exactness():
    """Verifies that FormalPerSampleDPQNNTrainer uses exact float B_exp = q * N
    where q = 1 / steps_per_epoch, rather than nominal integer batch_size."""
    N = 1132
    batch_size = 32
    trainer = FormalPerSampleDPQNNTrainer(
        input_dim=10,
        num_qubits=4,
        circuit_depth=1,
        epochs=1,
        batch_size=batch_size,
        target_delta=1e-5
    )

    X = np.random.randn(N, 10)
    y = np.random.randint(0, 2, size=N)

    model, ach_eps, ach_delta = trainer.train(X, y, target_epsilon=4.0)

    stats = trainer.last_run_stats
    expected_steps_per_epoch = math.ceil(N / batch_size)  # 36
    expected_q = 1.0 / expected_steps_per_epoch  # 1/36 = 0.027777777777777776
    expected_B_exp = N * expected_q  # 31.444444444444443

    assert abs(stats["sample_rate"] - expected_q) < 1e-8, (
        f"Sample rate mismatch: expected {expected_q}, got {stats['sample_rate']}"
    )
    assert abs(stats["expected_batch_size"] - expected_B_exp) < 1e-8, (
        f"Expected batch size mismatch: expected {expected_B_exp}, got {stats['expected_batch_size']}"
    )
    assert stats["nominal_batch_size"] == 32
    assert stats["clipping_bound"] == 1.0


def test_mlp_expected_batch_size_float_exactness():
    """Verifies that FormalOpacusMLPTrainer uses exact float B_exp = q * N
    where q = 1 / len(data_loader), overriding Opacus integer truncation."""
    N = 1132
    batch_size = 64
    trainer = FormalOpacusMLPTrainer(
        input_dim=10,
        hidden_dims=[16],
        lr=0.01,
        epochs=1,
        batch_size=batch_size,
        target_delta=1e-5
    )

    X = np.random.randn(N, 10)
    y = np.random.randint(0, 2, size=N)

    model, ach_eps, ach_delta = trainer.train(X, y, target_epsilon=4.0)

    stats = trainer.last_run_stats
    expected_steps_per_epoch = math.ceil(N / batch_size)  # 18
    expected_q = 1.0 / expected_steps_per_epoch  # 1/18 = 0.05555555555555555
    expected_B_exp = N * expected_q  # 62.888888888888886

    assert abs(stats["sample_rate"] - expected_q) < 1e-8, (
        f"Sample rate mismatch: expected {expected_q}, got {stats['sample_rate']}"
    )
    assert abs(stats["expected_batch_size"] - expected_B_exp) < 1e-8, (
        f"Expected batch size mismatch: expected {expected_B_exp}, got {stats['expected_batch_size']}"
    )
    assert stats["nominal_batch_size"] == 64
    assert stats["clipping_bound"] == 1.0


def test_unbiased_gradient_expectation():
    """Verifies mathematically that dividing by B_exp = q*N yields an unbiased estimator
    of the population mean gradient under Poisson subsampling, whereas dividing by
    nominal B yields a biased estimator."""
    N = 1132
    q = 1.0 / 36.0
    B_exp = q * N  # 31.4444...
    nominal_B = 32.0

    # Let individual gradients be ones
    g_sum_per_epoch = 0.0
    num_simulations = 50000
    rng = np.random.default_rng(42)

    # Simulate batch sizes |B| ~ Binomial(N, q)
    batch_sizes = rng.binomial(N, q, size=num_simulations)

    # Normalization by exact B_exp
    estimator_exact = np.mean(batch_sizes) / B_exp
    # Normalization by nominal B
    estimator_nominal = np.mean(batch_sizes) / nominal_B

    # True population mean per-sample gradient is 1.0
    assert abs(estimator_exact - 1.0) < 0.005, f"Exact estimator should be 1.0, got {estimator_exact}"
    assert abs(estimator_nominal - (B_exp / nominal_B)) < 0.005, (
        f"Nominal estimator has 1.74% bias: expected {B_exp / nominal_B}, got {estimator_nominal}"
    )
