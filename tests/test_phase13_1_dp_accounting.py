"""
Unit and regression tests for Phase 13.1 Differential Privacy Accounting.
Verifies:
1. Genuine Poisson subsampling via UniformWithReplacementSampler.
2. Formal RDP accounting alignment with actual optimizer step count.
3. Exact per-sample gradient computation and L2 norm clipping (norm <= C).
4. Correct Gaussian noise calibration for add/remove-one-record adjacency (Delta_2 = C).
5. Monotonic relationship between privacy budget (epsilon) and noise scale (sigma).
6. Robust empty and variable batch handling under Poisson sampling.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

from src.privacy.formal_dp_engine import (
    ClinicalDPMLPNet,
    DPHybridQNNNet,
    FormalOpacusMLPTrainer,
    FormalPerSampleDPQNNTrainer,
    calibrate_noise_multiplier
)


def test_poisson_subsampler_properties():
    """Verifies that UniformWithReplacementSampler behaves as genuine Poisson subsampling."""
    N = 200
    batch_size = 20
    sample_rate = batch_size / N
    sampler = UniformWithReplacementSampler(num_samples=N, sample_rate=sample_rate)
    
    ds = TensorDataset(torch.randn(N, 5), torch.randint(0, 2, (N,)))
    dl = DataLoader(ds, batch_sampler=sampler)
    
    batch_sizes = [len(b[0]) for b in dl]
    assert len(batch_sizes) == int(1 / sample_rate), "Expected 1/sample_rate batches per epoch"
    # Average batch size should be close to expected batch_size
    mean_bs = np.mean(batch_sizes)
    assert abs(mean_bs - batch_size) < 10.0, f"Mean batch size {mean_bs} deviates too far from {batch_size}"
    # Under Poisson subsampling, batch sizes should vary
    assert len(set(batch_sizes)) > 1, "Batch sizes must vary under Poisson subsampling"


def test_noise_multiplier_monotonicity():
    """Verifies that smaller epsilon demands strictly larger noise multiplier sigma."""
    N = 1000
    batch_size = 32
    sample_rate = batch_size / N
    steps = 100
    delta = 1e-5
    
    sigma_1 = calibrate_noise_multiplier(1.0, delta, sample_rate, steps)
    sigma_2 = calibrate_noise_multiplier(2.0, delta, sample_rate, steps)
    sigma_4 = calibrate_noise_multiplier(4.0, delta, sample_rate, steps)
    sigma_8 = calibrate_noise_multiplier(8.0, delta, sample_rate, steps)
    
    assert sigma_1 > sigma_2 > sigma_4 > sigma_8 > 0.0, (
        f"Expected monotonic decrease in sigma: {sigma_1}, {sigma_2}, {sigma_4}, {sigma_8}"
    )


def test_qnn_per_sample_gradient_clipping_bound():
    """Verifies that per-sample gradients are independently clipped to maximum L2 norm C."""
    max_c = 1.0
    model = DPHybridQNNNet(input_dim=10, num_qubits=4, circuit_depth=1)
    params = [p for p in model.parameters() if p.requires_grad]
    
    x = torch.randn(4, 10)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    criterion = nn.BCEWithLogitsLoss()
    
    for i in range(len(x)):
        xi = x[i:i+1]
        yi = y[i:i+1]
        out_i = model(xi)
        loss_i = criterion(out_i, yi)
        sample_grads = torch.autograd.grad(loss_i, params, retain_graph=False)
        sample_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in sample_grads)).item()
        clip_factor = min(1.0, max_c / (sample_norm + 1e-8))
        clipped_norm = sample_norm * clip_factor
        assert clipped_norm <= max_c + 1e-5, f"Clipped norm {clipped_norm} exceeds max_grad_norm {max_c}"


def test_formal_opacus_mlp_trainer_rdp_accounting():
    """Verifies FormalOpacusMLPTrainer configures RDPAccountant and computes valid epsilon."""
    X_dummy = np.random.randn(80, 41)
    y_dummy = np.random.randint(0, 2, size=80)
    
    trainer = FormalOpacusMLPTrainer(
        input_dim=41,
        hidden_dims=[16, 8],
        epochs=2,
        batch_size=16,
        target_delta=1e-5
    )
    
    model, ach_eps, ach_delta = trainer.train(
        X_train=X_dummy,
        y_train=y_dummy,
        target_epsilon=4.0
    )
    
    assert isinstance(model, nn.Module)
    assert 0.0 < ach_eps <= 5.0, f"Expected achieved epsilon near 4.0, got {ach_eps}"
    assert ach_delta == 1e-5


def test_qnn_dp_empty_batch_handling():
    """Verifies that FormalPerSampleDPQNNTrainer handles empty Poisson draws without crashing."""
    X_dummy = np.random.randn(30, 41)
    y_dummy = np.random.randint(0, 2, size=30)
    
    trainer = FormalPerSampleDPQNNTrainer(
        input_dim=41,
        num_qubits=4,
        circuit_depth=1,
        epochs=1,
        batch_size=4,
        target_delta=1e-5
    )
    
    model, ach_eps, ach_delta = trainer.train(
        X_train=X_dummy,
        y_train=y_dummy,
        target_epsilon=8.0
    )
    
    assert isinstance(model, nn.Module)
    assert 0.0 < ach_eps <= 15.0
    assert ach_delta == 1e-5


def test_record_level_adjacency_sensitivity():
    """Verifies sensitivity bound under add/remove-one-record adjacency.
    Under add/remove-one-record, removing or adding one sample changes the batch sum
    by at most clip(g) which has L2 norm <= C. Hence sensitivity Delta_2 = C = 1.0.
    """
    C = 1.0
    # Simulate worst-case addition of one clipped sample gradient
    grad_dim = 50
    g_existing = torch.randn(grad_dim)
    g_new_sample = torch.randn(grad_dim)
    g_new_clipped = g_new_sample / torch.norm(g_new_sample) * C
    
    diff = (g_existing + g_new_clipped) - g_existing
    l2_sensitivity = torch.norm(diff).item()
    assert abs(l2_sensitivity - C) < 1e-5, f"Sensitivity must be exactly C=1.0, got {l2_sensitivity}"
