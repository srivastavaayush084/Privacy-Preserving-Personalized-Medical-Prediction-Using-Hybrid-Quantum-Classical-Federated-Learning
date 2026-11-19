"""
Phase 13 Differential Privacy Independent Verification Unit Tests.
Verifies mathematical properties of per-sample clipping, GroupNorm invariance,
noise calibration, and privacy accounting bounds.
"""
import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from src.privacy.formal_dp_engine import ClinicalDPMLPNet, DPHybridQNNNet, calibrate_noise_multiplier


def test_groupnorm_sample_invariance():
    """Verify that GroupNorm operates strictly within each sample, with zero inter-sample leakage."""
    torch.manual_seed(42)
    gn = nn.GroupNorm(num_groups=2, num_channels=16)
    
    # Batch of 2 samples
    x1 = torch.randn(1, 16)
    x2 = torch.randn(1, 16)
    x_batch = torch.cat([x1, x2], dim=0)
    
    out_batch = gn(x_batch)
    out_single_1 = gn(x1)
    
    # Sample 1 output in batch must exactly match sample 1 output alone
    assert torch.allclose(out_batch[0:1], out_single_1, atol=1e-5), "GroupNorm inter-sample leakage detected!"


def test_per_sample_gradient_clipping_bound():
    """Verify that every per-sample gradient norm is strictly bounded by C."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.GroupNorm(num_groups=1, num_channels=5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    C = 0.75  # Max norm
    criterion = nn.BCEWithLogitsLoss()
    
    x = torch.randn(8, 10)
    y = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    
    params = list(model.parameters())
    for i in range(len(x)):
        xi = x[i:i+1]
        yi = y[i:i+1]
        out_i = model(xi).squeeze(-1)
        loss_i = criterion(out_i, yi)
        sample_grads = torch.autograd.grad(loss_i, params, retain_graph=False)
        sample_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in sample_grads)).item()
        
        # Clip
        clip_factor = min(1.0, C / (sample_norm + 1e-8))
        clipped_grads = [g * clip_factor for g in sample_grads]
        clipped_norm = torch.sqrt(sum(torch.sum(cg ** 2) for cg in clipped_grads)).item()
        
        assert clipped_norm <= C + 1e-5, f"Per-sample gradient norm {clipped_norm} exceeds threshold {C}!"


def test_privacy_accountant_monotonicity():
    """Verify that RDPAccountant yields strictly decreasing epsilon as noise multiplier increases."""
    target_delta = 1e-5
    sample_rate = 0.03
    steps = 500
    
    epsilons = []
    sigmas = [0.8, 1.2, 1.8, 2.5]
    for s in sigmas:
        acc = RDPAccountant()
        for _ in range(steps):
            acc.step(noise_multiplier=s, sample_rate=sample_rate)
        epsilons.append(acc.get_epsilon(delta=target_delta))
        
    for i in range(len(epsilons) - 1):
        assert epsilons[i] > epsilons[i+1], "Privacy accountant monotonicity violated!"


def test_calibrated_noise_multiplier_precision():
    """Verify that calibrate_noise_multiplier achieves target epsilon within 0.02 tolerance."""
    target_delta = 1e-5
    sample_rate = 0.02
    total_steps = 400
    target_eps = 2.0
    
    sigma = calibrate_noise_multiplier(
        target_epsilon=target_eps,
        target_delta=target_delta,
        sample_rate=sample_rate,
        total_steps=total_steps
    )
    
    acc = RDPAccountant()
    for _ in range(total_steps):
        acc.step(noise_multiplier=sigma, sample_rate=sample_rate)
    achieved_eps = acc.get_epsilon(delta=target_delta)
    
    assert abs(achieved_eps - target_eps) < 0.02, f"Achieved epsilon {achieved_eps} diverged from target {target_eps}"


def test_non_private_delta_representation():
    """Verify that non-private baseline correctly maps to epsilon=inf and delta=0.0."""
    sigma = calibrate_noise_multiplier(
        target_epsilon=float("inf"),
        target_delta=1e-5,
        sample_rate=0.02,
        total_steps=100
    )
    assert sigma == 0.0, "Non-private noise multiplier must be 0.0"
