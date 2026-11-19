"""
Differential Privacy Module for Federated Clinical Model Training.
Implements bounded L2 gradient clipping, calibrated Gaussian perturbation,
and strict privacy accountant tracking via Rényi Differential Privacy (RDP).
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.privacy.privacy_accountant import RDPAccountant

logger = logging.getLogger(__name__)


class DPGradientEngine:
    """Applies clipping and calibrated Gaussian noise to model parameter gradients or updates."""

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        target_delta: float = 1e-5
    ):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.accountant = RDPAccountant()

    def clip_and_perturb_gradients(
        self,
        model: nn.Module,
        batch_size: int,
        dataset_size: int
    ) -> float:
        """Clips total parameter gradient norm to clip_norm and injects Gaussian noise."""
        # 1. Compute total L2 norm of gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # 2. Bounded L2 clipping
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

        # 3. Add calibrated Gaussian noise: N(0, (noise_multiplier * clip_norm)^2)
        if self.noise_multiplier > 0:
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.randn_like(p.grad.data) * (self.noise_multiplier * self.clip_norm)
                    p.grad.data.add_(noise)

        # 4. Account for privacy loss
        sample_rate = float(batch_size) / float(max(dataset_size, 1))
        self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=sample_rate, steps=1)

        return float(self.accountant.get_epsilon(self.target_delta))

    def perturb_model_delta(
        self,
        weights_delta: Dict[str, torch.Tensor],
        num_samples: int,
        client_dataset_size: int
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Applies clipping and calibrated noise directly to weight updates (Client-level DP)."""
        # 1. Total delta L2 norm
        total_norm = 0.0
        for k, v in weights_delta.items():
            if v.dtype in (torch.float32, torch.float64):
                total_norm += float(torch.norm(v, p=2).item()) ** 2
        total_norm = total_norm ** 0.5

        # 2. Clip delta
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))
        perturbed_delta = {}
        for k, v in weights_delta.items():
            if v.dtype in (torch.float32, torch.float64):
                clipped_val = v * clip_coef
                if self.noise_multiplier > 0:
                    noise = torch.randn_like(v) * (self.noise_multiplier * self.clip_norm)
                    perturbed_delta[k] = clipped_val + noise
                else:
                    perturbed_delta[k] = clipped_val
            else:
                perturbed_delta[k] = v.clone()

        # 3. Privacy loss
        sample_rate = 1.0  # Entire client dataset participation
        self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=sample_rate, steps=1)
        eps = float(self.accountant.get_epsilon(self.target_delta))

        return perturbed_delta, eps
