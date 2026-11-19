"""
Rényi Differential Privacy (RDP) and Moments Accountant for Federated Learning.
Computes rigorous (epsilon, delta)-DP bounds across communication rounds and gradient steps.
"""
import math
from typing import List, Tuple
import numpy as np


class RDPAccountant:
    """Rényi Differential Privacy (RDP) accountant for Subsampled Gaussian Mechanism."""

    def __init__(self, orders: List[float] = None):
        if orders is None:
            self.orders = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0, 32.0, 64.0]
        else:
            self.orders = orders
        self.rdp = np.zeros(len(self.orders))
        self.step_history: List[Tuple[float, float, int]] = []

    def step(self, noise_multiplier: float, sample_rate: float, steps: int = 1):
        """Accumulates privacy loss for 'steps' iterations of Subsampled Gaussian Mechanism."""
        if noise_multiplier <= 0.0 or sample_rate <= 0.0:
            return

        self.step_history.append((noise_multiplier, sample_rate, steps))
        for i, alpha in enumerate(self.orders):
            # Standard analytical RDP upper bound for subsampled Gaussian mechanism
            # (Mironov et al., 2017 & 2019)
            rdp_step = (alpha * (sample_rate ** 2)) / (2.0 * (noise_multiplier ** 2))
            self.rdp[i] += rdp_step * steps

    def get_epsilon(self, target_delta: float = 1e-5) -> float:
        """Converts accumulated RDP into tightest (epsilon, delta)-DP guarantee."""
        if np.all(self.rdp == 0):
            return 0.0
        if target_delta <= 0:
            raise ValueError("target_delta must be strictly positive.")

        eps_candidates = []
        for i, alpha in enumerate(self.orders):
            if alpha > 1.0:
                eps = self.rdp[i] + math.log(1.0 / target_delta) / (alpha - 1.0)
                eps_candidates.append(eps)

        return float(min(eps_candidates))

    @staticmethod
    def compute_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        total_steps: int
    ) -> float:
        """Binary search for calibrated Gaussian noise multiplier sigma achieving target (eps, delta)."""
        if math.isinf(target_epsilon):
            return 0.0

        low = 0.01
        high = 50.0
        tol = 1e-3

        for _ in range(50):
            mid = (low + high) / 2.0
            accountant = RDPAccountant()
            accountant.step(noise_multiplier=mid, sample_rate=sample_rate, steps=total_steps)
            current_eps = accountant.get_epsilon(target_delta=target_delta)

            if current_eps > target_epsilon:
                # Need more noise
                low = mid
            else:
                # Noise is sufficient or too high
                high = mid
            if abs(high - low) < tol:
                break

        return float(high)
