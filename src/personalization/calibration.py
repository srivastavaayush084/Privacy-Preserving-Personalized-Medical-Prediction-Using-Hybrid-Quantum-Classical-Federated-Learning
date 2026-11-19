"""
Hospital-Specific Probability Calibration Module.
Applies temperature scaling and Platt scaling on local hospital validation data
to align predicted probabilities with local disease prevalence and mortality rates.
"""
import logging
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.federated.client import compute_ece

logger = logging.getLogger(__name__)


class HospitalCalibrator:
    """Calibrates model output logits to local hospital cohort prevalence."""

    def __init__(self, method: str = "temperature"):
        """
        Args:
            method: 'temperature' or 'platt'
        """
        self.method = method
        self.temperature = 1.0
        self.platt_model: Optional[LogisticRegression] = None

    def fit(self, logits: np.ndarray, y_val: np.ndarray):
        """Fits calibration parameters on local hospital validation logits."""
        if len(y_val) == 0 or len(np.unique(y_val)) < 2:
            return self

        if self.method == "temperature":
            # Optimize temperature T > 0 using NLL on validation set
            t = nn.Parameter(torch.ones(1) * 1.5)
            optimizer = optim.LBFGS([t], lr=0.01, max_iter=50)
            criterion = nn.BCEWithLogitsLoss()

            logits_t = torch.tensor(logits, dtype=torch.float32)
            y_t = torch.tensor(y_val, dtype=torch.float32)

            def eval_nll():
                optimizer.zero_grad()
                scaled_logits = logits_t / torch.clamp(t, min=0.01, max=10.0)
                loss = criterion(scaled_logits, y_t)
                loss.backward()
                return loss

            try:
                optimizer.step(eval_nll)
                self.temperature = float(torch.clamp(t, min=0.01, max=10.0).item())
            except Exception as e:
                logger.warning(f"Temperature optimization failed, defaulting to 1.0: {e}")
                self.temperature = 1.0

        elif self.method == "platt":
            # Platt scaling: fit univariate logistic regression on logits
            self.platt_model = LogisticRegression(C=1.0, solver="lbfgs")
            self.platt_model.fit(logits.reshape(-1, 1), y_val)

        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Transforms raw test logits into calibrated probability scores."""
        if self.method == "temperature":
            scaled = logits / max(self.temperature, 0.01)
            probs = 1.0 / (1.0 + np.exp(-scaled))
            return probs
        elif self.method == "platt" and self.platt_model is not None:
            return self.platt_model.predict_proba(logits.reshape(-1, 1))[:, 1]
        else:
            return 1.0 / (1.0 + np.exp(-logits))

    def evaluate_calibration(
        self,
        raw_logits: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculates Brier score and ECE before and after calibration."""
        uncal_probs = 1.0 / (1.0 + np.exp(-raw_logits))
        cal_probs = self.calibrate(raw_logits)

        brier_before = brier_score_loss(y_test, uncal_probs)
        brier_after = brier_score_loss(y_test, cal_probs)
        ece_before = compute_ece(uncal_probs, y_test)
        ece_after = compute_ece(cal_probs, y_test)

        return {
            "temperature": self.temperature,
            "brier_before": float(brier_before),
            "brier_after": float(brier_after),
            "brier_delta": float(brier_after - brier_before),
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "ece_reduction": float(ece_before - ece_after)
        }
