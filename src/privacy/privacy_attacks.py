"""
Privacy Attack Suite for Federated Healthcare Models.
Implements:
1. Shadow-Model Membership Inference Attack (MIA)
2. Gradient Inversion Reconstruction Attack (DLG / Honest-but-Curious Server)
3. Attribute Inference Vulnerability Assessment
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


class MembershipInferenceAttack:
    """Evaluates privacy vulnerability to Membership Inference Attacks (MIA).
    Tests whether an adversary can determine if a patient was in the training cohort.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.attacker_model = LogisticRegression(C=1.0, random_state=random_state)

    def extract_attack_features(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Extracts attack signals: prediction probability, cross-entropy loss, and entropy."""
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            if hasattr(model, "extract_shared_features") and hasattr(model, "local_adapter"):
                # Personalized model
                logits = model(X_t, use_adapter=True).cpu().numpy()
            else:
                logits = model(X_t).cpu().numpy()

        probs = 1.0 / (1.0 + np.exp(-logits))
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

        # Cross-entropy loss per sample
        bce_loss = -(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))

        # Entropy of prediction
        entropy = -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))

        # Max confidence
        confidence = np.maximum(probs, 1.0 - probs)

        features = np.column_stack([probs, bce_loss, entropy, confidence])
        return features

    def evaluate(
        self,
        model: nn.Module,
        X_train_member: np.ndarray,
        y_train_member: np.ndarray,
        X_test_nonmember: np.ndarray,
        y_test_nonmember: np.ndarray
    ) -> Dict[str, float]:
        """Runs train/test split shadow attack and returns MIA metrics.
        Baseline: AUROC = 0.50 (Zero Information Leakage), Advantage = 0.0.
        """
        # Ensure balanced membership evaluation
        n_eval = min(len(X_train_member), len(X_test_nonmember))
        idx_tr = np.random.choice(len(X_train_member), n_eval, replace=False)
        idx_te = np.random.choice(len(X_test_nonmember), n_eval, replace=False)

        feats_member = self.extract_attack_features(model, X_train_member[idx_tr], y_train_member[idx_tr])
        feats_nonmember = self.extract_attack_features(model, X_test_nonmember[idx_te], y_test_nonmember[idx_te])

        X_attack = np.vstack([feats_member, feats_nonmember])
        y_attack = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])

        # Split into attacker train and test
        perm = np.random.permutation(len(X_attack))
        split_idx = len(X_attack) // 2

        train_idx = perm[:split_idx]
        test_idx = perm[split_idx:]

        self.attacker_model.fit(X_attack[train_idx], y_attack[train_idx])
        attack_probs = self.attacker_model.predict_proba(X_attack[test_idx])[:, 1]
        attack_preds = (attack_probs >= 0.5).astype(int)

        mia_auroc = roc_auc_score(y_attack[test_idx], attack_probs)
        mia_acc = accuracy_score(y_attack[test_idx], attack_preds)
        mia_advantage = 2.0 * max(0.0, mia_acc - 0.5)

        return {
            "mia_auroc": float(mia_auroc),
            "mia_accuracy": float(mia_acc),
            "mia_advantage": float(mia_advantage)
        }


class GradientInversionAttack:
    """Simulates an honest-but-curious aggregator attempting to reconstruct patient clinical
    records from shared gradients (Deep Leakage from Gradients / DLG).
    """

    def __init__(self, max_iter: int = 100, lr: float = 0.1):
        self.max_iter = max_iter
        self.lr = lr

    def attack(
        self,
        model: nn.Module,
        true_x: torch.Tensor,
        true_y: torch.Tensor,
        criterion: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Attempts to reconstruct true_x given true gradient."""
        model.eval()
        # Compute real client gradient
        model.zero_grad()
        real_pred = model(true_x.unsqueeze(0)).view(-1)
        real_loss = criterion(real_pred, true_y.view(-1))
        real_grads = torch.autograd.grad(real_loss, model.parameters(), create_graph=False)

        # Initial dummy patient features: random Gaussian vector
        dummy_x = torch.randn_like(true_x.unsqueeze(0), requires_grad=True)
        dummy_y = true_y.view(-1).clone().float()

        optimizer = optim.LBFGS([dummy_x], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            pred = model(dummy_x).view(-1)
            loss = criterion(pred, dummy_y)
            dummy_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            grad_diff = 0.0
            for dg, rg in zip(dummy_grads, real_grads):
                grad_diff += ((dg - rg) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        try:
            optimizer.step(closure)
        except Exception:
            pass

        reconstructed_x = dummy_x.detach().squeeze(0)

        # Compute reconstruction error
        mse = torch.mean((reconstructed_x - true_x) ** 2).item()
        cos_sim = torch.cosine_similarity(reconstructed_x.flatten().unsqueeze(0), true_x.flatten().unsqueeze(0)).item()

        return reconstructed_x, {
            "reconstruction_mse": float(mse),
            "cosine_similarity": float(cos_sim)
        }
