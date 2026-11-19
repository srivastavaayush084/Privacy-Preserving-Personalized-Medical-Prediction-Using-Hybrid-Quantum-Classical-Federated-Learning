"""Classical machine learning baseline models for clinical risk prediction.
Includes Logistic Regression, Random Forest, Support Vector Classifier, XGBoost, and PyTorch MLP.
Configured with thread safety to prevent OpenMP/loky deadlocks on Windows.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Enforce single-thread execution in PyTorch to prevent OpenMP deadlocks with Scikit-learn
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# 1. LOGISTIC REGRESSION
# -------------------------------------------------------------
class LogisticRegressionBaseline:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


# -------------------------------------------------------------
# 2. RANDOM FOREST
# -------------------------------------------------------------
class RandomForestBaseline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 8, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1  # Thread-safe on Windows
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


# -------------------------------------------------------------
# 3. SUPPORT VECTOR MACHINE (with CalibratedClassifierCV)
# -------------------------------------------------------------
class SVMBaseline:
    def __init__(self, C: float = 1.0, kernel: str = "rbf", random_state: int = 42):
        base_svc = SVC(
            C=C,
            kernel=kernel,
            class_weight="balanced",
            random_state=random_state
        )
        self.model = CalibratedClassifierCV(base_svc, ensemble=False)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


# -------------------------------------------------------------
# 4. XGBOOST
# -------------------------------------------------------------
class XGBoostBaseline:
    def __init__(self, n_estimators: int = 150, max_depth: int = 4, learning_rate: float = 0.05, random_state: int = 42):
        import xgboost as xgb
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        num_neg = (y == 0).sum()
        num_pos = (y == 1).sum()
        if num_pos > 0:
            self.model.set_params(scale_pos_weight=float(num_neg / num_pos))
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self


# -------------------------------------------------------------
# 5. PYTORCH MULTI-LAYER PERCEPTRON (MLP)
# -------------------------------------------------------------
class ClinicalMLPNet(nn.Module):
    """Deep tabular neural network architecture with BatchNorm and Dropout."""
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16], dropout: float = 0.25):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x).squeeze(-1)
        return logits


class MLPBaseline:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 32, 16],
        dropout: float = 0.25,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 64,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = ClinicalMLPNet(input_dim, hidden_dims, dropout).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        n_samples = len(X_train)

        self.model.train()
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx: start_idx + self.batch_size]
                batch_x = X_t[batch_idx]
                batch_y = y_t[batch_idx]

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)

            if (epoch + 1) % 10 == 0:
                logger.info(f"MLP Epoch {epoch+1}/{self.epochs} - Loss: {total_loss / n_samples:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, input_dim: int):
        self.model = ClinicalMLPNet(input_dim).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        return self
