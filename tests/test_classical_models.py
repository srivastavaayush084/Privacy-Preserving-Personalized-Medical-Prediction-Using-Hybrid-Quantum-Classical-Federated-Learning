"""Unit tests for classical baseline models and evaluation metrics.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical.baselines import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    SVMBaseline,
    XGBoostBaseline,
    MLPBaseline
)
from src.evaluation.metrics import compute_clinical_metrics, compute_expected_calibration_error


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Simple linear separation with noise
    logits = X[:, 0] * 1.5 - X[:, 1] * 1.0 + np.random.randn(100) * 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)
    return X, y


def test_metrics_computation(synthetic_data):
    X, y = synthetic_data
    # Perfect predictor
    metrics_perfect = compute_clinical_metrics(y, y.astype(float))
    assert metrics_perfect["accuracy"] == 1.0
    assert metrics_perfect["roc_auc"] == 1.0
    assert metrics_perfect["brier_score"] == 0.0
    assert metrics_perfect["ece"] == 0.0


def test_logistic_regression_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = LogisticRegressionBaseline(C=1.0)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_random_forest_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = RandomForestBaseline(n_estimators=20, max_depth=4)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert len(model.get_feature_importances()) == X.shape[1]


def test_svm_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = SVMBaseline(C=1.0)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_xgboost_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = XGBoostBaseline(n_estimators=20, max_depth=3)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_mlp_fit_predict(synthetic_data):
    X, y = synthetic_data
    model = MLPBaseline(input_dim=X.shape[1], hidden_dims=[16, 8], epochs=10, batch_size=32)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(y)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
