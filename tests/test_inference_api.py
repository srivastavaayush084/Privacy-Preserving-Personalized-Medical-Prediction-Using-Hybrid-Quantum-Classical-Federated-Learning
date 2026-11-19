"""
Unit tests for Real-Time Medical Risk Inference API endpoints.
"""
import pytest
from src.inference.predict_api import (
    health_check,
    get_features,
    predict_patient_risk,
    PatientPredictionRequest
)


def test_health_check_endpoint():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["qubits"] == 4
    assert data["input_features"] == 41


def test_get_features_endpoint():
    data = get_features()
    assert "feature_names" in data
    assert data["num_features"] == 41


def test_predict_patient_risk_quantum():
    req = PatientPredictionRequest(
        features=[0.0] * 41,
        hospital_consortium_id=1,
        model_type="quantum",
        estimate_uncertainty=True
    )
    res = predict_patient_risk(req)
    assert 0.0 <= res.mortality_risk_score <= 1.0
    assert "Risk" in res.risk_category
    assert len(res.confidence_interval_95) == 2
    assert len(res.top_clinical_drivers) == 5


def test_predict_patient_risk_classical():
    req = PatientPredictionRequest(
        features=[0.5] * 41,
        hospital_consortium_id=3,
        model_type="classical",
        estimate_uncertainty=False
    )
    res = predict_patient_risk(req)
    assert 0.0 <= res.mortality_risk_score <= 1.0
    assert "Risk" in res.risk_category
