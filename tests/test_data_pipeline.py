"""Unit tests for eICU data loading, validation, feature engineering, and patient-disjoint splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_loader import EICUDataLoader
from src.data.data_validator import ClinicalDataValidator
from src.data.feature_engineering import ClinicalFeatureEngineer
from src.data.preprocessing import ClinicalPreprocessor
from src.data.dataset_splitter import PatientDisjointSplitter


def test_data_loader_initialization():
    loader = EICUDataLoader(data_dir="eicu-collaborative-research-database-demo-2.0.1")
    assert loader is not None


def test_clinical_bounds_sanitization():
    validator = ClinicalDataValidator()
    dummy_df = pd.DataFrame({
        "age": ["> 89", "45", "nan"],
        "heartrate": [75.0, 450.0, -1.0],  # 450 is out of bounds, -1.0 is missing indicator
        "meanbp": [85.0, 10.0, 320.0]     # 320 is out of bounds
    })
    cleaned = validator.sanitize_physiological_values(dummy_df)
    assert cleaned["age"].iloc[0] == 90.0
    assert cleaned["age"].iloc[1] == 45.0
    assert pd.isna(cleaned["age"].iloc[2])
    assert cleaned["heartrate"].iloc[0] == 75.0
    assert pd.isna(cleaned["heartrate"].iloc[1])
    assert pd.isna(cleaned["heartrate"].iloc[2])
    assert pd.isna(cleaned["meanbp"].iloc[2])


def test_feature_engineering_derived_indices():
    engineer = ClinicalFeatureEngineer()
    dummy_df = pd.DataFrame({
        "admissionheight": [175.0, 160.0],
        "admissionweight": [70.0, 60.0],
        "heartrate": [100.0, 80.0],
        "meanbp": [50.0, 80.0],
        "bun": [40.0, 15.0],
        "creatinine": [2.0, 1.0],
        "eyes": [3, 4],
        "motor": [5, 6],
        "verbal": [4, 5],
        "aids": [0, 1],
        "diabetes": [1, 0],
        "intubated": [1, 0],
        "vent": [1, 0],
        "dialysis": [0, 0]
    })
    feat_df = engineer.compute_features(dummy_df)
    # Check BMI: 70 / (1.75^2) = 22.857
    assert pytest.approx(feat_df["bmi"].iloc[0], rel=1e-2) == 22.86
    # Shock index: 100 / 50 = 2.0
    assert pytest.approx(feat_df["shock_index"].iloc[0], rel=1e-2) == 2.0
    # BUN to Creatinine: 40 / 2 = 20.0
    assert pytest.approx(feat_df["bun_to_creatinine"].iloc[0], rel=1e-2) == 20.0
    # GCS total: 3 + 5 + 4 = 12
    assert feat_df["gcs_total"].iloc[0] == 12
    # Comorbidity burden: 1
    assert feat_df["comorbidity_burden"].iloc[0] == 1
    # Organ support: intubated + vent = 2
    assert feat_df["organ_support_count"].iloc[0] == 2


def test_patient_disjoint_splitting_zero_leakage():
    splitter = PatientDisjointSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    # Simulate patients with multiple admissions
    patients = [f"patient_{i}" for i in range(100)] * 3
    df = pd.DataFrame({
        "uniquepid": patients,
        "patientunitstayid": range(len(patients)),
        "val": np.random.randn(len(patients))
    })
    df_train, df_val, df_test = splitter.split(df, patient_col="uniquepid")

    train_pts = set(df_train["uniquepid"])
    val_pts = set(df_val["uniquepid"])
    test_pts = set(df_test["uniquepid"])

    assert len(train_pts.intersection(val_pts)) == 0
    assert len(train_pts.intersection(test_pts)) == 0
    assert len(val_pts.intersection(test_pts)) == 0


def test_preprocessor_no_nans_output():
    df_train = pd.DataFrame({
        "age": [50.0, np.nan, 70.0],
        "bmi": [25.0, 30.0, np.nan],
        "heartrate": [80.0, 90.0, 70.0],
        "meanbp": [70.0, 80.0, 90.0],
        "respiratoryrate": [18.0, 20.0, 16.0],
        "temperature": [37.0, 36.5, 38.0],
        "sodium": [140.0, 138.0, 142.0],
        "creatinine": [1.0, 1.2, 0.9],
        "bun": [20.0, 25.0, 18.0],
        "glucose": [110.0, 120.0, 100.0],
        "wbc": [8.0, 9.0, 7.5],
        "hematocrit": [35.0, 38.0, 33.0],
        "shock_index": [1.1, 1.1, 0.77],
        "bun_to_creatinine": [20.0, 20.8, 20.0],
        "gcs_total": [15, 14, 15],
        "comorbidity_burden": [0, 1, 0],
        "organ_support_count": [0, 1, 0],
        "gender": ["Female", "Male", "Female"],
        "ethnicity": ["Caucasian", "African American", "Caucasian"],
        "unittype": ["MICU", "SICU", "MICU"],
        "region": ["Midwest", "South", "West"],
        "ph_is_missing": [0.0, 1.0, 0.0],
        "pao2_is_missing": [0.0, 1.0, 0.0],
        "bilirubin_is_missing": [0.0, 0.0, 1.0],
        "albumin_is_missing": [1.0, 0.0, 0.0]
    })
    prep = ClinicalPreprocessor()
    X = prep.fit_transform(df_train)
    assert not np.isnan(X).any()
    assert X.shape[0] == 3

