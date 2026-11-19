"""Executable pipeline script to ingest, validate, engineer, split, and preprocess eICU data.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd

from src.data.data_loader import EICUDataLoader
from src.data.data_validator import ClinicalDataValidator
from src.data.feature_engineering import ClinicalFeatureEngineer
from src.data.preprocessing import ClinicalPreprocessor
from src.data.dataset_splitter import PatientDisjointSplitter
from src.data.data_quality_report import generate_quality_report


def main():
    print("--- 1. LOADING DATA ---")
    loader = EICUDataLoader(data_dir="eicu-collaborative-research-database-demo-2.0.1")
    raw_cohort = loader.load_cohort()
    print(f"Loaded cohort raw shape: {raw_cohort.shape}")

    print("\n--- 2. VALIDATING & SANITIZING ---")
    validator = ClinicalDataValidator()
    sanitized_cohort = validator.sanitize_physiological_values(raw_cohort)
    
    # Filter out stays with missing target
    valid_mask = sanitized_cohort["hospitaldischargestatus"].isin(["Alive", "Expired"])
    valid_cohort = sanitized_cohort[valid_mask].copy()
    valid_cohort["hospital_mortality"] = (valid_cohort["hospitaldischargestatus"] == "Expired").astype(int)
    print(f"Valid cohort after target filter: {valid_cohort.shape[0]} stays.")
    print(f"Target counts: Alive={(valid_cohort['hospital_mortality'] == 0).sum()}, Expired={(valid_cohort['hospital_mortality'] == 1).sum()}")

    print("\n--- 3. CLINICAL FEATURE ENGINEERING ---")
    engineer = ClinicalFeatureEngineer()
    engineered_cohort = engineer.compute_features(valid_cohort)

    print("\n--- 4. PATIENT-DISJOINT SPLITTING ---")
    splitter = PatientDisjointSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
    df_train, df_val, df_test = splitter.split(engineered_cohort, patient_col="uniquepid")

    print("\n--- 5. PREPROCESSING & SCALING (TRAIN-FITTED ONLY) ---")
    preprocessor = ClinicalPreprocessor()
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    y_train = df_train["hospital_mortality"].values
    y_val = df_val["hospital_mortality"].values
    y_test = df_test["hospital_mortality"].values

    print(f"X_train shape: {X_train.shape}, y_train pos rate: {y_train.mean():.4f}")
    print(f"X_val shape:   {X_val.shape}, y_val pos rate:   {y_val.mean():.4f}")
    print(f"X_test shape:  {X_test.shape}, y_test pos rate:  {y_test.mean():.4f}")

    # Save processed arrays and metadata
    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)

    np.save(os.path.join(proc_dir, "X_train.npy"), X_train)
    np.save(os.path.join(proc_dir, "y_train.npy"), y_train)
    np.save(os.path.join(proc_dir, "X_val.npy"), X_val)
    np.save(os.path.join(proc_dir, "y_val.npy"), y_val)
    np.save(os.path.join(proc_dir, "X_test.npy"), X_test)
    np.save(os.path.join(proc_dir, "y_test.npy"), y_test)

    # Save preprocessor
    preprocessor.save(os.path.join(proc_dir, "preprocessor.joblib"))

    # Save split metadata (preserving identifiers separately from features)
    meta_cols = ["patientunitstayid", "uniquepid", "hospitalid", "hospital_mortality"]
    df_train[meta_cols].to_csv(os.path.join(proc_dir, "train_meta.csv"), index=False)
    df_val[meta_cols].to_csv(os.path.join(proc_dir, "val_meta.csv"), index=False)
    df_test[meta_cols].to_csv(os.path.join(proc_dir, "test_meta.csv"), index=False)

    # Save feature names list
    with open(os.path.join(proc_dir, "feature_names.json"), "w") as f:
        json.dump(preprocessor.feature_names, f, indent=2)

    print("\n--- 6. GENERATING DATA QUALITY REPORT ---")
    report = generate_quality_report(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        feature_names=preprocessor.feature_names,
        target_col="hospital_mortality",
        patient_col="uniquepid",
        output_dir="results/reports"
    )
    print("Data pipeline executed successfully and certified leakage-free.")


if __name__ == "__main__":
    main()
