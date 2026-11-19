"""
Unit and regression tests for Phase 13.1 Landmark Cohort.
Verifies:
1. Strict landmark eligibility (unitdischargeoffset >= 1440).
2. Elimination of retrospective outcome leakage (zero deaths before hour 24).
3. Exact accounting of all 885 early stays and 54 early deaths.
4. Strict patient-disjoint train/val/test splits on uniquepid.
5. Feature temporal validity and zero target leakage.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest


RESULTS_DIR = "results/phase13_1"
DATA_DIR = "data/processed_landmark"


def test_cohort_flow_table_exists():
    flow_path = os.path.join(RESULTS_DIR, "cohort_flow.csv")
    assert os.path.exists(flow_path), f"Cohort flow CSV missing at {flow_path}"
    df_flow = pd.read_csv(flow_path)
    assert len(df_flow) >= 5, "Cohort flow must have at least 5 audit stages"


def test_landmark_cohort_size_and_eligibility():
    flow_path = os.path.join(RESULTS_DIR, "cohort_flow.csv")
    df_flow = pd.read_csv(flow_path)
    
    # Check raw encounters
    raw_row = df_flow[df_flow["Stage"].str.contains("Raw eICU", case=False)].iloc[0]
    assert raw_row["Stays_Count"] == 2520, f"Expected 2520 raw stays, got {raw_row['Stays_Count']}"
    
    # Check missing outcome exclusions
    missing_row = df_flow[df_flow["Stage"].str.contains("Missing Hospital Discharge Status", case=False)].iloc[0]
    assert missing_row["Stays_Count"] == 28, f"Expected 28 missing outcome stays, got {missing_row['Stays_Count']}"
    
    # Check early ICU exclusions (< 1440 min)
    early_row = df_flow[df_flow["Stage"].str.contains("ICU Observation < 24 Hours", case=False)].iloc[0]
    assert early_row["Stays_Count"] == 885, f"Expected 885 early stays, got {early_row['Stays_Count']}"
    
    # Check final landmark cohort
    landmark_row = df_flow[df_flow["Stage"].str.contains("Primary 24-Hour Landmark", case=False)].iloc[0]
    assert landmark_row["Stays_Count"] == 1607, f"Expected 1607 landmark stays, got {landmark_row['Stays_Count']}"
    assert landmark_row["Deaths"] == 141, f"Expected 141 landmark deaths, got {landmark_row['Deaths']}"
    assert landmark_row["Survivors"] == 1466, f"Expected 1466 landmark survivors, got {landmark_row['Survivors']}"


def test_early_deaths_breakdown():
    flow_path = os.path.join(RESULTS_DIR, "cohort_flow.csv")
    df_flow = pd.read_csv(flow_path)
    
    # Category 4a: Died in ICU before 24h
    row_4a = df_flow[df_flow["Stage"].str.contains("Died in ICU Before 24h", case=False)].iloc[0]
    assert row_4a["Stays_Count"] == 46, f"Expected 46 ICU deaths before 24h, got {row_4a['Stays_Count']}"
    
    # Category 4b: Died in hospital before 24h post-ICU transfer
    row_4b = df_flow[df_flow["Stage"].str.contains("Died in Hospital Before 24h Post-ICU", case=False)].iloc[0]
    assert row_4b["Stays_Count"] == 8, f"Expected 8 hospital deaths before 24h post-transfer, got {row_4b['Stays_Count']}"
    
    # Total deaths before 24h = 46 + 8 = 54
    assert row_4a["Stays_Count"] + row_4b["Stays_Count"] == 54, "Total deaths before 24h landmark must be exactly 54"
    
    # Category 4c: Died in hospital >= 24h after early ICU transfer
    row_4c = df_flow[df_flow["Stage"].str.contains("Died in Hospital >= 24h After Early", case=False)].iloc[0]
    assert row_4c["Stays_Count"] == 17, f"Expected 17 hospital deaths after early transfer, got {row_4c['Stays_Count']}"
    
    # Category 4d: Discharged alive before 24h
    row_4d = df_flow[df_flow["Stage"].str.contains("Discharged Alive from ICU Before 24h", case=False)].iloc[0]
    assert row_4d["Stays_Count"] == 814, f"Expected 814 alive early discharges, got {row_4d['Stays_Count']}"
    
    # Sum of early subcategories must equal 885
    assert row_4a["Stays_Count"] + row_4b["Stays_Count"] + row_4c["Stays_Count"] + row_4d["Stays_Count"] == 885


def test_split_integrity_and_patient_disjointness():
    split_path = os.path.join(RESULTS_DIR, "split_integrity.csv")
    assert os.path.exists(split_path), f"Split integrity CSV missing at {split_path}"
    df_split = pd.read_csv(split_path)
    
    total_row = df_split[df_split["Partition"] == "Total"].iloc[0]
    assert total_row["Encounter_Count"] == 1607, f"Expected 1607 total encounters, got {total_row['Encounter_Count']}"
    assert total_row["Deaths"] == 141, f"Expected 141 total deaths, got {total_row['Deaths']}"
    
    # Load metadata splits to verify patient disjointness directly
    tr_meta = pd.read_csv(os.path.join(DATA_DIR, "train_meta.csv"))
    va_meta = pd.read_csv(os.path.join(DATA_DIR, "val_meta.csv"))
    te_meta = pd.read_csv(os.path.join(DATA_DIR, "test_meta.csv"))
    
    tr_pts = set(tr_meta["uniquepid"].dropna())
    va_pts = set(va_meta["uniquepid"].dropna())
    te_pts = set(te_meta["uniquepid"].dropna())
    
    assert len(tr_pts.intersection(va_pts)) == 0, "Patient leakage detected between Train and Val!"
    assert len(tr_pts.intersection(te_pts)) == 0, "Patient leakage detected between Train and Test!"
    assert len(va_pts.intersection(te_pts)) == 0, "Patient leakage detected between Val and Test!"
    
    # Verify sum of splits
    assert len(tr_meta) + len(va_meta) + len(te_meta) == 1607


def test_numpy_matrix_shapes_and_finite_values():
    X_tr = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_tr = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_va = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_va = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_te = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_te = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    assert X_tr.shape[0] == len(y_tr)
    assert X_va.shape[0] == len(y_va)
    assert X_te.shape[0] == len(y_te)
    
    assert X_tr.shape[1] == X_va.shape[1] == X_te.shape[1] == 41, "Feature dimension must be 41"
    
    # Ensure no NaN or Inf in feature matrices
    assert np.all(np.isfinite(X_tr)), "Non-finite values found in X_train"
    assert np.all(np.isfinite(X_va)), "Non-finite values found in X_val"
    assert np.all(np.isfinite(X_te)), "Non-finite values found in X_test"
    
    # Ensure binary target
    assert set(np.unique(y_tr)).issubset({0, 1})
    assert set(np.unique(y_va)).issubset({0, 1})
    assert set(np.unique(y_te)).issubset({0, 1})


def test_feature_temporal_audit_exists():
    temp_path = os.path.join(RESULTS_DIR, "feature_temporal_audit.csv")
    assert os.path.exists(temp_path), f"Feature temporal audit CSV missing at {temp_path}"
    df_temp = pd.read_csv(temp_path)
    assert not df_temp["Leaking_Post_Landmark"].any(), "Detected feature leaking post-landmark information!"
