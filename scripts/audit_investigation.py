"""
Independent Audit Investigation Script:
Audits:
1. Patient overlap across splits.
2. Hospital consortium distributions and positive counts.
3. Feature leakage inspection.
4. Preprocessing fitting inspection.
"""
import json
import os
import numpy as np
import pandas as pd

train_meta = pd.read_csv("data/processed/train_meta.csv")
val_meta = pd.read_csv("data/processed/val_meta.csv")
test_meta = pd.read_csv("data/processed/test_meta.csv")

train_pids = set(train_meta["uniquepid"])
val_pids = set(val_meta["uniquepid"])
test_pids = set(test_meta["uniquepid"])

print("=== 1. PATIENT OVERLAP AUDIT ===")
print(f"Train patients: {len(train_pids)}")
print(f"Val patients:   {len(val_pids)}")
print(f"Test patients:  {len(test_pids)}")
print(f"Train & Val overlap: {len(train_pids.intersection(val_pids))}")
print(f"Train & Test overlap: {len(train_pids.intersection(test_pids))}")
print(f"Val & Test overlap:   {len(val_pids.intersection(test_pids))}")

print("\n=== 2. HOSPITAL CONSORTIUM AUDIT ===")
print("Unique hospitals in train:", train_meta["hospitalid"].nunique())
for c in range(5):
    tr_sub = train_meta[train_meta["hospitalid"] % 5 == c]
    va_sub = val_meta[val_meta["hospitalid"] % 5 == c]
    te_sub = test_meta[test_meta["hospitalid"] % 5 == c]
    tr_pos = int(tr_sub["hospital_mortality"].sum())
    te_pos = int(te_sub["hospital_mortality"].sum())
    print(f"Consortium {c+1}: Train N={len(tr_sub)} (Pos={tr_pos}, {tr_pos/len(tr_sub):.1%}) | Test N={len(te_sub)} (Pos={te_pos}, {te_pos/len(te_sub):.1%})")

print("\n=== 3. FEATURE LEAKAGE AUDIT ===")
with open("data/processed/feature_names.json") as f:
    features = json.load(f)
print(f"Total features: {len(features)}")
print("Features list:\n", features)

suspicious = [f for f in features if any(w in f.lower() for w in ["discharge", "death", "mortality", "expired", "length", "los", "alive", "status", "apache"])]
print("\nPotentially outcome-related or APACHE feature names:", suspicious)
