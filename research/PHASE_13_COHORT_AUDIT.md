# PHASE 13 — INDEPENDENT CLINICAL COHORT & DATA SPLIT AUDIT REPORT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13 Independent Reproducibility & Claim Verification  
**Auditor Roles:** Senior Healthcare Data Scientist, Independent Clinical ML Auditor  
**Date:** November 19, 2026  
**Cohort Source:** eICU Collaborative Research Database Demo v2.0.1  

---

## 1. Executive Summary & Clinical Endpoint Formulation

An exhaustive, line-by-line audit of the dataset preprocessing pipeline, raw eICU demo tables (`patient.csv.gz`, `apacheApsVar.csv.gz`, `apachePredVar.csv.gz`, `hospital.csv.gz`), and the processed arrays (`data/processed/`) was conducted.

### Canonical Formulation of Clinical Prediction Endpoint
The clinical objective is formally defined as:
> **Prediction of subsequent in-hospital mortality using clinical information available during the first 24 hours of ICU admission.**

### Critical Clinical Distinction:
- **Prediction Target:** `hospitaldischargestatus == 'Expired'` (binary: 1 = Expired prior to hospital discharge; 0 = Discharged alive).
- **Observation Window:** Strictly the **first 24 hours of ICU admission** (physiological vital signs, baseline chemistry laboratories, APACHE acute physiology variables, and admission demographics).
- **Temporal Directionality:** This is **NOT** a prediction of "mortality occurring within the first 24 hours". It is an early prognostic risk stratification tool deployed at hour 24 of ICU admission to forecast terminal hospital mortality.

---

## 2. Exhaustive Cohort Attrition Flow

The table below details the step-by-step attrition from the raw eICU Demo release down to the final partitioned evaluation sets.

| Stage | Attrition Criteria | ICU Stays Remaining | Unique Patients Remaining | Excluded Records | Mortality Events (Expired) | Mortality Prevalence (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Raw eICU Demo v2.0.1** | Unfiltered baseline patient table | 2,520 | 1,841 | 0 | 212 | 8.41% |
| **Target Validity Filter** | Exclude unrecorded / null `hospitaldischargestatus` | 2,492 | 1,820 | 28 stays (21 patients) | 212 | 8.51% |
| **Final Modeled Cohort** | Valid target + 24-hr acute physiology | **2,492** | **1,820** | 0 | **212** | **8.51%** |
| **Partition: Train Set** | Patient-disjoint group split (70%) | 1,712 | 1,274 | — | 142 | 8.29% |
| **Partition: Validation Set**| Patient-disjoint group split (15%) | 387 | 273 | — | 30 | 7.75% |
| **Partition: Test Holdout** | Patient-disjoint group split (15%) | 393 | 273 | — | 40 | 10.18% |

*Audit Artifact Reference:* Detailed machine-readable attrition records are archived in [`results/phase13/cohort_flow.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/cohort_flow.csv).

---

## 3. Data Split Integrity & Patient-Disjoint Verification

To prevent patient identity leakage (where a patient with multiple ICU admissions appears in both training and test partitions), splitting was performed on the unique patient identifier (`uniquepid`).

### Partition Integrity Audit

| Integrity Metric | Train Set | Validation Set | Test Set Holdout | Combined / Total | Verification Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total ICU Unit Stays** | 1,712 | 387 | 393 | 2,492 | **VERIFIED (Strictly Additive)** |
| **Unique Patients (`uniquepid`)** | 1,274 | 273 | 273 | 1,820 | **VERIFIED (Strictly Additive)** |
| **Expired Cases ($y = 1$)** | 142 | 30 | 40 | 212 | **VERIFIED (Strictly Additive)** |
| **Alive Cases ($y = 0$)** | 1,570 | 357 | 353 | 2,280 | **VERIFIED (Strictly Additive)** |
| **Mortality Rate** | 8.29% | 7.75% | 10.18% | 8.51% | **VERIFIED (Statistically Balanced)** |
| **Pairwise Patient Overlap** | Train $\cap$ Val: 0 | Train $\cap$ Test: 0 | Val $\cap$ Test: 0 | **0 Overlap** | **VERIFIED (Zero Patient Leakage)** |
| **Hospitals Represented** | 186 | 146 | 156 | 186 | **VERIFIED (Consortium Coverage)** |

*Audit Artifact Reference:* Partition integrity records are archived in [`results/phase13/split_integrity.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/split_integrity.csv).

---

## 4. Observation Window & Target Leakage Prevention Audit

### Predictor Feature Inventory Audit
The final preprocessed feature representation consists of **41 engineered tabular features**:
1. **Continuous Physiology (First-24h):** `age`, `bmi`, `heartrate`, `meanbp`, `respiratoryrate`, `temperature`, `sodium`, `creatinine`, `bun`, `glucose`, `wbc`, `hematocrit`.
2. **Clinical Interaction Indices:** `shock_index` ($\frac{\text{heartrate}}{\text{meanbp}}$), `bun_to_creatinine` ($\frac{\text{bun}}{\text{creatinine}}$), `gcs_total` ($\text{eyes} + \text{motor} + \text{verbal}$).
3. **Comorbidity & Acuity Aggregates:** `comorbidity_burden` (sum of pre-existing condition flags from `apachePredVar`), `organ_support_count` (intubation, mechanical ventilation, dialysis).
4. **Demographic & Contextual Encodings:** One-hot encoded `gender` (2 categories), `ethnicity` (6 categories), ICU `unittype` (8 categories), hospital `region` (4 categories).
5. **Informed Missingness Indicators:** Binary missingness masks for sporadically measured biomarkers (`ph_is_missing`, `pao2_is_missing`, `bilirubin_is_missing`, `albumin_is_missing`).

### Target Leakage Verification Checklist
- [x] **No Discharge-Time Variables:** Variables such as `hospitaldischargeoffset`, `unitdischargeoffset`, `hospitaldischargelocation`, `unitdischargelocation`, and `dischargeweight` are **strictly excluded** from the feature matrix.
- [x] **No Post-Outcome Variables:** The APACHE IV predicted mortality probability (`apachescore`, `predictedhospitalmortality`) and acute therapy intervention records are **excluded** to prevent proxy circularity.
- [x] **Training-Only Preprocessing Fit:** Preprocessing scalers (median imputation for physiology, one-hot encoders, and standard scaling) were fit strictly on `df_train` and saved to `data/processed/preprocessor.joblib`. Test and validation sets were transformed using the frozen training parameters, eliminating distribution snooping.

---

## 5. Temporal Duration & Early Discharge Audit

An in-depth temporal analysis of `unitdischargeoffset` (ICU length of stay in minutes) in the valid cohort ($N = 2,492$) revealed:
- Stays with ICU length of stay $< 24$ hours ($< 1,440$ min): **885 stays (35.51%)**.
- Among stays with length of stay $< 24$ hours, **71 patients died in the ICU** and 814 were transferred out alive to step-down/general wards before hour 24.
- In eICU Demo v2.0.1, the `apacheApsVar` table provides standardized, worst-case physiology observed during the initial admission period (up to 24 hours, or the duration of stay if shorter).
- Because standard ICU benchmarking protocols retain patients transferred early to lower-acuity wards, this dataset represents a realistic heterogeneous ICU intake cohort.

---

## 6. Cohort Stability Across Phases

- **Phase 11 vs Phase 12 vs Phase 13 Cohort:** The data split files (`X_train.npy`, `X_val.npy`, `X_test.npy`, `train_meta.csv`, `val_meta.csv`, `test_meta.csv`) have remained **100% stable, identical, and immutable** across Phases 11, 12, and 13.
- No records were silently dropped or modified between audit phases.
- The reported difference in some metrics between early iterations and Phase 12 arose from **checkpoint selection policy** (reporting asymptotic Round 10 versus best early-stopping round) and **algorithmic remediation** (replacing heuristic batch-level clipping with formal per-sample DP-SGD), NOT from unannounced alterations to the clinical cohort.
