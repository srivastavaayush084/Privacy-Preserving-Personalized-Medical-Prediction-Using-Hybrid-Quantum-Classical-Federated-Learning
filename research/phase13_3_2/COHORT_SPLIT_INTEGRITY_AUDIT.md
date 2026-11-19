# Clinical Cohort and Split Integrity Audit

**Phase:** 13.3.2  
**Task:** Final Cohort Integrity, Test-Set Reproducibility, and Clinical Metric Audit  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Healthcare AI Specialist  

---

## 1. Executive Summary

This audit independently establishes the mathematical and clinical integrity of the dataset used across Phases 13.1, 13.2, 13.3, and 13.3.1 of the project *Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning*.

The investigation confirms that:
1. The primary cohort consists of **1,607 ICU stays** across **1,317 unique patients** with **141 total in-hospital deaths** (overall mortality prevalence: 8.7741%).
2. The split is strictly **patient-disjoint**: zero patient overlap exists between training, validation, and test sets ($921 + 198 + 198 = 1,317$).
3. The feature matrix contains **41 clinical variables** observed strictly during the first 24 hours ($\le 1,440$ minutes) of ICU stay, with **zero missing values (NaNs)**, **zero infinite values (Infs)**, and **zero post-landmark leakage**.
4. Feature imputers and scalers were fit **exclusively on the training split** and applied out-of-sample to validation and test sets.

---

## 2. Cohort Definition and Landmark Specification

### 2.1 Clinical Prediction Task
The clinical task is defined as:
> *"Prediction of subsequent in-hospital mortality using clinical observations available at the 24-hour ICU admission landmark ($t = 1,440$ minutes)."*

### 2.2 Inclusion & Exclusion Criteria
- **ICU Stay Duration:** Encounter length of stay $\ge 1,440$ minutes (24.0 hours). Stays shorter than 24 hours are excluded to prevent immortal time bias and ensure landmark observation.
- **Landmark Survival:** Patients who died prior to or at $t = 1,440$ minutes are excluded. The prediction is made strictly for *subsequent* in-hospital mortality.
- **Valid Outcome:** Recorded hospital discharge status (`ALIVE` or `EXPIRED`).
- **Observation Window:** Predictor features are derived solely from clinical data logged in the interval $[0, 1440]$ minutes after ICU admission.

---

## 3. Split Distribution and Patient Disjointness

| Split Partition | Encounter Count ($N$) | Unique Patients | Deaths ($y=1$) | Survivors ($y=0$) | Mortality Prevalence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 1,132 | 921 | 89 | 1,043 | 7.8622% |
| **Validation Split** | 234 | 198 | 24 | 210 | 10.2564% |
| **Test Split** | 241 | 198 | 28 | 213 | 11.6183% |
| **Total Cohort** | **1,607** | **1,317** | **141** | **1,466** | **8.7741%** |

### 3.1 Patient Disjointness Audit
Clustered patients with multiple ICU admissions were partitioned at the unique patient level (`uniquepid`):
- $\text{Train Patients} \cap \text{Validation Patients} = \emptyset$ (0 patients)
- $\text{Train Patients} \cap \text{Test Patients} = \emptyset$ (0 patients)
- $\text{Validation Patients} \cap \text{Test Patients} = \emptyset$ (0 patients)
- Total unique patients: $921 + 198 + 198 = 1,317$.

Because clustering exists (e.g., 198 test patients generate 241 encounters), evaluation metrics and statistical tests require patient-clustered bootstrap resampling to account for intra-patient correlation.

---

## 4. Feature Matrix and Data Cleanliness Audit

### 4.1 Feature Dimensionality and Integrity
- **Predictor Dimensions:** $D = 41$ features across vital signs, laboratory measurements, age, and ICU encounter characteristics.
- **NaN Counts:**
  - `X_train.npy`: 0 NaNs
  - `X_val.npy`: 0 NaNs
  - `X_test.npy`: 0 NaNs
- **Infinite Value Counts:**
  - `X_train.npy`: 0 Infs
  - `X_val.npy`: 0 Infs
  - `X_test.npy`: 0 Infs

### 4.2 Preprocessing Provenance
Preprocessing was audited against `src/data/data_pipeline.py` and saved pipeline artifacts:
1. Median imputer was fit strictly on `X_train`.
2. Standard scaler ($\mu, \sigma$) was fit strictly on `X_train`.
3. Validation and test sets were transformed using the pre-fit training pipeline without updating running statistics.
4. Data files in `data/processed_landmark/` (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`, and corresponding metadata CSVs) have timestamps of `2026-11-19 16:50:54` and have remained immutable across all subsequent phases.

---

## 5. Audit Conclusion

The landmark cohort of 1,607 encounters and 1,317 patients meets rigorous epidemiological and machine learning standards:
- Zero data contamination across splits.
- Zero feature leakage from post-landmark observation intervals.
- Ground truth test set composition is verified as exactly **241 encounters with 28 deaths (11.6183% prevalence)**.
