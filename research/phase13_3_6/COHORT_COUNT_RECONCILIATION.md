# COHORT COUNT & PATIENT-DISJOINT PARTITION RECONCILIATION

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase Identifier:** Phase 13.3.6 — Final Cohort, Prediction Artifact and Metric Reconciliation Audit  
**Auditor Role:** Senior Biostatistician, Clinical Machine-Learning Auditor, and Reproducibility Specialist  
**Date:** November 19, 2026  
**Artifact Path:** `research/phase13_3_6/COHORT_COUNT_RECONCILIATION.md`  

---

## 1. Executive Summary & Objective

This document delivers an exhaustive, source-level audit and reconciliation of patient and encounter counts across all splits of the clinical 24-hour landmark ICU mortality cohort.

Specifically, it resolves the question raised regarding:
- Earlier cohort specification: **234 validation encounters and 198 unique patients**.
- Phase 13.3.5 narrative reference: **234 validation encounters and 192 unique patients**.

---

## 2. Source Data File Inspection & Ground Truth Counts

All data files located in `data/processed_landmark/` were audited directly using automated Python inspection scripts without modifying or regenerating any dataset files.

### 2.1 File-by-File Integrity and Hashes

| Partition Split | Metadata File Path | SHA-256 Checksum | Total Encounters | Unique Patients (`uniquepid`) | Deaths ($Y=1$) | Survivors ($Y=0$) | Prevalence |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | `data/processed_landmark/train_meta.csv` | `0d6dd23c19d5fb89afe8b69a140d4a941f75868ca253386939a01feb6088b41b` | 1,132 | 921 | 89 | 1,043 | 7.8622% |
| **Validation Split** | `data/processed_landmark/val_meta.csv` | `7604f7b1d9e117d55a1037ab2f7a9c4c895df7850cf93f404228b3352c5831fe` | 234 | **198** | 24 | 210 | 10.2564% |
| **Test Split** | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | 241 | **198** | 28 | 213 | 11.6183% |
| **Total Cohort** | Consolidated Landmark | N/A (Additive Union) | **1,607** | **1,317** | **141** | **1,466** | **8.7741%** |

### 2.2 Encounters-per-Patient Distribution in Validation Cohort (`val_meta.csv`)

Direct enumeration of encounters per unique patient in `data/processed_landmark/val_meta.csv`:
- Patients with 1 encounter: **171 patients** $\times 1 = 171$ encounters
- Patients with 2 encounters: **20 patients** $\times 2 = 40$ encounters
- Patients with 3 encounters: **5 patients** $\times 3 = 15$ encounters
- Patients with 4 encounters: **2 patients** $\times 4 = 8$ encounters
- **Total Unique Patients:** $171 + 20 + 5 + 2 = \mathbf{198}$ patients.
- **Total Encounters:** $171 + 40 + 15 + 8 = \mathbf{234}$ encounters.
- **Null / Missing Value Count:** Exactly 0 in `patientunitstayid`, `uniquepid`, `hospitalid`, and `hospital_mortality`.

---

## 3. Investigation of the "192" vs "198" Discrepancy

### 3.1 Trace Analysis
1. **Source Code Check:** `experiments/phase13_3_4/run_phase13_3_4_audit.py` was inspected at line 188 and lines 413–415:
   ```python
   df_val_meta = pd.read_csv("data/processed_landmark/val_meta.csv")
   patient_ids_val = df_val_meta["uniquepid"].values
   val_unique_patients = np.unique(patient_ids_val)
   n_val_unique_patients = len(val_unique_patients)
   ```
   Direct execution yields `len(val_unique_patients) == 198`.
2. **Phase 13.3.5 Audit File Check:** In `research/phase13_3_5/VALIDATION_RESAMPLING_VERIFICATION.md` (line 13), the audit report correctly states:
   > `($N_{\text{val}} = 234$ encounters, $24$ deaths, $198$ unique patients)`
3. **Root Cause Identification:** The number "192" appeared solely in the conversational summary narrative of the Phase 13.3.5 response text and the updated notes in `PROJECT_STATUS.md`. It was a typographical transcription error (typing 192 instead of 198).
4. **Patient Identification Comparison:**
   - Are there 6 missing patients? **No.**
   - Was there filtering or exclusion of 6 patients? **No.**
   - Did resampling drop any patients? **No.**
   - All manifests from Phase 13.1, Phase 13.2, Phase 13.3, Phase 13.3.1, Phase 13.3.2, Phase 13.3.3, Phase 13.3.4, and the raw CSV `data/processed_landmark/val_meta.csv` record exactly 198 unique patients.

---

## 4. Verification of Zero Patient Overlap (Strict Disjointness)

Pairwise intersections of unique patient identifiers (`uniquepid`) were computed across all partitions:

$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} = \emptyset \quad (|921 \cap 198| = 0)$$
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset \quad (|921 \cap 198| = 0)$$
$$\mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} = \emptyset \quad (|198 \cap 198| = 0)$$

$$\text{Total Additive Patient Count} = 921 + 198 + 198 = \mathbf{1,317} \quad (\text{Strictly 0.00% Patient Leakage})$$

---

## 5. Patient Count Reconciliation Conclusion

The frozen landmark cohort is mathematically sound, fully additive, and verified across all source files:
- Validation encounters: **234**
- Validation unique patients: **198** (the narrative reference to 192 was an isolated typographical error and is formally corrected)
- Zero patient overlap across splits.
