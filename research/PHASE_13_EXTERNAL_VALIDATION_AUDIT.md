# PHASE 13 — INDEPENDENT EXTERNAL DATASET READINESS AUDIT REPORT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13 Independent Reproducibility & Claim Verification  
**Auditor Roles:** Clinical Data Architect, Healthcare Informatics Auditor  
**Date:** November 19, 2026  
**Implementation Modules Audited:** `src/data/external_dataset_adapter.py`, `research/EXTERNAL_VALIDATION_READINESS.md`  

---

## 1. Executive Summary & External Validation Status

An exhaustive review was performed on the external dataset adaptation framework (`src/data/external_dataset_adapter.py`) designed to interface external electronic health record (EHR) repositories:
1. **MIMIC-IV v2.2** (Beth Israel Deaconess Medical Center, Boston, MA; ~73,181 ICU admissions)
2. **eICU Collaborative Research Database v2.0 (Full)** (Philips Healthcare / 208 US Hospitals; 200,859 ICU admissions)

### High-Level Audit Verdict: **ADAPTER ARCHITECTURE VERIFIED; EXPERIMENTAL STATUS: SINGLE-COHORT STUDY**
- **Readiness Status:** The programmatic schema mapping, unit conversions, derived index calculators, and leakage validation rules in `src/data/external_dataset_adapter.py` are structurally sound and complete.
- **Mandatory Scientific Disclosure:** **External validation has NOT been executed.** Due to PhysioNet Credentialed Data Use Agreements (DUA) and HIPAA restrictions prohibiting unauthorized redistribution of full clinical repositories, all empirical results in Phases 1–13 were generated solely from the publicly accessible `eICU Demo Dataset v2.0.1`.
- **Publication Requirement:** The project must be classified strictly as an **exploratory single-cohort development study with an open-source external harmonization adapter**, NOT an externally validated clinical prediction algorithm.

---

## 2. Cross-Dataset Clinical Feature Mapping Audit

The table below documents the harmonization rules for the core continuous physiology, lab biomarkers, and clinical targets across source EHR schemas.

| Harmonized Feature | Clinical Category | Local eICU Demo v2.0.1 | Full eICU Database v2.0 | MIMIC-IV v2.2 Source & ItemID | Harmonized Unit / Range | Conversion / Normalization Rule |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`age`** | Demographics | `patient.age` | `patient.age` | `patients.anchor_age` | Years [18–100] | String `> 89` in eICU recoded to `90.0`; MIMIC `anchor_age` clamped $\le 90$. |
| **`heartrate`** | Vital Signs | `vitalPeriodic.heartrate` | `vitalPeriodic.heartrate` | `chartevents.itemid = 220045` | Beats / min [20–300] | Outliers $<20$ or $>300$ replaced with NaN for median imputation. |
| **`meanbp`** | Vital Signs | `vitalPeriodic.systemicmean` | `vitalPeriodic.systemicmean` | `itemid in (220052, 220181, 225312)` | mmHg [15–250] | Prioritize invasive arterial line (`220052`); fallback to NIBP (`220181`). |
| **`respiratoryrate`** | Vital Signs | `vitalPeriodic.respiration` | `vitalPeriodic.respiration` | `itemid in (220210, 224690)` | Breaths / min [2–90] | Filter ventilator harmonics; clamp to clinical bounds $[2, 90]$. |
| **`temperature`** | Vital Signs | `vitalPeriodic.temperature` | `vitalPeriodic.temperature` | `itemid in (223762, 223761)` | Celsius [26.0–45.0] | Convert Fahrenheit itemid `223761` to Celsius via: $T_C = (T_F - 32) \times \frac{5}{9}$. |
| **`sodium`** | Chemistry Lab | `lab.labname = 'sodium'` | `lab.labname = 'sodium'` | `labevents.itemid in (50983, 50824)` | mEq/L [90–190] | Both serum chemistry (`50983`) and whole-blood gas (`50824`) supported. |
| **`creatinine`** | Chemistry Lab | `lab.labname = 'creatinine'` | `lab.labname = 'creatinine'` | `labevents.itemid = 50912` | mg/dL [0.1–35.0] | Serum creatinine; bounds enforcement $[0.1, 35.0]$. |
| **`bun`** | Chemistry Lab | `lab.labname = 'BUN'` | `lab.labname = 'BUN'` | `labevents.itemid = 51006` | mg/dL [1.0–250.0] | Blood urea nitrogen; bounds enforcement $[1.0, 250.0]$. |
| **`glucose`** | Chemistry Lab | `lab.labname = 'bedside glucose'` | `lab.labname = 'bedside glucose'` | `labevents.itemid in (50931, 50809)` | mg/dL [10–1500] | Serum or fingerstick blood glucose; bounds $[10, 1500]$. |
| **`wbc`** | Hematology Lab| `lab.labname = 'WBC x 1000'` | `lab.labname = 'WBC x 1000'` | `labevents.itemid in (51301, 51300)` | K/$\mu$L [0.1–200.0] | White blood cell count; bounds $[0.1, 200.0]$. |
| **`hematocrit`** | Hematology Lab| `lab.labname = 'Hct'` | `lab.labname = 'Hct'` | `labevents.itemid in (51221, 50810)` | % [5.0–75.0] | Blood hematocrit percentage; bounds $[5.0, 75.0]$. |
| **`gcs_total`** | Neurological | `apacheApsVar.gcs` | `apacheApsVar.gcs` | `itemid in (220739, 223900, 223901)` | Score [3–15] | Sum of GCS Eye (`220739`), Motor (`223901`), and Verbal (`223900`). |
| **`shock_index`** | Derived Index | `heartrate / meanbp` | `heartrate / meanbp` | `heartrate / meanbp` | Ratio [0.1–5.0] | Clipped mean BP $\ge 20$ mmHg to prevent division by zero. |
| **`bun_to_creatinine`** | Derived Index | `bun / creatinine` | `bun / creatinine` | `bun / creatinine` | Ratio [1.0–200.0] | Clipped creatinine $\ge 0.1$ mg/dL to prevent division by zero. |
| **`hospital_mortality`** | **Primary Target**| `hospitaldischargestatus == 'Expired'` | `hospitaldischargestatus == 'Expired'` | `admissions.hospital_expire_flag == 1` | Binary {0, 1} | Positive: Death prior to hospital discharge; Negative: Discharged alive. |

---

## 3. Discrepancies and Incompatible Types Identified

During schema auditing, four systematic domain discrepancies were identified that require explicit handling during external deployment:

1. **Age Censoring Discrepancy:**
   - In eICU, ages above 89 are recorded as string `"> 89"`.
   - In MIMIC-IV, patient ages $\ge 89$ are shifted by anchor years into an `anchor_age = 91` grouping.
   - *Harmonization rule:* All ages $\ge 89$ or marked `> 89` are standardized to numerical `90.0`.
2. **Temperature Scale Units:**
   - eICU stores temperature strictly in degrees Celsius.
   - MIMIC-IV `chartevents` contains both Celsius (`itemid = 223762`) and Fahrenheit (`itemid = 223761`).
   - *Harmonization rule:* Automated unit conversion is enforced in `external_dataset_adapter.py`.
3. **Neurological Coma Scoring (GCS):**
   - In eICU, `apacheApsVar` contains pre-computed worst-24h GCS sub-scores (`eyes`, `motor`, `verbal`).
   - In MIMIC-IV, GCS components are logged as event timestamps across `chartevents`. Worst-24h minimum values must be aggregated temporally over the first 24 hours of each ICU stay.
4. **Hospital Administrative Metadata:**
   - eICU provides multi-hospital identifiers (`hospitalid`, `region`, `teachingstatus`, `numbedscategory`).
   - MIMIC-IV originates from a single academic medical center (Beth Israel Deaconess Medical Center, Boston, MA). Inter-hospital federated partitioning cannot be evaluated naturally within MIMIC-IV without synthetic client clustering (e.g. by ICU care unit: MICU, SICU, CCU, TSICU).

---

## 4. Scientific Conclusion & Publication Mandate

1. The external dataset harmonization pipeline is fully coded, unit-tested, and scientifically verified for schema compliance.
2. Because credentialed external datasets cannot be redistributed within local research workspaces, this study provides **demonstration on eICU Demo v2.0.1 accompanied by an open-access reproducible harmonization adapter**.
3. All publications must explicitly disclose that external generalizability remains a future research milestone, preventing overstated claims of cross-center deployment.
