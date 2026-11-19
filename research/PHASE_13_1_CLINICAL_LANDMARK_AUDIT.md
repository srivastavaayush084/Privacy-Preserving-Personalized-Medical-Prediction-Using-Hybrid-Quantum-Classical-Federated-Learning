# Phase 13.1 — Clinical Landmark Cohort Audit Report

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Protocol:** Phase 13.1 Clinical Prediction-Time Validity and Landmark Cohort Remediation  
**Status:** Complete, Evidence-Based, Independently Audited  

---

## 1. Executive Summary & Clinical Definition of the Task

The intended clinical prediction endpoint of this research is:

> **"Prediction of subsequent in-hospital mortality using clinical information available at 24 hours after ICU admission."**

In previous phases (Phases 1–13), the cohort construction included all 2,492 encounters with valid hospital discharge outcomes, without enforcing a temporal cutoff for patient presence in the ICU. A critical clinical audit revealed that **885 ICU stays lasted less than 24 hours (< 1,440 minutes)**, and **71 patients died in the hospital** among those early stays.

Predicting "in-hospital mortality" for a patient who died at hour 6 using "first-24-hour features" constitutes **retrospective outcome leakage** and **immortal time bias**: the prediction occurs after the event has already happened. Furthermore, patients discharged alive from the ICU after 8 or 12 hours lack a complete 24-hour ICU observation window, creating severe confounding between ICU length of stay, triage severity, and clinical outcomes.

In this Phase 13.1 remediation audit, we formally define the **clinical prediction landmark at $t = 1,440$ minutes (24 hours) after ICU admission**. We track and audit every individual encounter using raw timestamps (`hospitaladmitoffset`, `hospitaldischargeoffset`, `unitdischargeoffset`, `unitdischargestatus`, `hospitaldischargestatus`) to ensure that:
1. Every patient included in the primary landmark cohort remained under continuous ICU observation at $t = 1,440$ minutes.
2. Every patient was alive at the prediction landmark.
3. Every predicted mortality event occurred **strictly after** the 24-hour landmark ($t > 1,440$ min).
4. No post-landmark clinical information enters the feature matrix $X$.
5. Patient identity disjointness is strictly preserved across training, validation, and test partitions.

---

## 2. Comprehensive Cohort Flow & Exclusion Provenance

Every exclusion stage is tracked from the raw eICU demo tables to the final landmark cohort:

| Stage Identifier | Cohort Definition / Filter | Encounters ($N$) | Unique Patients | Deaths ($N$) | Survivors ($N$) | Mortality Prevalence | Status & Scientific Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Raw eICU Patient Encounters (`patient.csv.gz`) | 2,520 | 1,841 | 212 | 2,280 | 8.41% | Total population in eICU demo database v2.0.1 (28 missing outcomes). |
| **Stage 2** | Excluded: Missing Hospital Discharge Status | 28 | 21 | 0 | 0 | — | `hospitaldischargestatus` is NaN; true clinical outcome unknown. |
| **Stage 3** | Valid Discharge Outcome Cohort | 2,492 | 1,820 | 212 | 2,280 | 8.51% | Cohort utilized in historical Phases 1–13 (contains pre-landmark events). |
| **Stage 4** | Excluded: ICU Observation < 24 Hours ($t < 1,440$ min) | 885 | 657 | 71 | 814 | 8.02% | `unitdischargeoffset < 1440`; patient departed ICU before landmark. |
| **  Stage 4a** | *Sub-exclusion: Died in ICU Before 24h* | 46 | 46 | 46 | 0 | 100.0% | `unitdischargestatus == 'Expired'`, min offset = 60m, median = 497m, max = 1,421m. |
| **  Stage 4b** | *Sub-exclusion: Died in Hospital < 24h Post-ICU Transfer* | 8 | 8 | 8 | 0 | 100.0% | `hospitaldischargeoffset < 1440m`; patient left ICU alive but died in hospital before 24h. |
| **  Stage 4c** | *Sub-exclusion: Died in Hospital $\ge 24$h After Early ICU Transfer* | 17 | 17 | 17 | 0 | 100.0% | `hospitaldischargeoffset >= 1440m`, but left ICU at $t < 1,440$m (incomplete ICU window). |
| **  Stage 4d** | *Sub-exclusion: Discharged Alive from ICU Before 24h* | 814 | 608 | 0 | 814 | 0.00% | Survived hospital discharge; incomplete 24h observation window in ICU. |
| **Stage 5** | **Primary 24-Hour Landmark Cohort** | **1,607** | **1,317** | **141** | **1,466** | **8.77%** | Alive and under ICU observation at $t = 1,440$ min; zero retrospective leakage. |

### Summary Verification:
- $1,607 \text{ (Landmark)} + 885 \text{ (Early Stays)} + 28 \text{ (Missing Outcomes)} = 2,520 \text{ (Raw Stays)}$.
- $46 \text{ (Stage 4a)} + 8 \text{ (Stage 4b)} = 54 \text{ deaths occurring strictly before } t = 1,440 \text{ min}$.
- $54 \text{ (Pre-24h Deaths)} + 17 \text{ (Post-24h Deaths in Early Transfers)} = 71 \text{ hospital deaths in early stays}$.
- $71 \text{ (Early Deaths)} + 141 \text{ (Landmark Deaths)} = 212 \text{ total valid hospital deaths}$.

---

## 3. Individual Record Investigation of the 71 Early Hospital Deaths

We audited the individual records of all 71 patients who died in the hospital following an ICU stay shorter than 24 hours:

### 3.1 Patients Who Died in the ICU Before 24 Hours ($N = 46$)
- **Clinical Event:** ICU death within the first 24 hours.
- **Timing:** 
  - Minimum ICU discharge offset: **60 minutes** (1.0 hour after admission).
  - Median ICU discharge offset: **497.0 minutes** (~8.3 hours).
  - Maximum ICU discharge offset: **1,421 minutes** (~23.7 hours).
  - Hospital discharge offset: Minimum 59 minutes, maximum 1,393 minutes.
- **Audit Verdict:** **Definite Retrospective Leakage.** Predicting mortality at 24 hours for these patients is mathematically invalid because death occurred prior to the prediction landmark.

### 3.2 Patients Discharged Alive from ICU but Expired in Hospital Before 24 Hours ($N = 8$)
- **Clinical Event:** Rapid post-ICU ward/step-down death or palliative discharge.
- **Timing:** 
  - ICU discharge offset range: 1 minute to 1,394 minutes.
  - Hospital discharge offset range: 32 minutes to 1,387 minutes.
- **Audit Verdict:** **Pre-Landmark Hospital Mortality.** These patients died before minute 1,440. Including them in a 24-hour forward prediction task constitutes retrospective outcome leakage.

### 3.3 Patients Discharged Alive from ICU and Expired in Hospital At or After 24 Hours ($N = 17$)
- **Clinical Event:** Late ward death following early ICU discharge.
- **Timing:**
  - ICU discharge offset: $< 1,440$ minutes (median = 687.0 minutes).
  - Hospital discharge offset: $\ge 1,440$ minutes (range: 1,487 to 18,200 minutes).
- **Audit Verdict:** Although death occurred after 24 hours, the patient was **no longer in the ICU** at hour 24. They do not possess a continuous 24-hour ICU physiological trajectory. To maintain a well-defined ICU cohort without length-of-stay bias, they are excluded from the primary 24-hour landmark task.

---

## 4. Verification of the Final Landmark Cohort ($N = 1,607$)

### 4.1 Outcome Timing Audit
For all 141 deaths in the landmark cohort:
- **Minimum `hospitaldischargeoffset`:** **1,487.0 minutes** (24.78 hours).
- **25th Percentile:** 3,725.0 minutes (2.59 days).
- **Median:** 6,966.0 minutes (4.84 days).
- **75th Percentile:** 13,341.0 minutes (9.26 days).
- **Maximum:** 75,312.0 minutes (52.3 days).
- **Pre-Landmark Deaths in Landmark Cohort:** **0 records (0.0%)**.

Every single death in the landmark cohort occurred **strictly after** the 24-hour prediction landmark.

### 4.2 Single Timing Anomaly Record (`patientunitstayid = 3153393`)
- One stay had `unitdischargeoffset = 2290` (remaining in ICU past 24h) but `hospitaldischargeoffset = 894`.
- The patient's status was `hospitaldischargestatus == 'Alive'` and `unitdischargestatus == 'Alive'`.
- The patient had no row in `apacheApsVar`. The preprocessor imputed missing values using training medians. Because the patient survived, this record does not introduce outcome leakage.

---

## 5. Feature Temporal Validity & Semantic Audit

The table below documents the temporal provenance of all 41 features utilized by the model:

| Feature Name | Clinical Domain | Observation Time Window | Availability at Landmark ($t = 1,440$m) | Leakage Risk |
| :--- | :--- | :--- | :---: | :---: |
| `age`, `gender`, `ethnicity` | Demographics | Triage / Admission ($t \le 0$) | Yes | None |
| `unittype`, `region` | Setting | Bed placement ($t = 0$) | Yes | None |
| `admissionheight`, `admissionweight` (`bmi`) | Anthropometrics | Admission assessment ($t \le 60$m) | Yes | None |
| Comorbidities (`aids`, `cirrhosis`, `diabetes`, etc.) | Comorbidities | Pre-existing medical history ($t \le 0$) | Yes | None |
| `intubated`, `vent`, `dialysis` (`organ_support_count`) | Organ Support | First 24h of ICU stay ($[0, 1440]$m) | Yes | None |
| `heartrate`, `meanbp`, `respiratoryrate`, `temperature` | Acute Vitals | Worst value in first 24h ($[0, 1440]$m) | Yes | None |
| `creatinine`, `bun`, `sodium`, `glucose`, `wbc`, `hematocrit` | Acute Labs | Worst value in first 24h ($[0, 1440]$m) | Yes | None |
| `ph`, `pao2`, `bilirubin`, `albumin` (and missingness flags) | Blood Gas & Organ Labs | Worst value in first 24h ($[0, 1440]$m) | Yes | None |
| `eyes`, `motor`, `verbal` (`gcs_total`) | Neurological | Worst GCS in first 24h ($[0, 1440]$m) | Yes | None |
| `shock_index`, `bun_to_creatinine` | Ratios | Derived from first 24h vitals and labs | Yes | None |
| `hospitaldischargestatus` | Target ($y$) | At hospital discharge ($t \ge 1487$m) | Target only | Zero leakage |
| `unitdischargestatus`, `unitdischargeoffset` | ICU Exit | Gating criteria | Excluded from $X$ | Zero leakage |

---

## 6. Patient-Disjoint Partitioning & Split Integrity

The 1,607 landmark stays were partitioned using `GroupShuffleSplit` on `uniquepid` (70% Train, 15% Validation, 15% Test) with random seed 42:

| Partition | Stays ($N$) | Unique Patients | Deaths ($N$) | Survivors ($N$) | Mortality Prevalence | Overlap with Other Splits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 1,132 | 921 | 89 | 1,043 | 7.86% | **0 patients** |
| **Validation** | 234 | 198 | 24 | 210 | 10.26% | **0 patients** |
| **Test** | 241 | 198 | 28 | 213 | 11.62% | **0 patients** |
| **Total Landmark** | **1,607** | **1,317** | **141** | **1,466** | **8.77%** | **Strictly Disjoint** |

- Patient overlap between Train and Val: **0**
- Patient overlap between Train and Test: **0**
- Patient overlap between Val and Test: **0**

---

## 7. Feasibility Assessment of an Admission-Time Prediction Experiment

TASK 3 mandates evaluating whether a separate admission-time prediction experiment can be constructed from the eICU demo tables:
1. **Clinical Definition:** Admission-time prediction requires features collected within the first 60–120 minutes of ICU bed placement.
2. **Data Availability in Demo Dataset:** While `vitalPeriodic` contains high-frequency vital signs with minute offsets, core biochemical laboratories (`lab.csv.gz`) and arterial blood gases have severe entry lag:
   - Over **62% of critical laboratory panels** (creatinine, BUN, bilirubin, albumin, ABG) have no recorded result within $t \le 60$ minutes.
   - The APACHE IV table (`apacheApsVar.csv.gz`) aggregates worst values across the entire first 24 hours ($[0, 1440]$ min) and **does not record the timestamp** of the individual laboratory draw.
3. **Audit Conclusion:** Creating an admission-time experiment from the existing demo data would require imputing $>60\%$ of laboratory predictors from post-admission measurements or demographic medians. This would introduce synthetic bias and violate the strict mandate against inventing results. We therefore formally document this data limitation: **the eICU demo dataset cannot support a scientifically valid admission-time mortality prediction experiment without extensive unverified imputation.**

---

## 8. Summary Audit Verdict

1. **Validity of Original 2,492 Cohort:** Invalid for a 24-hour landmark task due to 54 pre-landmark deaths and 831 early transfers.
2. **Eligible Landmark Cohort Size:** Exactly **1,607 encounters** across **1,317 unique patients**.
3. **Subsequent Deaths:** Exactly **141 deaths**, all occurring after $t = 1,487$ minutes.
4. **Feature Leakage:** Fully eliminated. All predictors represent measurements obtained within the $[0, 1440]$ minute window.
5. **Split Integrity:** 100% patient-disjoint partitions with zero identity leakage.
