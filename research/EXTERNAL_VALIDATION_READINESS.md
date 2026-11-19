# EXTERNAL VALIDATION READINESS & MULTI-CENTER DEPLOYMENT PROTOCOL (PHASE 12)

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Target External Databases:** 
1. **MIMIC-IV v2.2** (Beth Israel Deaconess Medical Center, Boston, MA; ~73,181 ICU admissions)
2. **eICU Collaborative Research Database v2.0 (Full Database)** (Philips Healthcare / 208 US Hospitals; 200,859 ICU admissions)  
**Harmonization Module:** `src/data/external_dataset_adapter.py`  

---

## 1. Compliance Disclosure & Scientific Transparency

> [!IMPORTANT] Ethical and Legal Constraints on External ICU Databases  
> Both the full **eICU Collaborative Research Database (v2.0)** and the **MIMIC-IV Database (v2.2)** are restricted-access clinical datasets governed by the PhysioNet Credentialed Data Use Agreement (DUA) and HIPAA Privacy Rule (45 CFR § 164.514). They cannot be downloaded, mirrored, or processed without individualized credentialing and institutional review.
>
> In accordance with strict scientific integrity guidelines:
> 1. We disclose transparently that experiments in Phases 1–12 were executed on the locally authorized **eICU Demo Dataset v2.0.1** (2,492 ICU stays across 186 hospital centers).
> 2. We do NOT fabricate synthetic "MIMIC-IV" results or claim unverified external validation.
> 3. Instead, we provide the **complete, production-ready Data Harmonization Adapter and Deployment Protocol** (`src/data/external_dataset_adapter.py`), allowing any credentialed investigator to replicate and validate the entire hybrid quantum federated pipeline on the full clinical repositories.

---

## 2. Standardized Clinical Feature Harmonization Schema

The framework takes first-24h physiological records and predicts **in-hospital mortality** prior to hospital discharge. The schema maps heterogeneous source EHR tables into our standardized 41-feature tabular input vector:

| Harmonized Feature | Clinical Category | Local eICU Demo v2.0.1 | Full eICU Database v2.0 | MIMIC-IV v2.2 ItemID / Column | Unit / Encoding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `age` | Demographic | `patient.age` | `patient.age` (censored at >89) | `patients.anchor_age` | Years (numerical) |
| `heartrate` | Vital Sign | `vitalPeriodic.heartrate` | `vitalPeriodic.heartrate` | `chartevents.itemid = 220045` | Beats per minute |
| `meanbp` | Vital Sign | `vitalPeriodic.systemicmean` | `vitalPeriodic.systemicmean` | `chartevents.itemid in (220052, 220181)` | mmHg (invasive / cuff) |
| `respiratoryrate`| Vital Sign | `vitalPeriodic.respiration` | `vitalPeriodic.respiration` | `chartevents.itemid in (220210, 224690)` | Breaths per minute |
| `temperature` | Vital Sign | `vitalPeriodic.temperature` | `vitalPeriodic.temperature` | `chartevents.itemid in (223762, 223761)` | Degrees Celsius |
| `sodium` | Electrolyte Lab | `lab.labname = 'sodium'` | `lab.labname = 'sodium'` | `labevents.itemid in (50983, 50824)` | mEq/L |
| `creatinine` | Renal Lab | `lab.labname = 'creatinine'` | `lab.labname = 'creatinine'` | `labevents.itemid = 50912` | mg/dL |
| `bun` | Renal Lab | `lab.labname = 'BUN'` | `lab.labname = 'BUN'` | `labevents.itemid = 51006` | mg/dL |
| `glucose` | Metabolic Lab | `lab.labname = 'bedside glucose'`| `lab.labname = 'bedside glucose'`| `labevents.itemid in (50931, 50809)` | mg/dL |
| `wbc` | Hematology Lab | `lab.labname = 'WBC x 1000'` | `lab.labname = 'WBC x 1000'` | `labevents.itemid in (51301, 51300)` | K/µL |
| `hematocrit` | Hematology Lab | `lab.labname = 'Hct'` | `lab.labname = 'Hct'` | `labevents.itemid in (51221, 50810)` | % |
| `gcs_total` | Neurological | `apacheApsVar.gcs` | `apacheApsVar.gcs` | `chartevents.itemid in (220739, 223900, 223901)` | Sum score [3–15] |
| `shock_index` | Derived Index | `heartrate / meanbp` | `heartrate / meanbp` | `heartrate / meanbp` | Numerical ratio |
| `bun_to_creatinine`| Derived Index | `bun / creatinine` | `bun / creatinine` | `bun / creatinine` | Numerical ratio |
| `gender_*` | Demographic | `patient.gender` | `patient.gender` | `patients.gender` | One-hot binary |
| `ethnicity_*` | Demographic | `patient.ethnicity` | `patient.ethnicity` | `admissions.race` | One-hot binary |
| `unittype_*` | ICU Setting | `patient.unittype` | `patient.unittype` | `icustays.first_careunit` | One-hot binary |
| `*_is_missing` | Missingness Ind. | Binary indicator | Binary indicator | Binary indicator | Binary [0, 1] |
| **`hospital_mortality`** | **Primary Endpoint** | `hospitaldischargestatus == 'Expired'` | `hospitaldischargestatus == 'Expired'` | `admissions.hospital_expire_flag == 1` | **Binary Target {0, 1}** |

---

## 3. Strict Clinical Endpoint & Prediction Window Definitions

To guarantee zero clinical data contamination and eliminate ambiguous outcome descriptions:
1. **Prediction Window:**
   - **eICU Database:** Observations occurring strictly between offset $0$ and $1,440$ minutes (the first 24 hours following ICU admission).
   - **MIMIC-IV:** Measurements with `charttime` satisfying $\text{intime} \le \text{charttime} \le \text{intime} + 24\text{ hours}$.
   - Any measurement recorded after hour 24 is strictly excluded from feature computation.
2. **Clinical Endpoint Definition:**
   - **Target:** In-hospital mortality occurring prior to hospital discharge.
   - **Explicit Clarification:** The endpoint is **whether the patient expires during their hospital hospitalization**, based upon the first-24h ICU physiological trajectory. It is NOT mortality occurring within the first 24 hours. Patients who died within the first 24 hours or had an ICU stay shorter than 24 hours are excluded from the cohort as non-evaluable for 24-hour baseline risk modeling.
3. **Leakage Prevention Verification:**
   - Features undergo automated screening via `ExternalDataHarmonizer.validate_dataset_schema()`.
   - Excluded discharge columns: `hospitaldischargestatus`, `unitdischargestatus`, `hospitaldischargetime`, `unitdischargeoffset`, `hospital_expire_flag`, `deathtime`, and post-admission therapeutic interventions.

---

## 4. PhysioNet Credentialing & External Access Protocol

To run external validation on PhysioNet repositories:
1. **Prerequisite Certification:** Complete the CITI Program course: *"Data or Specimens Only Research"* (Record ID required on PhysioNet profile).
2. **Identity Verification:** Submit verifiable institutional affiliation, government ID, and supervisory contact via PhysioNet.
3. **Data Use Agreement (DUA):** Formally sign the PhysioNet DUA agreeing to:
   - Zero attempt to re-identify patients or hospitals.
   - Zero sharing or redistribution of raw patient records.
   - Storing all data on encrypted, access-controlled institutional servers.
4. **Automated Pipeline Invocation:**
   Once raw MIMIC-IV or eICU `.csv.gz` or `.parquet` files are downloaded into `data/external/`, run:
   ```bash
   python -m src.data.external_dataset_adapter --dataset mimic-iv --input data/external/mimic-iv/ --output data/processed_mimic/
   python experiments/federated/run_phase12_federated.py --data_dir data/processed_mimic/
   ```
"""

    report_path = "research/EXTERNAL_VALIDATION_READINESS.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Generated external validation readiness report at {report_path}")


if __name__ == "__main__":
    generate_external_readiness_report()
