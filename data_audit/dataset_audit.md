# Comprehensive Dataset Audit: eICU Demo v2.0.1

## Executive Summary
This audit inspects the eICU Collaborative Research Database Demo version 2.0.1 located in the local workspace. The dataset contains de-identified clinical records across 186 distinct hospital facilities.

- **Total ICU Unit Stays (`patientunitstayid`):** 2,520
- **Total Unique Patients (`uniquepid`):** 1,841
- **Total Unique Hospitals (`hospitalid`):** 186
- **Hospital Discharge Status:**
  - Alive: 2,280 (90.48%)
  - Expired (In-Hospital Mortality): 212 (8.41%)
  - Missing/Unrecorded: 28 (1.11%)
- **ICU Unit Discharge Status:**
  - Alive: 2,392 (94.92%)
  - Expired (ICU Mortality): 126 (5.00%)
  - Missing/Unrecorded: 2 (0.08%)

## Hospital Distribution & Federated Learning Feasibility
- In the full eICU database (200,000+ stays), hospitals contribute thousands of encounters.
- In this **Demo v2.0.1 release**, the 2,520 records are distributed over 186 hospitals.
- **Top 5 Hospitals by stay count:**
  - Hospital 146: 40 stays
  - Hospital 123: 30 stays
  - Hospital 171: 25 stays
  - Hospital 157: 25 stays
  - Hospital 167: 24 stays
- **Small-volume sites:** The vast majority of hospitals have between 10 and 15 patient records.

### Federated Learning Structuring Strategy:
1. **Real-Hospital Multi-Site Partitioning:**
   - We identify the top participating hospitals (e.g., top 4 to 8 hospitals with 20-40 patient stays each) as real distinct hospital clients.
   - We group remaining small-volume sites into regional / bed-category clusters or realistic non-IID hospital nodes.
2. **Simulation Governance Rule:**
   - Consistent with Section 4 of the project specifications, we document this explicitly: any clustering or Dirichlet partitioning across clients must be clearly reported as simulated/clustered multi-site configurations, ensuring absolute scientific integrity.

## Resolution of the Primary Prediction Target
In healthcare informatics and machine learning, "predicting the hospital ID" is a hospital identification task (confounding site classification), NOT a clinical outcome.

Based on clinical significance, dataset availability, and benchmark literature (e.g., APACHE IV, MIMIC/eICU benchmark papers):
1. **Primary Clinical Endpoint (Recommended): In-Hospital Mortality (`hospitaldischargestatus == 'Expired'`)**
   - Binary classification: 0 (Alive) vs 1 (Expired).
   - Class balance: 212 positive cases (8.5% mortality) vs 2,280 negative cases.
   - Prediction window: First 24 hours of ICU admission (using admission vitals, first 24h labs, APACHE physiology variables, admission diagnoses).
   - Clinical Utility: Early warning risk stratification for critical ICU deterioration.
2. **Secondary Clinical Endpoint: Prolonged ICU Length of Stay (> 3 days / > 4,320 minutes)**
   - Binary classification: ~33% positive prevalence.
   - Resource planning and capacity management endpoint.

## Data Leakage Safeguards
To prevent target leakage:
1. All post-discharge variables (`hospitaldischargeoffset`, `hospitaldischargelocation`, `unitdischargelocation`, `dischargeweight`) must be excluded from feature sets.
2. Only observations from the first 24 hours (`offset <= 1440` minutes) or baseline admission data are used as predictor inputs.
3. Splits are strictly **patient-disjoint** using `uniquepid`, ensuring that readmitted patients never leak across training, validation, calibration, or test sets.
