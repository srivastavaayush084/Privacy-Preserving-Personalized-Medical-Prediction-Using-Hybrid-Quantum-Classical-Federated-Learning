# Data Quality and Leakage Audit Report

## 1. Zero-Leakage Certification
- **Leakage-Free Status:** PASSED (Zero Overlap)
- **Train - Val Patient Overlap:** 0
- **Train - Test Patient Overlap:** 0
- **Val - Test Patient Overlap:** 0

## 2. Partition Summary
| Split | Total Stays | Unique Patients | Mortality Positives (Expired) | Mortality Negatives (Alive) | Mortality Rate (%) |
|---|---|---|---|---|---|
| **Train** | 1712 | 1274 | 142 | 1570 | 8.29% |
| **Val** | 387 | 273 | 30 | 357 | 7.75% |
| **Test** | 393 | 273 | 40 | 353 | 10.18% |

## 3. Extracted Features (41 features)
```
age, bmi, heartrate, meanbp, respiratoryrate, temperature, sodium, creatinine, bun, glucose, wbc, hematocrit, shock_index, bun_to_creatinine, gcs_total, comorbidity_burden, organ_support_count, gender_Female, gender_Male, ethnicity_African American, ethnicity_Asian, ethnicity_Caucasian, ethnicity_Hispanic, ethnicity_Native American, ethnicity_Other/Unknown, unittype_CCU-CTICU, unittype_CSICU, unittype_CTICU, unittype_Cardiac ICU, unittype_MICU, unittype_Med-Surg ICU, unittype_Neuro ICU, unittype_SICU, region_Midwest, region_Northeast, region_South, region_West, ph_is_missing, pao2_is_missing, bilirubin_is_missing, albumin_is_missing
```
