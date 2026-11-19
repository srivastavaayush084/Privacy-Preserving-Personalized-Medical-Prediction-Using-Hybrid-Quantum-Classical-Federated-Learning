# Table Relationships in eICU Collaborative Research Database Demo v2.0.1

## Entity Relationship Overview

The eICU Collaborative Research Database is organized hierarchically around patients, hospital admissions, and ICU unit stays:

```
[hospital] (186 hospitals: hospitalid)
   │
   ▼
[patient] (1,841 unique patients: uniquepid)
   │
   ├── patienthealthsystemstayid (Hospital encounters)
   │
   └── patientunitstayid (2,520 ICU admissions - PRIMARY FOREIGN KEY)
         │
         ├── [apachePatientResult] (APACHE IV scores, predicted mortality)
         ├── [apacheApsVar] (Acute Physiology Score variables: vitals, lab worst values)
         ├── [apachePredVar] (Prediction variables: comorbidities, admit source)
         ├── [vitalPeriodic] (Routine 5-minute bedside vitals: HR, BP, SaO2, RR)
         ├── [vitalAperiodic] (Non-invasive BP, intermittent measurements)
         ├── [lab] (Laboratory chemistry, hematology, blood gas)
         ├── [medication] (Scheduled and PRN medications administered)
         ├── [infusiondrug] (Continuous vasoactive and sedative infusions)
         ├── [admissionDx] / [diagnosis] (ICD-9/10 and eICU clinical diagnoses)
         ├── [treatment] (Clinical interventions, ventilation, dialysis)
         ├── [nurseCharting] (GCS, pain scales, nursing assessments)
         └── [physicalExam] (Structured physical exam findings)
```

## Primary and Foreign Key Mapping

| Table | Primary Key | Foreign Key | Parent Table | Relationship | Description |
|---|---|---|---|---|---|
| `hospital` | `hospitalid` | - | - | 1 : N | Hospital characteristics (beds, teaching status, region) |
| `patient` | `patientunitstayid` | `hospitalid` | `hospital` | N : 1 | Core demographic and ICU admission record |
| `patient` | `patientunitstayid` | `uniquepid` | - | N : 1 | Patient master identity (allows multi-admission tracking) |
| `apachePatientResult`| `apachepatientresultsid`| `patientunitstayid` | `patient` | N : 1 | APACHE IV/IVa scoring and predictions |
| `apacheApsVar` | `apacheapsvarid` | `patientunitstayid` | `patient` | 1 : 1 | First 24-hour worst physiologic values |
| `apachePredVar` | `apachepredvarid`| `patientunitstayid` | `patient` | 1 : 1 | Pre-existing conditions and admission variables |
| `lab` | `labid` | `patientunitstayid` | `patient` | N : 1 | Longitudinal laboratory test results with time offsets |
| `vitalPeriodic` | `vitalperiodicid` | `patientunitstayid` | `patient` | N : 1 | 5-minute automated bedside telemetry observations |
| `vitalAperiodic` | `vitalaperiodicid` | `patientunitstayid` | `patient` | N : 1 | Intermittent non-invasive telemetry observations |
| `diagnosis` | `diagnosisid` | `patientunitstayid` | `patient` | N : 1 | Clinical diagnoses with ICD-9/10 strings |
| `medication` | `medicationid` | `patientunitstayid` | `patient` | N : 1 | Inpatient medications with dosing and route |
| `treatment` | `treatmentid` | `patientunitstayid` | `patient` | N : 1 | Procedures, intubations, dialysis, lines |

## Temporal Anchors and Offset Seminology
- All events are referenced to ICU admission (`patientunitstayid`) at `offset = 0` (minutes).
- Negative offsets indicate events that occurred prior to ICU admission (e.g. emergency department, operating room).
- `hospitaladmitoffset`: Minutes from ICU admission to hospital admission (typically negative).
- `unitdischargeoffset`: Length of ICU stay in minutes.
- `hospitaldischargeoffset`: Total length of hospital stay in minutes.
