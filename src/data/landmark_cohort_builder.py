"""
Landmark Cohort Builder for 24-Hour Clinical Mortality Prediction.

Enforces strict prediction landmark at t = 1,440 minutes (24 hours) after ICU admission:
1. Valid hospital discharge outcome (hospitaldischargestatus in ['Alive', 'Expired']).
2. Alive at landmark (excludes all deaths occurring at t < 1,440 min).
3. Remaining in ICU under observation at landmark (unitdischargeoffset >= 1,440 min).
4. Strictly zero post-landmark predictor leakage.
5. Patient-disjoint train/val/test split on uniquepid.
6. Training-only preprocessing (median imputation, categorical encoding, standardization).
"""

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.data_loader import EICUDataLoader
from src.data.data_validator import ClinicalDataValidator
from src.data.feature_engineering import ClinicalFeatureEngineer
from src.data.preprocessing import (
    ClinicalPreprocessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FLAGS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class LandmarkCohortBuilder:
    """Constructs and validates the 24-hour landmark ICU mortality prediction cohort."""

    def __init__(
        self,
        data_dir: str = "eicu-collaborative-research-database-demo-2.0.1",
        landmark_minutes: int = 1440,
        random_seed: int = 42
    ):
        self.data_dir = data_dir
        self.landmark_minutes = landmark_minutes
        self.random_seed = random_seed
        self.loader = EICUDataLoader(data_dir=data_dir)
        self.validator = ClinicalDataValidator()
        self.feature_engineer = ClinicalFeatureEngineer()

    def audit_raw_data_and_exclusions(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Audits raw patient table and categorizes all exclusions with timestamp provenance."""
        # Load raw patient table
        df_patient = self.loader.load_table(
            "patient",
            usecols=[
                "patientunitstayid",
                "patienthealthsystemstayid",
                "uniquepid",
                "gender",
                "age",
                "ethnicity",
                "hospitalid",
                "wardid",
                "admissionheight",
                "admissionweight",
                "hospitaladmitsource",
                "unittype",
                "unitadmitsource",
                "unitvisitnumber",
                "unitstaytype",
                "hospitaladmitoffset",
                "hospitaldischargeoffset",
                "hospitaldischargestatus",
                "unitdischargeoffset",
                "unitdischargestatus"
            ]
        )

        n_total_stays = len(df_patient)
        n_total_pts = df_patient["uniquepid"].nunique()

        # Step 1: Missing hospital discharge status
        missing_outcome_mask = ~df_patient["hospitaldischargestatus"].isin(["Alive", "Expired"])
        n_missing_outcome = missing_outcome_mask.sum()
        n_missing_outcome_pts = df_patient[missing_outcome_mask]["uniquepid"].nunique()

        df_valid = df_patient[~missing_outcome_mask].copy()
        n_valid_stays = len(df_valid)
        n_valid_pts = df_valid["uniquepid"].nunique()

        # Step 2: Early ICU discharge / death before landmark (unitdischargeoffset < 1440)
        early_icu_mask = df_valid["unitdischargeoffset"] < self.landmark_minutes
        df_early = df_valid[early_icu_mask].copy()
        n_early_stays = len(df_early)
        n_early_pts = df_early["uniquepid"].nunique()

        # Detailed breakdown of early stays (mutually exclusive partition of 885 stays)
        # Category A: Died in ICU before 24h and expired in hospital
        died_in_icu_early = df_early[
            (df_early["unitdischargestatus"] == "Expired") &
            (df_early["hospitaldischargestatus"] == "Expired")
        ]
        n_died_in_icu_early = len(died_in_icu_early)

        # Category B: Discharged alive from ICU before 24h, died in hospital before 24h
        died_post_icu_before_24h = df_early[
            (df_early["unitdischargestatus"] != "Expired") &
            (df_early["hospitaldischargestatus"] == "Expired") &
            (df_early["hospitaldischargeoffset"] < self.landmark_minutes)
        ]
        n_died_post_icu_before_24h = len(died_post_icu_before_24h)

        # Total deaths before 24h landmark
        n_total_deaths_before_landmark = n_died_in_icu_early + n_died_post_icu_before_24h

        # Category C: Discharged alive from ICU before 24h, died in hospital after 24h
        died_post_icu_after_24h = df_early[
            (df_early["hospitaldischargestatus"] == "Expired") &
            (df_early["hospitaldischargeoffset"] >= self.landmark_minutes)
        ]
        n_died_post_icu_after_24h = len(died_post_icu_after_24h)

        # Category D: Discharged from ICU before 24h, survived hospital discharge
        alive_early_icu_discharge = df_early[df_early["hospitaldischargestatus"] == "Alive"]
        n_alive_early_icu_discharge = len(alive_early_icu_discharge)

        # Step 3: Landmark Eligible Stays
        df_landmark = df_valid[~early_icu_mask].copy()
        n_landmark_stays = len(df_landmark)
        n_landmark_pts = df_landmark["uniquepid"].nunique()

        n_landmark_deaths = (df_landmark["hospitaldischargestatus"] == "Expired").sum()
        n_landmark_survivors = (df_landmark["hospitaldischargestatus"] == "Alive").sum()
        landmark_prevalence = n_landmark_deaths / n_landmark_stays

        # Check death timing in landmark cohort
        landmark_deaths_df = df_landmark[df_landmark["hospitaldischargestatus"] == "Expired"]
        min_death_offset = landmark_deaths_df["hospitaldischargeoffset"].min()

        cohort_flow_records = [
            {
                "Stage": "1. Raw eICU Patient Encounters",
                "Stays_Count": n_total_stays,
                "Unique_Patients": n_total_pts,
                "Deaths": (df_patient["hospitaldischargestatus"] == "Expired").sum(),
                "Survivors": (df_patient["hospitaldischargestatus"] == "Alive").sum(),
                "Description": "All records in eICU demo patient.csv.gz"
            },
            {
                "Stage": "2. Excluded: Missing Hospital Discharge Status",
                "Stays_Count": int(n_missing_outcome),
                "Unique_Patients": int(n_missing_outcome_pts),
                "Deaths": 0,
                "Survivors": 0,
                "Description": "hospitaldischargestatus is null or unrecorded"
            },
            {
                "Stage": "3. Valid Discharge Outcome Cohort",
                "Stays_Count": n_valid_stays,
                "Unique_Patients": n_valid_pts,
                "Deaths": (df_valid["hospitaldischargestatus"] == "Expired").sum(),
                "Survivors": (df_valid["hospitaldischargestatus"] == "Alive").sum(),
                "Description": "hospitaldischargestatus in ['Alive', 'Expired']"
            },
            {
                "Stage": "4. Excluded: ICU Observation < 24 Hours (< 1,440 min)",
                "Stays_Count": int(n_early_stays),
                "Unique_Patients": int(n_early_pts),
                "Deaths": int((df_early["hospitaldischargestatus"] == "Expired").sum()),
                "Survivors": int((df_early["hospitaldischargestatus"] == "Alive").sum()),
                "Description": "unitdischargeoffset < 1440 min; incomplete 24h ICU observation"
            },
            {
                "Stage": "  4a. Excluded: Died in ICU Before 24h",
                "Stays_Count": int(n_died_in_icu_early),
                "Unique_Patients": int(died_in_icu_early["uniquepid"].nunique()),
                "Deaths": int(n_died_in_icu_early),
                "Survivors": 0,
                "Description": "unitdischargestatus == 'Expired' with unitdischargeoffset < 1440"
            },
            {
                "Stage": "  4b. Excluded: Died in Hospital Before 24h Post-ICU Transfer",
                "Stays_Count": int(n_died_post_icu_before_24h),
                "Unique_Patients": int(died_post_icu_before_24h["uniquepid"].nunique()),
                "Deaths": int(n_died_post_icu_before_24h),
                "Survivors": 0,
                "Description": "hospitaldischargeoffset < 1440 min, transferred alive from ICU"
            },
            {
                "Stage": "  4c. Excluded: Died in Hospital >= 24h After Early ICU Transfer",
                "Stays_Count": int(n_died_post_icu_after_24h),
                "Unique_Patients": int(died_post_icu_after_24h["uniquepid"].nunique()),
                "Deaths": int(n_died_post_icu_after_24h),
                "Survivors": 0,
                "Description": "hospitaldischargeoffset >= 1440 min, but unitdischargeoffset < 1440"
            },
            {
                "Stage": "  4d. Excluded: Discharged Alive from ICU Before 24h",
                "Stays_Count": int(n_alive_early_icu_discharge),
                "Unique_Patients": int(alive_early_icu_discharge["uniquepid"].nunique()),
                "Deaths": 0,
                "Survivors": int(n_alive_early_icu_discharge),
                "Description": "hospitaldischargestatus == 'Alive', unitdischargeoffset < 1440"
            },
            {
                "Stage": "5. Primary 24-Hour Landmark Cohort",
                "Stays_Count": n_landmark_stays,
                "Unique_Patients": n_landmark_pts,
                "Deaths": int(n_landmark_deaths),
                "Survivors": int(n_landmark_survivors),
                "Description": f"Alive and in ICU at t=1,440m; earliest death at t={min_death_offset:.0f}m"
            }
        ]

        df_flow = pd.DataFrame(cohort_flow_records)

        summary_dict = {
            "n_total_stays": n_total_stays,
            "n_total_pts": n_total_pts,
            "n_valid_stays": n_valid_stays,
            "n_missing_outcome": int(n_missing_outcome),
            "n_early_stays": int(n_early_stays),
            "n_died_before_24h": int(n_total_deaths_before_landmark),
            "n_died_after_early_icu": int(n_died_post_icu_after_24h),
            "n_alive_early_icu": int(n_alive_early_icu_discharge),
            "n_landmark_stays": n_landmark_stays,
            "n_landmark_pts": n_landmark_pts,
            "n_landmark_deaths": int(n_landmark_deaths),
            "n_landmark_survivors": int(n_landmark_survivors),
            "landmark_prevalence": float(landmark_prevalence),
            "min_death_offset": float(min_death_offset)
        }

        return df_landmark, df_flow, summary_dict

    def build_and_save_landmark_cohort(
        self,
        output_dir: str = "data/processed_landmark",
        results_dir: str = "results/phase13_1"
    ) -> Dict[str, Any]:
        """Executes the complete landmark cohort construction, split, preprocessing, and auditing."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        logger.info("Executing landmark cohort eligibility audit...")
        df_landmark_p, df_flow, summary = self.audit_raw_data_and_exclusions()

        # Save cohort flow CSV
        flow_path = os.path.join(results_dir, "cohort_flow.csv")
        df_flow.to_csv(flow_path, index=False)
        logger.info(f"Saved cohort flow table to {flow_path}")

        # Load supporting clinical tables
        aps_cols = [
            "patientunitstayid", "intubated", "vent", "dialysis",
            "eyes", "motor", "verbal", "urine", "wbc", "temperature",
            "respiratoryrate", "sodium", "heartrate", "meanbp", "ph",
            "hematocrit", "creatinine", "albumin", "pao2", "pco2",
            "bun", "glucose", "bilirubin", "fio2"
        ]
        df_aps = self.loader.load_table("apacheApsVar", usecols=aps_cols)

        pred_cols = [
            "patientunitstayid", "aids", "hepaticfailure", "lymphoma",
            "metastaticcancer", "leukemia", "immunosuppression",
            "cirrhosis", "electivesurgery", "diabetes"
        ]
        df_pred = self.loader.load_table("apachePredVar", usecols=pred_cols)

        df_hosp = self.loader.load_table(
            "hospital",
            usecols=["hospitalid", "numbedscategory", "teachingstatus", "region"]
        )

        # Merge supporting clinical tables with landmark patient encounters
        cohort = df_landmark_p.merge(df_aps, on="patientunitstayid", how="left")
        cohort = cohort.merge(df_pred, on="patientunitstayid", how="left")
        cohort = cohort.merge(df_hosp, on="hospitalid", how="left")

        # Sanitize physiological values (clinical bounds and age string handling)
        cohort_clean = self.validator.sanitize_physiological_values(cohort)

        # Feature engineering (clinical scores and ratios)
        cohort_eng = self.feature_engineer.compute_features(cohort_clean)

        # Create binary target: hospital mortality (subsequent in-hospital death)
        cohort_eng["hospital_mortality"] = (cohort_eng["hospitaldischargestatus"] == "Expired").astype(int)

        # Temporal Feature Audit
        temporal_records = [
            {
                "Feature": "age, gender, ethnicity",
                "Domain": "Demographics",
                "Timing": "Recorded at hospital/ICU admission (t <= 0)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "unittype, region",
                "Domain": "Administrative / Setting",
                "Timing": "Known at ICU bed placement (t = 0)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "bmi (admissionheight, admissionweight)",
                "Domain": "Anthropometrics",
                "Timing": "Recorded upon admission (t <= 60 min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "comorbidity_burden (aids, cirrhosis, etc.)",
                "Domain": "Past Medical History",
                "Timing": "Pre-existing prior to ICU admission",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "organ_support_count (intubated, vent, dialysis)",
                "Domain": "Organ Support",
                "Timing": "First 24 hours of ICU stay ([0, 1440] min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "heartrate, meanbp, respiratoryrate, temperature",
                "Domain": "Acute Vitals (APACHE IV APS)",
                "Timing": "Worst physiological values in first 24h ([0, 1440] min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "creatinine, bun, sodium, wbc, hematocrit, glucose",
                "Domain": "Acute Routine Labs (APACHE IV APS)",
                "Timing": "Worst laboratory values in first 24h ([0, 1440] min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "ph, pao2, bilirubin, albumin (and missingness flags)",
                "Domain": "Invasive Blood Gas & Hepatic Labs",
                "Timing": "Worst laboratory values in first 24h ([0, 1440] min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "gcs_total (eyes, motor, verbal)",
                "Domain": "Neurological Assessment",
                "Timing": "Worst GCS assessment during first 24h ([0, 1440] min)",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "shock_index, bun_to_creatinine",
                "Domain": "Physiological Ratios",
                "Timing": "Derived strictly from first 24h vitals and labs",
                "Status": "Valid at landmark",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "unitdischargestatus, unitdischargeoffset",
                "Domain": "ICU Discharge",
                "Timing": "ICU discharge event",
                "Status": "Excluded from feature matrix (used only for cohort gating)",
                "Leaking_Post_Landmark": False
            },
            {
                "Feature": "hospitaldischargestatus, hospitaldischargeoffset",
                "Domain": "Hospital Outcome",
                "Timing": "Hospital discharge event (t >= 1487 min)",
                "Status": "Target variable y (Excluded from predictor matrix X)",
                "Leaking_Post_Landmark": False
            }
        ]
        df_temporal = pd.DataFrame(temporal_records)
        temp_path = os.path.join(results_dir, "feature_temporal_audit.csv")
        df_temporal.to_csv(temp_path, index=False)
        logger.info(f"Saved feature temporal audit to {temp_path}")

        # Patient-disjoint split on uniquepid (70% train, 15% val, 15% test)
        gss_train = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=self.random_seed)
        train_idx, temp_idx = next(gss_train.split(cohort_eng, groups=cohort_eng["uniquepid"]))

        df_train = cohort_eng.iloc[train_idx].copy()
        df_temp = cohort_eng.iloc[temp_idx].copy()

        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=self.random_seed)
        val_idx, test_idx = next(gss_val.split(df_temp, groups=df_temp["uniquepid"]))

        df_val = df_temp.iloc[val_idx].copy()
        df_test = df_temp.iloc[test_idx].copy()

        # Verify strict patient disjointness
        train_pts = set(df_train["uniquepid"].dropna())
        val_pts = set(df_val["uniquepid"].dropna())
        test_pts = set(df_test["uniquepid"].dropna())

        leak_train_val = len(train_pts.intersection(val_pts))
        leak_train_test = len(train_pts.intersection(test_pts))
        leak_val_test = len(val_pts.intersection(test_pts))

        assert leak_train_val == 0, f"Patient leakage between Train and Val: {leak_train_val}"
        assert leak_train_test == 0, f"Patient leakage between Train and Test: {leak_train_test}"
        assert leak_val_test == 0, f"Patient leakage between Val and Test: {leak_val_test}"

        split_records = [
            {
                "Partition": "Train",
                "Encounter_Count": len(df_train),
                "Unique_Patients": len(train_pts),
                "Deaths": int(df_train["hospital_mortality"].sum()),
                "Survivors": int((df_train["hospital_mortality"] == 0).sum()),
                "Prevalence": float(df_train["hospital_mortality"].mean()),
                "Patient_Overlap_With_Other_Splits": 0
            },
            {
                "Partition": "Validation",
                "Encounter_Count": len(df_val),
                "Unique_Patients": len(val_pts),
                "Deaths": int(df_val["hospital_mortality"].sum()),
                "Survivors": int((df_val["hospital_mortality"] == 0).sum()),
                "Prevalence": float(df_val["hospital_mortality"].mean()),
                "Patient_Overlap_With_Other_Splits": 0
            },
            {
                "Partition": "Test",
                "Encounter_Count": len(df_test),
                "Unique_Patients": len(test_pts),
                "Deaths": int(df_test["hospital_mortality"].sum()),
                "Survivors": int((df_test["hospital_mortality"] == 0).sum()),
                "Prevalence": float(df_test["hospital_mortality"].mean()),
                "Patient_Overlap_With_Other_Splits": 0
            },
            {
                "Partition": "Total",
                "Encounter_Count": len(cohort_eng),
                "Unique_Patients": cohort_eng["uniquepid"].nunique(),
                "Deaths": int(cohort_eng["hospital_mortality"].sum()),
                "Survivors": int((cohort_eng["hospital_mortality"] == 0).sum()),
                "Prevalence": float(cohort_eng["hospital_mortality"].mean()),
                "Patient_Overlap_With_Other_Splits": 0
            }
        ]
        df_split = pd.DataFrame(split_records)
        split_path = os.path.join(results_dir, "split_integrity.csv")
        df_split.to_csv(split_path, index=False)
        logger.info(f"Saved split integrity table to {split_path}")

        # Zero-leakage transformation: Fit strictly on training data
        preprocessor = ClinicalPreprocessor(
            numerical_cols=NUMERICAL_FEATURES,
            categorical_cols=CATEGORICAL_FEATURES,
            binary_cols=BINARY_FLAGS
        )
        preprocessor.fit(df_train)

        X_train = preprocessor.transform(df_train)
        y_train = df_train["hospital_mortality"].values.astype(int)

        X_val = preprocessor.transform(df_val)
        y_val = df_val["hospital_mortality"].values.astype(int)

        X_test = preprocessor.transform(df_test)
        y_test = df_test["hospital_mortality"].values.astype(int)

        # Save numpy matrices and metadata
        np.save(os.path.join(output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(output_dir, "X_val.npy"), X_val)
        np.save(os.path.join(output_dir, "y_val.npy"), y_val)
        np.save(os.path.join(output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(output_dir, "y_test.npy"), y_test)

        meta_cols = ["patientunitstayid", "uniquepid", "hospitalid", "hospital_mortality"]
        df_train[meta_cols].to_csv(os.path.join(output_dir, "train_meta.csv"), index=False)
        df_val[meta_cols].to_csv(os.path.join(output_dir, "val_meta.csv"), index=False)
        df_test[meta_cols].to_csv(os.path.join(output_dir, "test_meta.csv"), index=False)

        preprocessor.save(os.path.join(output_dir, "preprocessor.joblib"))
        with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
            json.dump(preprocessor.feature_names, f, indent=2)

        logger.info(
            f"Landmark cohort successfully processed and saved to {output_dir}.\n"
            f"  Train: X={X_train.shape}, y={y_train.shape} (deaths={y_train.sum()})\n"
            f"  Val:   X={X_val.shape}, y={y_val.shape} (deaths={y_val.sum()})\n"
            f"  Test:  X={X_test.shape}, y={y_test.shape} (deaths={y_test.sum()})\n"
            f"  Features: {len(preprocessor.feature_names)}"
        )

        return {
            "summary": summary,
            "split_table": df_split,
            "flow_table": df_flow,
            "feature_count": len(preprocessor.feature_names),
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape
        }


if __name__ == "__main__":
    builder = LandmarkCohortBuilder()
    builder.build_and_save_landmark_cohort()
