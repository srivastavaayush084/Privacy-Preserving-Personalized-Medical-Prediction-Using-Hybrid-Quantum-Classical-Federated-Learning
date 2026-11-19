"""Data loader module for eICU Collaborative Research Database Demo v2.0.1.
Loads core ICU tables while preserving institutional provenance.
"""

import os
import logging
from typing import Dict, Optional, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EICUDataLoader:
    """Loads and merges eICU demo tables for medical risk prediction."""

    def __init__(self, data_dir: str = "eicu-collaborative-research-database-demo-2.0.1"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"eICU directory not found at: {self.data_dir}")

    def load_table(self, table_name: str, usecols: Optional[list] = None) -> pd.DataFrame:
        """Loads a single gzipped CSV table."""
        filename = f"{table_name}.csv.gz"
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Table file not found: {path}")
        logger.info(f"Loading {filename}...")
        return pd.read_csv(path, usecols=usecols, low_memory=False)

    def load_cohort(self) -> pd.DataFrame:
        """Loads and integrates core admission, physiology, and comorbidity tables.
        
        Returns:
            pd.DataFrame: Merged patient cohort indexed by patientunitstayid.
        """
        # 1. Patient table
        patient_cols = [
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
            "hospitaldischargestatus",
            "unitdischargestatus",
            "unitdischargeoffset"
        ]
        df_patient = self.load_table("patient", usecols=patient_cols)

        # 2. APACHE APS Physiology table (first 24-hour acute physiology values)
        aps_cols = [
            "patientunitstayid",
            "intubated",
            "vent",
            "dialysis",
            "eyes",
            "motor",
            "verbal",
            "urine",
            "wbc",
            "temperature",
            "respiratoryrate",
            "sodium",
            "heartrate",
            "meanbp",
            "ph",
            "hematocrit",
            "creatinine",
            "albumin",
            "pao2",
            "pco2",
            "bun",
            "glucose",
            "bilirubin",
            "fio2"
        ]
        df_aps = self.load_table("apacheApsVar", usecols=aps_cols)

        # 3. APACHE Prediction Variables (comorbidities, pre-existing conditions)
        pred_cols = [
            "patientunitstayid",
            "aids",
            "hepaticfailure",
            "lymphoma",
            "metastaticcancer",
            "leukemia",
            "immunosuppression",
            "cirrhosis",
            "electivesurgery",
            "diabetes",
            "diedinhospital"
        ]
        df_pred = self.load_table("apachePredVar", usecols=pred_cols)

        # 4. Hospital characteristics
        df_hosp = self.load_table("hospital", usecols=["hospitalid", "numbedscategory", "teachingstatus", "region"])

        # Merge tables left-joining on patientunitstayid
        cohort = df_patient.merge(df_aps, on="patientunitstayid", how="left")
        cohort = cohort.merge(df_pred, on="patientunitstayid", how="left")
        cohort = cohort.merge(df_hosp, on="hospitalid", how="left")

        logger.info(f"Loaded integrated cohort: {cohort.shape[0]} encounters, {cohort.shape[1]} raw columns.")
        return cohort

