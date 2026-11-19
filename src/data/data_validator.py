"""Data validation module for clinical EHR records.
Enforces clinical plausible boundaries, verifies key integrity, and flags target anomalies.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Clinically accepted physiological plausibility ranges for ICU patients
CLINICAL_BOUNDS = {
    "heartrate": (20.0, 300.0),
    "meanbp": (15.0, 250.0),
    "respiratoryrate": (2.0, 90.0),
    "temperature": (26.0, 45.0),
    "ph": (6.7, 7.8),
    "sodium": (90.0, 190.0),
    "creatinine": (0.1, 35.0),
    "bun": (1.0, 250.0),
    "glucose": (10.0, 1500.0),
    "wbc": (0.1, 200.0),
    "hematocrit": (5.0, 75.0),
    "pao2": (15.0, 700.0),
    "pco2": (10.0, 150.0)
}


class ClinicalDataValidator:
    """Validates clinical data ranges, schema consistency, and integrity constraints."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]] = CLINICAL_BOUNDS):
        self.bounds = bounds

    def validate_keys(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Validates primary and foreign key integrity."""
        results = {
            "has_patientunitstayid": "patientunitstayid" in df.columns,
            "has_uniquepid": "uniquepid" in df.columns,
            "has_hospitalid": "hospitalid" in df.columns,
            "unique_stays_equal_rows": len(df) == df["patientunitstayid"].nunique() if "patientunitstayid" in df.columns else False
        }
        return results

    def sanitize_physiological_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flags or replaces physically impossible values with NaN for safe imputation."""
        df_clean = df.copy()
        
        # Age cleaning: in eICU, age > 89 is often stored as '> 89' string
        if "age" in df_clean.columns:
            df_clean["age"] = df_clean["age"].astype(str).str.replace("> 89", "90").replace("nan", np.nan)
            df_clean["age"] = pd.to_numeric(df_clean["age"], errors="coerce")
            
        for col, (lower, upper) in self.bounds.items():
            if col in df_clean.columns:
                # eICU sometimes uses -1.0 as a missing indicator in APACHE tables
                invalid_mask = (df_clean[col] < lower) | (df_clean[col] > upper) | (df_clean[col] == -1.0)
                num_invalid = invalid_mask.sum()
                if num_invalid > 0:
                    logger.debug(f"Column '{col}': setting {num_invalid} out-of-range values to NaN.")
                    df_clean.loc[invalid_mask, col] = np.nan
                    
        return df_clean

    def validate_targets(self, df: pd.DataFrame) -> Dict[str, int]:
        """Verifies distribution of clinical prediction targets."""
        if "hospitaldischargestatus" not in df.columns:
            raise ValueError("hospitaldischargestatus column not found.")
            
        status_counts = df["hospitaldischargestatus"].value_counts(dropna=False).to_dict()
        valid_targets = df["hospitaldischargestatus"].isin(["Alive", "Expired"]).sum()
        missing_targets = df["hospitaldischargestatus"].isna().sum()
        
        return {
            "status_counts": status_counts,
            "valid_targets": int(valid_targets),
            "missing_targets": int(missing_targets)
        }

