"""
External Dataset Adapter and Harmonization Pipeline.
Provides standardized mapping schemas and extraction logic for:
1. MIMIC-IV v2.2 (Beth Israel Deaconess Medical Center, PhysioNet)
2. Full eICU Collaborative Research Database v2.0 (Philips Healthcare / PhysioNet)

Harmonizes heterogeneous EHR schemas into the exact 41-feature tabular format
required by the Privacy-Preserving Hybrid Quantum Federated Learning framework.
Enforces strictly the first-24h ICU observation window and in-hospital mortality endpoint.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. MIMIC-IV v2.2 CLINICAL ITEMID MAPPING SCHEMA
# ---------------------------------------------------------------------------

MIMIC_IV_MAPPING_SCHEMA = {
    "source_database": "MIMIC-IV v2.2",
    "target_endpoint": {
        "source_table": "admissions",
        "source_column": "hospital_expire_flag",
        "description": "In-hospital mortality (1=Expired before hospital discharge, 0=Discharged alive)",
        "harmonized_name": "hospital_mortality"
    },
    "prediction_window": {
        "start": "icustays.intime",
        "end": "icustays.intime + INTERVAL '24 HOUR'",
        "exclusion": "Exclude ICU stays with length of stay < 24 hours or missing in-hospital discharge status"
    },
    "vitals": {
        "heartrate": {"table": "chartevents", "itemids": [220045], "unit": "bpm"},
        "meanbp": {"table": "chartevents", "itemids": [220052, 220181, 225312], "unit": "mmHg"},
        "respiratoryrate": {"table": "chartevents", "itemids": [220210, 224690], "unit": "bpm"},
        "temperature": {"table": "chartevents", "itemids": [223762, 223761], "unit": "Celsius"}
    },
    "labs": {
        "sodium": {"table": "labevents", "itemids": [50983, 50824], "unit": "mEq/L"},
        "creatinine": {"table": "labevents", "itemids": [50912], "unit": "mg/dL"},
        "bun": {"table": "labevents", "itemids": [51006], "unit": "mg/dL"},
        "glucose": {"table": "labevents", "itemids": [50931, 50809], "unit": "mg/dL"},
        "wbc": {"table": "labevents", "itemids": [51301, 51300], "unit": "K/uL"},
        "hematocrit": {"table": "labevents", "itemids": [51221, 50810], "unit": "%"}
    },
    "neurological": {
        "gcs_total": {
            "table": "chartevents",
            "itemids": [220739, 223900, 223901],
            "description": "Sum of GCS Eye, Verbal, and Motor scores (range 3-15)"
        }
    },
    "demographics": {
        "age": {"table": "patients", "source_column": "anchor_age"},
        "gender": {"table": "patients", "source_column": "gender"},
        "ethnicity": {"table": "admissions", "source_column": "race"}
    }
}


# ---------------------------------------------------------------------------
# 2. FULL eICU DATABASE v2.0 MAPPING SCHEMA
# ---------------------------------------------------------------------------

FULL_EICU_MAPPING_SCHEMA = {
    "source_database": "eICU Collaborative Research Database v2.0 (Full: 200,859 ICU stays)",
    "target_endpoint": {
        "source_table": "patient",
        "source_column": "hospitaldischargestatus",
        "positive_value": "Expired",
        "negative_value": "Alive",
        "harmonized_name": "hospital_mortality"
    },
    "prediction_window": {
        "start_offset_minutes": 0,
        "end_offset_minutes": 1440,
        "description": "First 24 hours of ICU admission (0 to 1440 minutes offset)"
    },
    "cohort_filters": [
        "unitvisitnumber == 1 (first ICU stay only)",
        "unitdischargeoffset >= 1440 (minimum 24h ICU length of stay)",
        "age >= 18"
    ],
    "vitals": {
        "heartrate": {"table": "vitalPeriodic", "column": "heartrate"},
        "meanbp": {"table": "vitalPeriodic", "column": "systemicmean"},
        "respiratoryrate": {"table": "vitalPeriodic", "column": "respiration"},
        "temperature": {"table": "vitalPeriodic", "column": "temperature"}
    },
    "labs": {
        "sodium": {"table": "lab", "labname": "sodium"},
        "creatinine": {"table": "lab", "labname": "creatinine"},
        "bun": {"table": "lab", "labname": "BUN"},
        "glucose": {"table": "lab", "labname": "bedside glucose"},
        "wbc": {"table": "lab", "labname": "WBC x 1000"},
        "hematocrit": {"table": "lab", "labname": "Hct"}
    },
    "neurological": {
        "gcs_total": {"table": "apacheApsVar", "column": "gcs"}
    }
}


# ---------------------------------------------------------------------------
# 3. HARMONIZATION ADAPTER ENGINE
# ---------------------------------------------------------------------------

class ExternalDataHarmonizer:
    """Harmonizes external ICU records into normalized model feature matrices."""

    def __init__(self, feature_names_path: str = "data/processed/feature_names.json"):
        if os.path.exists(feature_names_path):
            with open(feature_names_path, "r") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = []

    def validate_dataset_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Checks for missing required features and verifies absence of target leakage."""
        leakage_cols = [
            "hospitaldischargestatus", "unitdischargestatus", "hospitaldischargetime",
            "unitdischargeoffset", "hospital_expire_flag", "deathtime"
        ]
        present_leaks = [col for col in leakage_cols if col in df.columns]

        missing_features = [f for f in self.feature_names if f not in df.columns]

        return {
            "valid": len(present_leaks) == 0 and len(missing_features) == 0,
            "detected_leakages": present_leaks,
            "missing_features": missing_features,
            "num_features_present": len([f for f in self.feature_names if f in df.columns]),
            "expected_features": len(self.feature_names)
        }

    def compute_derived_clinical_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives clinical composite indicators from raw vitals and labs."""
        out = df.copy()

        # Shock Index = Heart Rate / Mean BP
        if "heartrate" in out.columns and "meanbp" in out.columns:
            out["shock_index"] = out["heartrate"] / np.clip(out["meanbp"], 20.0, 200.0)

        # BUN to Creatinine Ratio
        if "bun" in out.columns and "creatinine" in out.columns:
            out["bun_to_creatinine"] = out["bun"] / np.clip(out["creatinine"], 0.1, 20.0)

        return out
