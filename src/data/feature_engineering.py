"""Feature engineering for ICU clinical prediction.
Extracts established physiological scores, organ failure markers, and interaction terms.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ClinicalFeatureEngineer:
    """Computes established ICU physiological scores, organ failure indices, and risk ratios."""

    def __init__(self):
        pass

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives clinical features and interactions."""
        df_feat = df.copy()

        # 1. Body Mass Index (BMI)
        if "admissionheight" in df_feat.columns and "admissionweight" in df_feat.columns:
            height_m = df_feat["admissionheight"] / 100.0
            # Clip height to plausible adult range (1.2m to 2.3m)
            height_m = height_m.clip(lower=1.2, upper=2.3)
            df_feat["bmi"] = df_feat["admissionweight"] / (height_m ** 2)
            # Clip BMI to plausible bounds (10 to 80)
            df_feat["bmi"] = df_feat["bmi"].clip(lower=10.0, upper=80.0)

        # 2. Shock Index = Heart Rate / Mean Arterial Pressure
        if "heartrate" in df_feat.columns and "meanbp" in df_feat.columns:
            safe_mbp = df_feat["meanbp"].replace(0, np.nan)
            df_feat["shock_index"] = df_feat["heartrate"] / safe_mbp
            df_feat["shock_index"] = df_feat["shock_index"].clip(lower=0.1, upper=5.0)

        # 3. BUN / Creatinine Ratio (Renal hypoperfusion marker)
        if "bun" in df_feat.columns and "creatinine" in df_feat.columns:
            safe_creat = df_feat["creatinine"].replace(0, np.nan)
            df_feat["bun_to_creatinine"] = df_feat["bun"] / safe_creat
            df_feat["bun_to_creatinine"] = df_feat["bun_to_creatinine"].clip(lower=1.0, upper=150.0)

        # 4. Neurological Score (GCS component sum: eyes + motor + verbal)
        gcs_cols = ["eyes", "motor", "verbal"]
        if all(c in df_feat.columns for c in gcs_cols):
            # Clean non-numeric or negative entries
            for c in gcs_cols:
                df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce").fillna(4) # default normal
            df_feat["gcs_total"] = df_feat["eyes"] + df_feat["motor"] + df_feat["verbal"]
            df_feat["gcs_total"] = df_feat["gcs_total"].clip(lower=3, upper=15)

        # 5. Organ Failure Comorbidity Burden
        comorb_cols = [
            "aids", "hepaticfailure", "lymphoma", "metastaticcancer",
            "leukemia", "immunosuppression", "cirrhosis", "diabetes"
        ]
        present_comorbs = [c for c in comorb_cols if c in df_feat.columns]
        if present_comorbs:
            df_feat["comorbidity_burden"] = df_feat[present_comorbs].fillna(0).sum(axis=1)

        # 6. Organ Support Intensity (Intubation + Mechanical Vent + Dialysis)
        support_cols = ["intubated", "vent", "dialysis"]
        present_support = [c for c in support_cols if c in df_feat.columns]
        if present_support:
            df_feat["organ_support_count"] = df_feat[present_support].apply(
                lambda s: (s > 0).astype(int)
            ).sum(axis=1)

        # 7. Missingness indicators for critical invasive tests
        for inv_col in ["ph", "pao2", "bilirubin", "albumin"]:
            if inv_col in df_feat.columns:
                df_feat[f"{inv_col}_is_missing"] = df_feat[inv_col].isna().astype(float)

        logger.info(f"Feature engineering complete. Total columns: {df_feat.shape[1]}")
        return df_feat

