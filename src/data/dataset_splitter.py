"""Dataset splitting module ensuring strict patient-disjoint partitions.
Prevents identity leakage by grouping all admissions of a patient into the same partition.
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


class PatientDisjointSplitter:
    """Splits EHR datasets by patient identifier (uniquepid) to eliminate data leakage."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

    def split(self, df: pd.DataFrame, patient_col: str = "uniquepid") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs a 3-way patient-disjoint split: Train, Validation, Test.
        
        Returns:
            Tuple of (df_train, df_val, df_test).
        """
        if patient_col not in df.columns:
            raise KeyError(f"Patient ID column '{patient_col}' not found in DataFrame.")

        # Step 1: Separate Train vs Temp (Val + Test)
        temp_ratio = self.val_ratio + self.test_ratio
        gss_train = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=self.random_seed)
        train_idx, temp_idx = next(gss_train.split(df, groups=df[patient_col]))

        df_train = df.iloc[train_idx].copy()
        df_temp = df.iloc[temp_idx].copy()

        # Step 2: Split Temp into Validation and Test
        relative_test_ratio = self.test_ratio / temp_ratio
        gss_val_test = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=self.random_seed)
        val_sub_idx, test_sub_idx = next(gss_val_test.split(df_temp, groups=df_temp[patient_col]))

        df_val = df_temp.iloc[val_sub_idx].copy()
        df_test = df_temp.iloc[test_sub_idx].copy()

        # Verify zero patient overlap
        train_pts = set(df_train[patient_col].dropna())
        val_pts = set(df_val[patient_col].dropna())
        test_pts = set(df_test[patient_col].dropna())

        assert len(train_pts.intersection(val_pts)) == 0, "Patient leakage between Train and Val!"
        assert len(train_pts.intersection(test_pts)) == 0, "Patient leakage between Train and Test!"
        assert len(val_pts.intersection(test_pts)) == 0, "Patient leakage between Val and Test!"

        logger.info(
            f"Patient-disjoint split created: "
            f"Train={len(df_train)} ({len(train_pts)} pts), "
            f"Val={len(df_val)} ({len(val_pts)} pts), "
            f"Test={len(df_test)} ({len(test_pts)} pts)."
        )
        return df_train, df_val, df_test

    def split_hospital_clients(
        self,
        df: pd.DataFrame,
        hospital_col: str = "hospitalid",
        patient_col: str = "uniquepid",
        min_samples: int = 15
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Partitions data into hospital-specific client subsets, each split patient-disjointly into train/val/test."""
        hosp_counts = df[hospital_col].value_counts()
        eligible_hosps = hosp_counts[hosp_counts >= min_samples].index.tolist()
        logger.info(f"Identified {len(eligible_hosps)} hospitals with >= {min_samples} stays.")

        clients = {}
        for hid in eligible_hosps:
            df_h = df[df[hospital_col] == hid].copy()
            # Split each hospital into train/test (e.g. 70/30)
            gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=self.random_seed)
            t_idx, v_idx = next(gss.split(df_h, groups=df_h[patient_col]))
            clients[f"hospital_{hid}"] = {
                "train": df_h.iloc[t_idx].copy(),
                "test": df_h.iloc[v_idx].copy(),
                "total_stays": len(df_h),
                "unique_patients": df_h[patient_col].nunique()
            }
        return clients

