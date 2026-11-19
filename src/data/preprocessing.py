"""Data preprocessing pipeline for tabular ICU clinical data.
Ensures zero-leakage transformation by fitting statistics strictly on training data.
"""

import logging
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "heartrate",
    "meanbp",
    "respiratoryrate",
    "temperature",
    "sodium",
    "creatinine",
    "bun",
    "glucose",
    "wbc",
    "hematocrit",
    "shock_index",
    "bun_to_creatinine",
    "gcs_total",
    "comorbidity_burden",
    "organ_support_count"
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "unittype",
    "region"
]

BINARY_FLAGS = [
    "ph_is_missing",
    "pao2_is_missing",
    "bilirubin_is_missing",
    "albumin_is_missing"
]


class ClinicalPreprocessor:
    """Zero-leakage preprocessor: fits imputation and scalers only on training partitions."""

    def __init__(
        self,
        numerical_cols: List[str] = NUMERICAL_FEATURES,
        categorical_cols: List[str] = CATEGORICAL_FEATURES,
        binary_cols: List[str] = BINARY_FLAGS
    ):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.binary_cols = binary_cols

        self.impute_medians: Dict[str, float] = {}
        self.scaler = StandardScaler()
        self.cat_categories: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "ClinicalPreprocessor":
        """Fits imputation medians, categorical categories, and standard scalers on training data."""
        # 1. Numerical medians
        for col in self.numerical_cols:
            if col in df_train.columns:
                med = df_train[col].dropna().median()
                self.impute_medians[col] = float(med if pd.notna(med) else 0.0)

        # 2. Fit Scaler on imputed numerical train
        df_num_imputed = df_train[self.numerical_cols].copy()
        for col in self.numerical_cols:
            if col in df_num_imputed.columns:
                df_num_imputed[col] = df_num_imputed[col].fillna(self.impute_medians[col])
        self.scaler.fit(df_num_imputed)

        # 3. Categorical unique categories
        for col in self.categorical_cols:
            if col in df_train.columns:
                unique_cats = df_train[col].dropna().astype(str).unique().tolist()
                self.cat_categories[col] = sorted(unique_cats)

        # Record final feature names
        names = list(self.numerical_cols)
        for col in self.categorical_cols:
            for cat in self.cat_categories.get(col, []):
                names.append(f"{col}_{cat}")
        for col in self.binary_cols:
            if col in df_train.columns:
                names.append(col)

        self.feature_names = names
        self.is_fitted = True
        logger.info(f"ClinicalPreprocessor fitted on {len(df_train)} rows. Total features: {len(self.feature_names)}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input dataframe into scaled feature matrix using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        # 1. Transform numerical features
        df_num = df[self.numerical_cols].copy()
        for col in self.numerical_cols:
            if col in df_num.columns:
                df_num[col] = df_num[col].fillna(self.impute_medians.get(col, 0.0))
            else:
                df_num[col] = self.impute_medians.get(col, 0.0)
        num_scaled = self.scaler.transform(df_num)

        # 2. Transform categorical features (one-hot aligned with fit)
        cat_arrays = []
        for col in self.categorical_cols:
            known_cats = self.cat_categories.get(col, [])
            if col in df.columns:
                s = df[col].astype(str)
                for cat in known_cats:
                    cat_arrays.append((s == cat).astype(float).values[:, None])
            else:
                for cat in known_cats:
                    cat_arrays.append(np.zeros((len(df), 1)))

        # 3. Binary flags
        bin_arrays = []
        for col in self.binary_cols:
            if col in df.columns:
                bin_arrays.append(df[col].fillna(0.0).astype(float).values[:, None])
            else:
                bin_arrays.append(np.zeros((len(df), 1)))

        all_parts = [num_scaled]
        if cat_arrays:
            all_parts.append(np.hstack(cat_arrays))
        if bin_arrays:
            all_parts.append(np.hstack(bin_arrays))

        X = np.hstack(all_parts)
        return X

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fits on train and returns transformed matrix."""
        return self.fit(df_train).transform(df_train)

    def save(self, filepath: str):
        """Serializes fitted preprocessor."""
        joblib.dump(self, filepath)
        logger.info(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ClinicalPreprocessor":
        """Loads serialized preprocessor."""
        return joblib.load(filepath)

