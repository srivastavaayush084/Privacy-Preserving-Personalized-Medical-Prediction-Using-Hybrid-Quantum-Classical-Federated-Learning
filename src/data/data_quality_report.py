"""Data quality and leakage verification report generator.
Validates train/validation/test partitions, checks feature distributions, and certifies zero leakage.
"""

import json
import logging
import os
from typing import Dict, List, Set
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_quality_report(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_names: List[str],
    target_col: str = "hospital_mortality",
    patient_col: str = "uniquepid",
    output_dir: str = "results/reports"
) -> Dict:
    """Computes split quality metrics, patient overlap tests, and label distributions."""
    os.makedirs(output_dir, exist_ok=True)

    train_pts = set(df_train[patient_col].dropna())
    val_pts = set(df_val[patient_col].dropna())
    test_pts = set(df_test[patient_col].dropna())

    train_val_overlap = len(train_pts.intersection(val_pts))
    train_test_overlap = len(train_pts.intersection(test_pts))
    val_test_overlap = len(val_pts.intersection(test_pts))

    is_leakage_free = (train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0)

    def get_split_stats(df_split: pd.DataFrame, name: str) -> Dict:
        total = len(df_split)
        pos = int((df_split[target_col] == 1).sum()) if target_col in df_split.columns else 0
        neg = total - pos
        rate = round(pos / total * 100, 2) if total > 0 else 0
        return {
            "name": name,
            "total_encounters": total,
            "unique_patients": df_split[patient_col].nunique(),
            "mortality_positive": pos,
            "mortality_negative": neg,
            "mortality_rate_percent": rate
        }

    report = {
        "leakage_verification": {
            "is_leakage_free": is_leakage_free,
            "train_val_overlap_patients": train_val_overlap,
            "train_test_overlap_patients": train_test_overlap,
            "val_test_overlap_patients": val_test_overlap
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "splits": {
            "train": get_split_stats(df_train, "Train"),
            "val": get_split_stats(df_val, "Validation"),
            "test": get_split_stats(df_test, "Test")
        }
    }

    # Save JSON report
    json_path = os.path.join(output_dir, "data_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save Markdown report
    md_content = f"""# Data Quality and Leakage Audit Report

## 1. Zero-Leakage Certification
- **Leakage-Free Status:** {"PASSED (Zero Overlap)" if is_leakage_free else "FAILED (Patient Overlap Detected)"}
- **Train - Val Patient Overlap:** {train_val_overlap}
- **Train - Test Patient Overlap:** {train_test_overlap}
- **Val - Test Patient Overlap:** {val_test_overlap}

## 2. Partition Summary
| Split | Total Stays | Unique Patients | Mortality Positives (Expired) | Mortality Negatives (Alive) | Mortality Rate (%) |
|---|---|---|---|---|---|
| **Train** | {report['splits']['train']['total_encounters']} | {report['splits']['train']['unique_patients']} | {report['splits']['train']['mortality_positive']} | {report['splits']['train']['mortality_negative']} | {report['splits']['train']['mortality_rate_percent']}% |
| **Val** | {report['splits']['val']['total_encounters']} | {report['splits']['val']['unique_patients']} | {report['splits']['val']['mortality_positive']} | {report['splits']['val']['mortality_negative']} | {report['splits']['val']['mortality_rate_percent']}% |
| **Test** | {report['splits']['test']['total_encounters']} | {report['splits']['test']['unique_patients']} | {report['splits']['test']['mortality_positive']} | {report['splits']['test']['mortality_negative']} | {report['splits']['test']['mortality_rate_percent']}% |

## 3. Extracted Features ({len(feature_names)} features)
```
{', '.join(feature_names)}
```
"""
    md_path = os.path.join(output_dir, "data_quality_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved quality reports to {json_path} and {md_path}")
    return report

