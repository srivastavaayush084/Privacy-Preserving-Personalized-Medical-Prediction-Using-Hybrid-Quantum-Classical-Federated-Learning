# Classical Baseline Models Performance Report

## Task: In-Hospital Mortality Prediction (First 24h Clinical Features)
- **Cohort Size:** Train = 1712 | Val = 387 | Test = 393
- **Feature Dimension:** 41 standardized clinical features
- **Test Set Mortality Prevalence:** 10.18% (40 positive / 393 total)

## Evaluation Benchmark Table

| Model | Test AUROC | Test PR-AUC | Test Accuracy | Test F1 | Sensitivity | Specificity | Brier Score | ECE |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.817** | 0.3569 | 0.7328 | 0.3713 | 0.775 | 0.728 | 0.1948 | 0.2879 |
| **Random Forest** | **0.8411** | 0.3403 | 0.855 | 0.4 | 0.475 | 0.898 | 0.1216 | 0.2013 |
| **Support Vector Machine** | **0.7948** | 0.2508 | 0.8982 | 0.0 | 0.0 | 1.0 | 0.0842 | 0.0272 |
| **Multi-Layer Perceptron** | **0.7944** | 0.3295 | 0.715 | 0.3488 | 0.75 | 0.711 | 0.1804 | 0.2203 |
| **XGBoost** | **0.8445** | 0.3849 | 0.799 | 0.3577 | 0.55 | 0.8272 | 0.1286 | 0.19 |

## Key Methodological Insights
1. **Tree Ensembles & Regularization:** Random Forest provides robust discrimination across physiological variables with competitive Brier scores.
2. **Clinical Trade-offs:** Balanced class-weighting elevates sensitivity on high-risk mortality cases, a crucial healthcare safety objective.
3. **Calibration:** ECE assesses whether predicted risks reflect true observation rates, serving as a direct benchmark for the upcoming hybrid quantum-classical calibration comparisons.
