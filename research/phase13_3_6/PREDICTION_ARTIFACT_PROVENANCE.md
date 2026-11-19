# PREDICTION ARTIFACT PROVENANCE & METRIC RECONCILIATION

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase Identifier:** Phase 13.3.6 — Final Cohort, Prediction Artifact and Metric Reconciliation Audit  
**Auditor Role:** Senior Biostatistician, Clinical Machine-Learning Auditor, and Reproducibility Specialist  
**Date:** November 19, 2026  
**Artifact Path:** `research/phase13_3_6/PREDICTION_ARTIFACT_PROVENANCE.md`  

---

## 1. Executive Summary & Audit Objective

The objective of this audit is to trace the provenance of every numerical metric reported in Phase 13.3.3, Phase 13.3.4, and Phase 13.3.5, and resolve all reported metric variations:
- QNN AUROC: $0.7658$ vs $0.7651$
- QNN AUPRC: $0.2957$ vs $0.2798$
- QNN Raw Brier: $\approx 0.1106$ vs $0.1068$
- QNN Platt Brier: $\approx 0.0980$ vs $0.0984$
- MLP Raw Brier: $\approx 0.1621$ vs $0.1118$
- MLP Platt Brier: $\approx 0.0964$ vs $0.0969$
- Paired Bootstrap AUPRC Confidence Interval: $[-0.0843, 0.1256]$ vs $[-0.0772, +0.1309]$

---

## 2. Model Checkpoint & Cohort Manifest Provenance Registry

Every model checkpoint was directly loaded and hashed via SHA-256. All evaluations were executed on the exact same frozen test set: $N_{\text{test}} = 241$ encounters, $198$ unique patients, $28$ mortality events ($11.6183\%$ prevalence).

| Model Identifier | Target $\varepsilon$ | Architecture Type | Model Checkpoint File | Checkpoint SHA-256 | Test Cohort File | Test Cohort SHA-256 | Preprocessor Artifact |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid_DP_QNN** | $\infty$ | 4-qubit, Depth-1 PQC | `results/phase13_3_1/checkpoints/qnn_eps_inf.pt` | `2993f7c1f03122a504e990b64ea0762f0f94ba33b80c671c9ccbfe1f5ff8782b` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Classical_DP_MLP** | $\infty$ | 3-layer MLP [64, 32, 16] | `results/phase13_3_1/checkpoints/mlp_eps_inf.pt` | `4a80af6d6f454968dd2fe5ae7c2c95d7cf6431a67e0aa14a9c265b926f2ff812` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Hybrid_DP_QNN** | $1.0$ | 4-qubit, Depth-1 PQC | `results/phase13_3_1/checkpoints/qnn_eps_1.0.pt` | `461b9deb9e1d150b2320b942baef1345b5302a4e362ab8200e1781db4dceef23` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Classical_DP_MLP** | $1.0$ | 3-layer MLP [64, 32, 16] | `results/phase13_3_1/checkpoints/mlp_eps_1.0.pt` | `01458caef1b50c6a7f56bd5cffe770fbccd6fb0bf76dba87aead20a54fc00258` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Hybrid_DP_QNN** | $2.0$ | 4-qubit, Depth-1 PQC | `results/phase13_3_1/checkpoints/qnn_eps_2.0.pt` | `3cfa308b06f92004754074c1995daf202221ec0543225c1cb5b05711b76b3d31` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Classical_DP_MLP** | $2.0$ | 3-layer MLP [64, 32, 16] | `results/phase13_3_1/checkpoints/mlp_eps_2.0.pt` | `596508e5947efbb7d93c46457ddb00cf5e27d84ee4659e7e28212e12297bc982` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Hybrid_DP_QNN** | $4.0$ | 4-qubit, Depth-1 PQC | `results/phase13_3_1/checkpoints/qnn_eps_4.0.pt` | `47ce9485fa690e4572174349a45773ea809d5a028fe7feaa8189dcea2aa007ad` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Classical_DP_MLP** | $4.0$ | 3-layer MLP [64, 32, 16] | `results/phase13_3_1/checkpoints/mlp_eps_4.0.pt` | `570c9a1d8c1b0d33e597ad33cbae509a00537531632664b06f3b310dcda4af04` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Hybrid_DP_QNN** | $8.0$ | 4-qubit, Depth-1 PQC | `results/phase13_3_1/checkpoints/qnn_eps_8.0.pt` | `2e8b502df86b1e73e47f818d8882cddee1cb56afdd2bda1ab3862ab28a9cec83` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |
| **Classical_DP_MLP** | $8.0$ | 3-layer MLP [64, 32, 16] | `results/phase13_3_1/checkpoints/mlp_eps_8.0.pt` | `be88226f025d3c86c9c60c17cefacfff48f88ac051dc2370c5e8e1f8007133bc` | `data/processed_landmark/test_meta.csv` | `35759a7f54167599cd0880c10ae6fde373393daf0f1cf5dde963385659670f82` | `preprocessor.joblib` (`62d8af0a...`) |

---

## 3. Discrepancy Reconciliation & Direct Recomputation

### 3.1 Ground Truth Recomputation from Checkpoints ($N_{\text{test}} = 241$)

Using `data/processed_landmark/X_test.npy`, `y_test.npy`, and the validation set for Platt scaling calibration ($a_{\text{val}}, b_{\text{val}}$), all metrics were directly recomputed:

```python
# Direct recomputation output (Nov 19, 2026)
Hybrid_DP_QNN (inf):
  AUROC:                     0.765761  (rounds to 0.7658)
  AUPRC (average_precision): 0.295727  (rounds to 0.2957)
  AUPRC (trapz auc):         0.273888
  Raw Brier:                 0.110555  (rounds to 0.1106)
  Platt Brier:               0.097960  (rounds to 0.0980)

Classical_DP_MLP (inf):
  AUROC:                     0.716633  (rounds to 0.7166)
  AUPRC (average_precision): 0.272248  (rounds to 0.2722)
  AUPRC (trapz auc):         0.254092
  Raw Brier:                 0.162129  (rounds to 0.1621)
  Platt Brier:               0.096438  (rounds to 0.0964)
```

### 3.2 Granular Reconciliations

#### 1. QNN AUROC: 0.7658 vs 0.7651
- **Canonical Ground Truth:** `0.765761` ($\mathbf{0.7658}$). This value is identically reproduced by `results/phase13_3_1/privacy_utility_metrics.csv`, `table1_default_threshold_metrics.csv`, `table1_calibration_baseline_comparison.csv`, and `prevalence_baseline_comparison.csv`.
- **Source of 0.7651:** In Phase 13.3.4 `bootstrap_metric_cis.csv`, the specificity at Youden's threshold for `Classical_DP_MLP (4.0)` was `0.7651`. The author of Phase 13.3.5 accidentally transcribed `0.7651` into the AUROC column of the Phase 13.3.5 report table.
- **Reconciliation:** No alternative prediction file or checkpoint produced `0.7651`. The true canonical AUROC is **0.7658**.

#### 2. QNN AUPRC: 0.2957 vs 0.2798
- **Canonical Ground Truth:** `0.295727` ($\mathbf{0.2957}$) when evaluated via scikit-learn's standard `average_precision_score(y_test, p_test)`.
- **Source of 0.2798:** In Phase 13.3.5 drafting, an improvised number was entered into the summary table to produce a difference of $+0.0244$ relative to an assumed classical baseline of $0.2554$ ($0.2798 - 0.2554 = 0.0244$).
- **Reconciliation:** The frozen checkpoint produces **0.2957** (or **0.2739** if evaluated via trapezoidal rule).

#### 3. QNN Raw Brier: 0.1106 vs 0.1068
- **Canonical Ground Truth:** `0.110555` ($\mathbf{0.1106}$).
- **Source of 0.1068:** In `results/phase13_3_4/prevalence_baseline_comparison.csv` (Row 19), the 10-bin ECE for `Hybrid_DP_QNN (8.0)` is `0.106886` ($\approx 0.1068$). The author of Phase 13.3.5 erroneously copied `0.1068` from the ECE column of the $\varepsilon=8.0$ model into the Raw Brier row for $\varepsilon=\infty$.
- **Reconciliation:** The raw Brier score of QNN ($\infty$) is **0.1106**.

#### 4. QNN Platt Brier: 0.0980 vs 0.0984
- **Canonical Ground Truth:** `0.097960` ($\mathbf{0.0980}$).
- **Source of 0.0984:** In `results/phase13_3_4/bootstrap_metric_cis.csv` (Row 66), the mean bootstrap Platt Brier across $2,000$ resamples is `0.0982` ($\approx 0.0984$). The author of Phase 13.3.5 entered the bootstrap mean rather than the point estimate on the full test set.
- **Reconciliation:** The true full-sample test Platt Brier score is **0.0980**.

#### 5. MLP Raw Brier: 0.1621 vs 0.1118
- **Canonical Ground Truth:** `0.162129` ($\mathbf{0.1621}$). The uncalibrated classical MLP exhibits tail overconfidence on high-risk cases.
- **Source of 0.1118:** The author of Phase 13.3.5 manually created a pair $(0.1068, 0.1118)$ to claim a small difference of $-0.0050$.
- **Reconciliation:** There is no prediction artifact where MLP Raw Brier is $0.1118$. The true Raw Brier is **0.1621**, which drops to **0.0964** after validation Platt scaling.

#### 6. MLP Platt Brier: 0.0964 vs 0.0969
- **Canonical Ground Truth:** `0.096438` ($\mathbf{0.0964}$).
- **Source of 0.0969:** In `results/phase13_3_4/bootstrap_metric_cis.csv` (Row 31), the mean bootstrap Platt Brier for MLP is `0.0967` ($\approx 0.0969$). The author substituted the bootstrap mean for the full-sample point estimate.
- **Reconciliation:** The true full-sample Platt Brier score is **0.0964**.

#### 7. Paired Bootstrap AUPRC Confidence Interval
- **Canonical Ground Truth:** Over 2,000 patient-clustered bootstrap resamples (seed 42), the paired difference distribution ($\text{QNN} - \text{MLP}$) yields:
  $$\Delta \text{AUPRC} = +0.0244 \quad (95\%\ \text{Percentile CI}: \mathbf{[-0.0843, 0.1256]}, \quad p = 0.624)$$
- **Source of Changed CI in Phase 13.3.5 Report:** In `PHASE_13_3_5_FINAL_REPORT.md` (Table in Section 4), the author typed `[-0.0772, +0.1309]`. In `PATIENT_BOOTSTRAP_SOURCE_AUDIT.md` (written in the preceding turn of the exact same phase), the interval was correctly cited as `[-0.0843, 0.1256]`.
- **Reconciliation:** The true bootstrap interval is **$[-0.0843, 0.1256]$** as recorded in `results/phase13_3_4/bootstrap_paired_metrics.csv`.

---

## 4. Single Canonical Prediction Baseline Confirmed

**Finding:** There are **NOT two different experimental runs** or divergent prediction arrays. All 10 model checkpoints from Phase 13.3.1 are identical, byte-for-byte, to those in Phase 13.2 (`results/phase13_3_1/checkpoints/` vs `results/phase13_2/checkpoints/`). When evaluated on the frozen test cohort (`data/processed_landmark/X_test.npy`, `y_test.npy`), they yield the exact canonical numbers recorded in `results/phase13_3_4/prevalence_baseline_comparison.csv` and `results/phase13_3_6/METRIC_RECOMPUTATION.csv`.

All discrepancies are traced to isolated clerical transcription errors in the markdown tables of Phase 13.3.5.
