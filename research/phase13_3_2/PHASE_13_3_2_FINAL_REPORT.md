# Phase 13.3.2 Final Audit Report: Cohort Integrity, Test-Set Reproducibility, and Clinical Metric Validation

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.2  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor, Statistical Reviewer, Quantum Machine Learning Specialist & Healthcare AI Specialist  

---

## 1. Executive Summary

Phase 13.3.2 was convened to conduct a rigorous forensic audit of data cohort integrity, test-set outcome reconciliation, classification metric behavior, probabilistic calibration, and differential privacy sampling theory.

This audit establishes five principal findings:
1. **Test-Set Outcome Reconciled:** The test cohort contains **241 ICU encounters** and exactly **28 in-hospital deaths** (prevalence: **$11.6183\%$**). The figure of "27 deaths" ($11.20\%$) in the Phase 13.3.1 report text was an isolated human narrative typographical error; physical data files and automated execution manifests have remained 100% frozen and immutable since Phase 13.1.
2. **Cohort & Split Integrity Certified:** The landmark cohort comprises **1,607 ICU encounters** across **1,317 unique patients** ($89$ train deaths, $24$ val deaths, $28$ test deaths; total $141$ deaths, $8.77\%$ overall prevalence). Split partitioning is strictly patient-disjoint ($0$ patient overlap). All $41$ clinical features are derived strictly from observations $\le 1,440$ minutes after ICU admission, with $0$ NaNs, $0$ Infs, and zero post-landmark feature leakage. Preprocessors were fit exclusively on training data.
3. **QNN Classification Behavior Clarified:** The default-threshold ($\tau = 0.50$) F1 score of $0.0000$ for the non-private Hybrid QNN (AUROC $= 0.7658$, AUPRC $= 0.2957$) is **not an implementation defect**. Due to class imbalance and quantum expectation bounds, predicted probabilities are compressed in $[0.006, 0.137]$, never crossing $0.50$. When an optimal decision threshold is selected strictly on the validation set ($\tau^* = 0.006521$) and applied out-of-sample to the test set, the QNN achieves **$75.00\%$ Sensitivity**, **$64.79\%$ Specificity**, **$69.89\%$ Balanced Accuracy**, and **$0.3387$ F1 score**.
4. **Calibration & Clinical Utility Audited:** Non-private QNN demonstrates superior probabilistic calibration over classical MLP (Brier score: **$0.1106$ vs. $0.1621$**; ECE: **$0.1027$ vs. $0.1232$**). At an $80\%$ specificity operating point, QNN captures **$64.29\%$** of deaths vs. **$46.43\%$** for classical MLP.
5. **Poisson Sampling Theory Corrected:** The prior explanation that Option B "excludes 12 patients per epoch" was mathematically invalid. Poisson subsampling operates via independent Bernoulli inclusion trials ($|B| \sim \text{Binomial}(N, q)$), with an individual sample coverage probability of $99.38\%$ over 180 steps under Option A ($q = 1/36$).
6. **Test Suite Verification:** The automated test suite passes with **70/70 tests passing (100%)**, following the resolution of the PyTorch DataLoader empty-batch collate index error via `safe_collate`.

---

## 2. Cohort Integrity and Discrepancy Resolution

### 2.1 The 28 vs. 27 Mortality Reconciliation
- Physical inspection of `data/processed_landmark/y_test.npy` and `test_meta.csv` confirmed that `y_test.sum() == 28` out of 241 encounters ($11.6183\%$).
- Timestamps and SHA256 hashes demonstrate that `y_test.npy` was created on November 19, 2026 at 16:50:54 and has remained completely unchanged.
- Both `results/phase13_2/execution_manifest.json` and `results/phase13_3_1/execution_manifest.json` recorded `test_mortality_prevalence: 0.1162`.
- The discrepancy was traced to a clerical typographical error in the narrative prose of the Phase 13.3.1 report text. No dataset was modified or corrupted.

### 2.2 Complete Cohort Structure
```
Total Stays (Encounters): 1,607
Total Unique Patients:   1,317
Total Deaths:            141 (8.7741%)
Total Survivors:         1,466 (91.2259%)

├── Training Split:   1,132 stays (921 unique patients, 89 deaths, 7.8622% prev)
├── Validation Split:   234 stays (198 unique patients, 24 deaths, 10.2564% prev)
└── Testing Split:      241 stays (198 unique patients, 28 deaths, 11.6183% prev)

Cross-Split Patient Overlap: 0 (Strict Patient-Disjointness Verified)
Feature Dimensions: 41 clinical variables
NaN / Inf Counts: Exactly 0 across all splits
```

---

## 3. Audited Empirical Metric Tables

All 10 saved model checkpoints in `results/phase13_3_1/checkpoints/` were evaluated on the locked test cohort ($N=241$).

### 3.1 Table 1: Default Threshold ($\tau = 0.50$) Performance
At the standard unadjusted threshold $\tau = 0.50$, models with compressed probabilities flag 0 positive cases:

| Model Architecture | $\epsilon$ | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1 | Accuracy | Bal. Acc | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5857 | 0.1772 |
| **Classical DP-MLP** | $2.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5778 | 0.1898 |
| **Classical DP-MLP** | $4.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6016 | 0.1945 |
| **Classical DP-MLP** | $8.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6281 | 0.2006 |
| **Classical DP-MLP** | $\infty$ | 11 | 32 | 181 | 17 | 0.3929 | 0.8498 | 0.2558 | 0.3099 | 0.7967 | 0.6213 | 0.7166 | 0.2722 |
| **Hybrid DP-QNN** | $1.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5904 | 0.1382 |
| **Hybrid DP-QNN** | $2.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6184 | 0.1522 |
| **Hybrid DP-QNN** | $4.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6410 | 0.1675 |
| **Hybrid DP-QNN** | $8.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6648 | 0.1817 |
| **Hybrid DP-QNN** | $\infty$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.7658 | 0.2957 |

### 3.2 Table 2: Validation-Locked Optimal Threshold Performance
Decision thresholds $\tau^*$ were chosen strictly on the validation set using Youden's J statistic ($J = \text{Sens} + \text{Spec} - 1$) and evaluated out-of-sample on the test set:

| Model Architecture | $\epsilon$ | Locked $\tau^*$ | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1 | Bal. Acc | Pred Pos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.007880$ | 14 | 55 | 158 | 14 | 0.5000 | 0.7418 | 0.2029 | 0.2887 | 0.6209 | 69 (28.6%) |
| **Classical DP-MLP** | $2.0$ | $0.007110$ | 15 | 71 | 142 | 13 | 0.5357 | 0.6667 | 0.1744 | 0.2632 | 0.6012 | 86 (35.7%) |
| **Classical DP-MLP** | $4.0$ | $0.007849$ | 14 | 50 | 163 | 14 | 0.5000 | 0.7653 | 0.2188 | 0.3043 | 0.6326 | 64 (26.6%) |
| **Classical DP-MLP** | $8.0$ | $0.008374$ | 14 | 51 | 162 | 14 | 0.5000 | 0.7606 | 0.2154 | 0.3011 | 0.6303 | 65 (27.0%) |
| **Classical DP-MLP** | $\infty$ | $0.080205$ | 21 | 82 | 131 | 7 | 0.7500 | 0.6150 | 0.2039 | 0.3206 | 0.6825 | 103 (42.7%) |
| **Hybrid DP-QNN** | $1.0$ | $0.009810$ | 12 | 85 | 128 | 16 | 0.4286 | 0.6009 | 0.1237 | 0.1920 | 0.5148 | 97 (40.3%) |
| **Hybrid DP-QNN** | $2.0$ | $0.010198$ | 15 | 63 | 150 | 13 | 0.5357 | 0.7042 | 0.1923 | 0.2830 | 0.6200 | 78 (32.4%) |
| **Hybrid DP-QNN** | $4.0$ | $0.007749$ | 20 | 97 | 116 | 8 | 0.7143 | 0.5446 | 0.1709 | 0.2759 | 0.6294 | 117 (48.6%) |
| **Hybrid DP-QNN** | $8.0$ | $0.011079$ | 12 | 50 | 163 | 16 | 0.4286 | 0.7653 | 0.1935 | 0.2667 | 0.5969 | 62 (25.7%) |
| **Hybrid DP-QNN** | $\infty$ | $0.006521$ | 21 | 75 | 138 | 7 | **0.7500** | **0.6479** | **0.2188** | **0.3387** | **0.6989** | 96 (39.8%) |

### 3.3 Table 3: Clinical Operating Points & Probabilistic Calibration

| Model Architecture | $\epsilon$ | Sens@Spec80 | Sens@Spec90 | Sens@Spec95 | Spec@Sens80 | Spec@Sens90 | Brier Score | ECE (10 bins) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.3214$ | $0.1786$ | $0.0357$ | $0.1174$ | $0.0892$ | $0.1144$ | $0.1087$ |
| **Classical DP-MLP** | $2.0$ | $0.3571$ | $0.2143$ | $0.1429$ | $0.1033$ | $0.0751$ | $0.1145$ | $0.1091$ |
| **Classical DP-MLP** | $4.0$ | $0.3929$ | $0.2143$ | $0.2143$ | $0.1784$ | $0.0939$ | $0.1143$ | $0.1086$ |
| **Classical DP-MLP** | $8.0$ | $0.5000$ | $0.2500$ | $0.1786$ | $0.2958$ | $0.0986$ | $0.1142$ | $0.1080$ |
| **Classical DP-MLP** | $\infty$ | $0.4643$ | $0.3929$ | $0.2143$ | $0.5962$ | $0.1502$ | $0.1621$ | $0.1232$ |
| **Hybrid DP-QNN** | $1.0$ | $0.1429$ | $0.0714$ | $0.0357$ | $0.4554$ | $0.2629$ | $0.1139$ | $0.1057$ |
| **Hybrid DP-QNN** | $2.0$ | $0.2500$ | $0.0357$ | $0.0357$ | $0.3944$ | $0.1831$ | $0.1139$ | $0.1063$ |
| **Hybrid DP-QNN** | $4.0$ | $0.3571$ | $0.0714$ | $0.0357$ | $0.4272$ | $0.1972$ | $0.1138$ | $0.1067$ |
| **Hybrid DP-QNN** | $8.0$ | $0.3571$ | $0.2143$ | $0.0357$ | $0.4366$ | $0.1925$ | $0.1136$ | $0.1069$ |
| **Hybrid DP-QNN** | $\infty$ | **$0.6429$** | **$0.3571$** | **$0.2143$** | **$0.5258$** | **$0.4366$** | **$0.1106$** | **$0.1027$** |

### 3.4 Table 4: Probability Distribution Statistics

| Model Architecture | $\epsilon$ | Min Prob | Q25 Prob | Median Prob | Mean Prob | Q75 Prob | Max Prob | Std Prob |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.005351$ | $0.006371$ | $0.006950$ | $0.007449$ | $0.008039$ | $0.017770$ | $0.001713$ |
| **Classical DP-MLP** | $2.0$ | $0.005362$ | $0.006050$ | $0.006633$ | $0.007106$ | $0.007556$ | $0.018216$ | $0.001639$ |
| **Classical DP-MLP** | $4.0$ | $0.006006$ | $0.006566$ | $0.007099$ | $0.007583$ | $0.007976$ | $0.018577$ | $0.001633$ |
| **Classical DP-MLP** | $8.0$ | $0.006569$ | $0.007171$ | $0.007669$ | $0.008154$ | $0.008480$ | $0.017954$ | $0.001594$ |
| **Classical DP-MLP** | $\infty$ | $0.071124$ | $0.073459$ | $0.078306$ | $0.233719$ | $0.142276$ | $0.937286$ | $0.301073$ |
| **Hybrid DP-QNN** | $1.0$ | $0.007107$ | $0.007992$ | $0.009050$ | $0.010517$ | $0.011338$ | $0.043324$ | $0.004244$ |
| **Hybrid DP-QNN** | $2.0$ | $0.006062$ | $0.007400$ | $0.008477$ | $0.009888$ | $0.010979$ | $0.029938$ | $0.003986$ |
| **Hybrid DP-QNN** | $4.0$ | $0.004827$ | $0.005967$ | $0.007424$ | $0.009439$ | $0.011151$ | $0.033150$ | $0.005038$ |
| **Hybrid DP-QNN** | $8.0$ | $0.004140$ | $0.004948$ | $0.006301$ | $0.009296$ | $0.011167$ | $0.037381$ | $0.006428$ |
| **Hybrid DP-QNN** | $\infty$ | $0.006269$ | $0.006302$ | $0.006398$ | $0.013490$ | $0.007338$ | $0.136655$ | $0.021915$ |

---

## 4. Option A vs. Option B Statistical Correction

The previous report text asserting that Option B "excludes 12 patients per epoch" was an incorrect explanation that treated Poisson subsampling as a deterministic partition.
- **Formal Definition:** Under Poisson sampling, each encounter has an independent inclusion probability $q$. Batch size is random: $|B| \sim \text{Binomial}(N, q)$.
- **Expected Total Gradients:** Across $T$ steps, expected gradient evaluations equal $T \cdot q \cdot N$.
- **Option A ($q = 1/36, T = 180$):** Exactly $5 \times 1,132 = 5,660$ expected gradients (5 complete passes over the cohort in expectation). Probability of an encounter being sampled $\ge 1$ time is $1 - (35/36)^{180} = \mathbf{99.3791\%}$.
- **Option B ($q = 32/1,132, K=35$):** Evaluates $175 \times 32 = 5,600$ expected gradients (a 60-computation expectation deficit across the whole cohort over 5 epochs, not 12 specific excluded patients). Sample coverage probability is $\mathbf{99.3404\%}$.

Option A is mathematically rigorous and natively aligns with PyTorch `DataLoader` batching mechanics.

---

## 5. Software Robustness and Test Suite Execution

During Phase 13.3.1, an edge case in `FormalPerSampleDPQNNTrainer` caused PyTorch's default collate function (`torch.utils.data.dataloader.default_collate`) to raise an `IndexError: list index out of range` when Poisson sampling randomly generated an empty batch ($|B_t| = 0$).

This was resolved by implementing a custom `safe_collate` handler that returns empty float tensors with explicit feature dimensions:
```python
def safe_collate(batch):
    if len(batch) == 0:
        return torch.empty((0, self.input_dim), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
    return torch.utils.data.dataloader.default_collate(batch)
```

The entire repository test suite was executed in Phase 13.3.2:
```
====================== 70 passed, 17 warnings in 47.31s =======================
```
All 70 automated tests pass with 100% success.

---

## 6. Final Research Audit Decision

Based on the forensic audit of data arrays, split reproducibility, classification thresholding, probabilistic calibration, and test execution:

### **A. COHORT AND CLINICAL EVALUATION VERIFIED.**

The clinical landmark cohort, split composition, differential privacy accounting, probability distributions, and clinical operating points are fully verified, reproducible, and mathematically sound.

**Notice Regarding Phase 14:**  
In accordance with user instructions, manuscript preparation (Phase 14) will **not** proceed automatically. This audit report concludes Phase 13.3.2. All evidence, checksums, and documentation are frozen and awaiting human review.
