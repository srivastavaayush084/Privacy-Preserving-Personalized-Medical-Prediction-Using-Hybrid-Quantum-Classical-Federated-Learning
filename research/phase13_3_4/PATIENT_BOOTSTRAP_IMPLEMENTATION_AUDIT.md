# PATIENT-CLUSTERED BOOTSTRAP IMPLEMENTATION AUDIT & PAIRED STATISTICAL VERIFICATION

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.4 — Final Statistical Audit, Calibration Validation and Decision-Curve Correction  
**Author:** Senior Medical AI Researcher, Biostatistician & DP Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_4/PATIENT_BOOTSTRAP_IMPLEMENTATION_AUDIT.md`  

---

## 1. Executive Summary

In observational ICU cohorts, individual patients frequently undergo multiple ICU admissions during a hospital stay or across time. When evaluating predictive algorithms, treating each encounter as an independent observation introduces **intra-cluster correlation**, artificially compressing standard errors and yielding overly narrow confidence intervals.

In Phase 13.3.3 and Phase 13.3.4, an independent biostatistical audit verified the implementation of a rigorous **patient-clustered bootstrap ($B = 2,000$ replicates)**. This report documents the line-by-line audit against the **seven formal criteria** for clustered clinical resampling, presents the verified confidence intervals, and confirms the paired hypothesis testing results comparing the hybrid QNN against the classical MLP.

---

## 2. Formal Audit Against Seven Clustered Bootstrap Criteria

| Criterion # | Methodological Requirement | Implementation Verification & Audit Evidence | Audit Finding |
| :---: | :--- | :--- | :---: |
| **1** | **Resampling Unit is `uniquepid`** | Resampling draws from `np.unique(patient_ids_test)` ($198$ unique patients in the 241-encounter test cohort), not encounter rows. | **PASS** |
| **2** | **Multi-Stay Clusters Kept Intact** | All encounters corresponding to a sampled patient are extracted via `pt_to_indices[pt]`, preserving correlation structure intact. | **PASS** |
| **3** | **Paired Quantum-Classical Draws** | For each bootstrap replicate $b$, both the hybrid QNN and classical MLP predictions are evaluated on the exact same row indices `sample_indices`. | **PASS** |
| **4** | **Duplicate Patient Independence** | When a patient is sampled multiple times with replacement, their encounter indices are concatenated repeatedly, representing independent cluster draws. | **PASS** |
| **5** | **Single-Class Guard Handling** | A guard `if len(np.unique(y_b)) < 2: continue` prevents degenerate ROC/PRC calculations. In $B=2,000$ resamples (seed 42), **zero** resamples had 0 deaths. | **PASS** |
| **6** | **Empirical CIs & Two-Sided P-Values** | 95% CIs are derived as the 2.5th and 97.5th percentiles of the replicate distribution. Empirical two-sided p-values: $2 \min(\mathbb{P}(\Delta \le 0), \mathbb{P}(\Delta \ge 0))$. | **PASS** |
| **7** | **Zero Test-Set Refitting** | All Platt calibrators $(a, b)$ and decision thresholds $\tau$ are frozen strictly from the validation split. Zero parameters are re-estimated on test resamples. | **PASS** |

---

## 3. Paired Quantum-Classical Hypothesis Tests ($B = 2,000$ Replicates)

To evaluate whether the hybrid QNN exhibits statistically significant superiority over the classical baseline, paired difference distributions were computed across all 2,000 patient-clustered replicates:
$$\Delta \mathcal{M}^{(b)} = \mathcal{M}_{\text{QNN}}^{(b)} - \mathcal{M}_{\text{MLP}}^{(b)}$$
Evaluated for the non-private baseline ($\varepsilon = \infty$) on the frozen test set ($N=241, 28\text{ deaths}$):

| Difference Metric | Mean Difference | Median Difference | 95% Confidence Interval | Empirical Two-Sided $p$-Value | Statistically Significant ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Delta$ AUROC** | **+0.0494** | +0.0486 | **[-0.0409, 0.1393]** | **$p = 0.290$** | **No (Not Significant)** |
| **$\Delta$ AUPRC** | **+0.0244** | +0.0237 | **[-0.0843, 0.1256]** | **$p = 0.624$** | **No (Not Significant)** |
| **$\Delta$ Brier (Platt)** | **+0.0015** | +0.0015 | **[-0.0042, 0.0082]** | **$p = 0.629$** | **No (Not Significant)** |
| **$\Delta$ Brier (Raw)** | -0.0519 | -0.0523 | [-0.0933, -0.0087] | $p = 0.020$ | Yes (MLP Tail Overconfidence) |
| **$\Delta$ Sensitivity (Youden)** | -0.0011 | 0.0000 | [-0.2000, 0.1905] | $p = 1.000$ | No (Identical Point: 75.0%) |
| **$\Delta$ Specificity (Youden)** | +0.0333 | +0.0330 | [-0.0193, 0.0876] | $p = 0.239$ | No (Not Significant) |
| **$\Delta$ F1 (Youden)** | +0.0174 | +0.0164 | [-0.0596, 0.0940] | $p = 0.667$ | No (Not Significant) |

### Key Statistical Takeaways:
1. **Confidence Intervals Span Zero:** For all primary discriminative and calibrated error metrics ($\Delta \text{AUROC}$, $\Delta \text{AUPRC}$, $\Delta \text{Brier (Platt)}$), the 95% confidence intervals cross zero symmetrically.
2. **P-Values Well Above 0.05:** With $p = 0.290$ for AUROC and $p = 0.624$ for AUPRC, there is no empirical evidence to reject the null hypothesis of parity between the hybrid quantum and classical architectures.
3. **Raw Brier Artifact:** The statistically significant difference in raw Brier ($\Delta = -0.0519, p = 0.020$) reflects unregularized overconfidence in the classical MLP rather than true quantum superiority. Once both models are properly calibrated via validation-fit Platt scaling, this difference evaporates completely ($\Delta = +0.0015, p = 0.629$).

---

## 4. Full Model Bootstrap Metric Distributions

The table below presents the patient-clustered bootstrap distributions (Mean, Median, 95% CI, and CI Width) across all 10 models evaluated in `results/phase13_3_4/bootstrap_metric_cis.csv`:

| Architecture | $\varepsilon$ | Metric | Mean | Median | 95% CI Lower | 95% CI Upper | CI Width |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Classical DP-MLP | $1.0$ | AUROC | 0.5898 | 0.5910 | 0.4682 | 0.7077 | 0.2395 |
| Classical DP-MLP | $1.0$ | AUPRC | 0.1873 | 0.1775 | 0.1018 | 0.3168 | 0.2150 |
| Classical DP-MLP | $1.0$ | Brier Platt | 0.1027 | 0.1024 | 0.0818 | 0.1264 | 0.0446 |
| Classical DP-MLP | $2.0$ | AUROC | 0.5960 | 0.5968 | 0.4776 | 0.7107 | 0.2331 |
| Classical DP-MLP | $2.0$ | AUPRC | 0.1989 | 0.1895 | 0.1066 | 0.3341 | 0.2275 |
| Classical DP-MLP | $2.0$ | Brier Platt | 0.1022 | 0.1020 | 0.0817 | 0.1258 | 0.0441 |
| Classical DP-MLP | $4.0$ | AUROC | 0.6508 | 0.6521 | 0.5367 | 0.7588 | 0.2221 |
| Classical DP-MLP | $4.0$ | AUPRC | 0.2359 | 0.2269 | 0.1245 | 0.3862 | 0.2617 |
| Classical DP-MLP | $4.0$ | Brier Platt | 0.1017 | 0.1016 | 0.0815 | 0.1248 | 0.0433 |
| Classical DP-MLP | $8.0$ | AUROC | 0.6722 | 0.6744 | 0.5645 | 0.7725 | 0.2080 |
| Classical DP-MLP | $8.0$ | AUPRC | 0.2520 | 0.2443 | 0.1337 | 0.4079 | 0.2742 |
| Classical DP-MLP | $8.0$ | Brier Platt | 0.1014 | 0.1012 | 0.0814 | 0.1244 | 0.0430 |
| **Classical DP-MLP** | $\infty$ | **AUROC** | **0.7175** | **0.7196** | **0.6124** | **0.8142** | **0.2018** |
| **Classical DP-MLP** | $\infty$ | **AUPRC** | **0.3015** | **0.2917** | **0.1610** | **0.4735** | **0.3125** |
| **Classical DP-MLP** | $\infty$ | **Brier Platt** | **0.0961** | **0.0957** | **0.0766** | **0.1189** | **0.0423** |
| Hybrid DP-QNN | $1.0$ | AUROC | 0.5480 | 0.5487 | 0.4287 | 0.6695 | 0.2408 |
| Hybrid DP-QNN | $1.0$ | AUPRC | 0.1558 | 0.1477 | 0.0882 | 0.2638 | 0.1756 |
| Hybrid DP-QNN | $1.0$ | Brier Platt | 0.1030 | 0.1028 | 0.0820 | 0.1264 | 0.0444 |
| Hybrid DP-QNN | $2.0$ | AUROC | 0.6406 | 0.6424 | 0.5286 | 0.7483 | 0.2197 |
| Hybrid DP-QNN | $2.0$ | AUPRC | 0.2131 | 0.2037 | 0.1171 | 0.3551 | 0.2380 |
| Hybrid DP-QNN | $2.0$ | Brier Platt | 0.1034 | 0.1031 | 0.0820 | 0.1274 | 0.0454 |
| Hybrid DP-QNN | $4.0$ | AUROC | 0.6577 | 0.6599 | 0.5498 | 0.7602 | 0.2104 |
| Hybrid DP-QNN | $4.0$ | AUPRC | 0.2372 | 0.2285 | 0.1260 | 0.3879 | 0.2619 |
| Hybrid DP-QNN | $4.0$ | Brier Platt | 0.1027 | 0.1024 | 0.0817 | 0.1263 | 0.0446 |
| Hybrid DP-QNN | $8.0$ | AUROC | 0.6468 | 0.6480 | 0.5367 | 0.7516 | 0.2149 |
| Hybrid DP-QNN | $8.0$ | AUPRC | 0.2443 | 0.2339 | 0.1299 | 0.3999 | 0.2700 |
| Hybrid DP-QNN | $8.0$ | Brier Platt | 0.1013 | 0.1010 | 0.0811 | 0.1246 | 0.0435 |
| **Hybrid DP-QNN** | $\infty$ | **AUROC** | **0.7669** | **0.7690** | **0.6728** | **0.8521** | **0.1793** |
| **Hybrid DP-QNN** | $\infty$ | **AUPRC** | **0.3259** | **0.3168** | **0.1729** | **0.5097** | **0.3368** |
| **Hybrid DP-QNN** | $\infty$ | **Brier Platt** | **0.0976** | **0.0973** | **0.0782** | **0.1197** | **0.0415** |

---

## 5. Audit Verdict & Methodological Certification

1. **Bootstrap Validity:** The patient-clustered bootstrap implementation in `experiments/phase13_3_4/run_phase13_3_4_audit.py` fully adheres to established biostatistical protocols for hierarchical/clustered clinical data.
2. **Confidence Interval Reporting:** All metric intervals in the manuscript must cite the 2.5th and 97.5th percentiles from this audit.
3. **Scientific Claim Constraint:** The empirical finding that $\Delta \text{AUROC} = +0.0494$ has $p = 0.290$ mathematically precludes any claim of quantum superiority or quantum advantage.
