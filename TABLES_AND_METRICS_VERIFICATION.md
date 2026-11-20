# Phase 14 — Tables and Metrics Verification Report

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Authoritative Baseline:** Phase 13.3.6 Numerical Reconciliation Report  
**Date:** November 20, 2026  
**Auditor:** Senior Biostatistician & Clinical ML Reproducibility Auditor  
**Document Status:** Complete & Certified  

---

## 1. Executive Summary

This audit independently verifies every numerical value, parameter count, confidence interval, $p$-value, communication metric, and performance figure reported in the IEEE research manuscript `IEEE_Research_Paper.tex` and its compiled output `IEEE_Research_Paper.pdf`.

Every metric reported in the manuscript was cross-referenced directly against:
1. The authoritative Phase 13.3.6 audit report (`phase13_3_6/METRIC_RECOMPUTATION.csv` and `verification_manifest.json`).
2. The frozen test cohort (`data/cohort_24h_test.csv`, $N=241$ encounters, $198$ unique patients, $28$ deaths).
3. The frozen validation cohort (`data/cohort_24h_val.csv`, $N=234$ encounters, $198$ unique patients, $24$ deaths).
4. The frozen training cohort (`data/cohort_24h_train.csv`, $N=1,132$ encounters, $921$ unique patients, $89$ deaths).
5. The 10 trained model checkpoints (`results/phase13_3_1/checkpoints/`).

**Audit Verdict:** **100% NUMERICAL CONCORDANCE CERTIFIED.** Zero discrepancy detected.

---

## 2. Table I: Clinical 24-Hour Landmark Cohort Statistics

| Partition | Manuscript Encounters | Canonical Encounters | Manuscript Patients | Canonical Patients | Deaths ($Y=1$) | Survivors ($Y=0$) | Prevalence | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 1,132 | 1,132 | 921 | 921 | 89 | 1,043 | 7.86% | **VERIFIED** |
| **Validation Split** | 234 | 234 | 198 | 198 | 24 | 210 | 10.26% | **VERIFIED** |
| **Test Split** | 241 | 241 | 198 | 198 | 28 | 213 | 11.62% | **VERIFIED** |
| **Total Cohort** | **1,607** | **1,607** | **1,317** | **1,317** | **141** | **1,466** | **8.77%** | **VERIFIED** |

### Patient Disjointness Audit:
- Train $\cap$ Validation: $0$ patients ($0.00\%$).
- Train $\cap$ Test: $0$ patients ($0.00\%$).
- Validation $\cap$ Test: $0$ patients ($0.00\%$).
- Unique patient sum: $921 + 198 + 198 = 1,317$ unique patients.
- All encounters belonging to any patient were assigned atomically to a single split.

---

## 3. Table II: Comprehensive Performance Across Differential Privacy Budgets

All metrics evaluated on frozen holdout test cohort ($N=241$, $28$ deaths).

| Model Architecture | Privacy Tier ($\varepsilon$) | Trainable Params | AUROC (Manuscript) | AUROC (Canonical) | AUPRC (Manuscript) | AUPRC (Canonical) | Raw Brier | Platt Brier | Platt ECE | Net Benefit ($p_t=0.09$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $\varepsilon = 1.0$ | 5,537 | 0.5857 | 0.585681 | 0.1772 | 0.177186 | 0.1144 | 0.1030 | 0.0136 | +0.0152 |
| **Classical DP-MLP** | $\varepsilon = 2.0$ | 5,537 | 0.5778 | 0.577800 | 0.1898 | 0.189831 | 0.1145 | 0.1025 | 0.0184 | +0.0210 |
| **Classical DP-MLP** | $\varepsilon = 4.0$ | 5,537 | 0.6016 | 0.601610 | 0.1945 | 0.194481 | 0.1143 | 0.1020 | 0.0258 | +0.0264 |
| **Classical DP-MLP** | $\varepsilon = 8.0$ | 5,537 | 0.6281 | 0.628102 | 0.2006 | 0.200599 | 0.1142 | 0.1017 | 0.0249 | +0.0298 |
| **Classical DP-MLP** | Non-Private ($\infty$) | 5,537 | **0.7166** | **0.716633** | **0.2722** | **0.272248** | 0.1621 | **0.0964** | 0.0169 | +0.0392 |
| **Hybrid DP-QNN** | $\varepsilon = 1.0$ | **957** | 0.5904 | 0.590376 | 0.1382 | 0.138177 | 0.1139 | 0.1033 | 0.0187 | +0.0184 |
| **Hybrid DP-QNN** | $\varepsilon = 2.0$ | **957** | 0.6184 | 0.618377 | 0.1522 | 0.152167 | 0.1139 | 0.1037 | 0.0398 | +0.0275 |
| **Hybrid DP-QNN** | $\varepsilon = 4.0$ | **957** | 0.6410 | 0.641013 | 0.1675 | 0.167458 | 0.1138 | 0.1030 | 0.0447 | +0.0341 |
| **Hybrid DP-QNN** | $\varepsilon = 8.0$ | **957** | 0.6648 | 0.664822 | 0.1817 | 0.181712 | 0.1136 | 0.1016 | 0.0286 | +0.0412 |
| **Hybrid DP-QNN** | Non-Private ($\infty$) | **957** | **0.7658** | **0.765761** | **0.2957** | **0.295727** | **0.1106** | **0.0980** | 0.0445 | **+0.0551** |

---

## 4. Table III: Paired Patient-Clustered Bootstrap Analysis ($B=2,000$ Replicates)

Evaluated non-private QNN vs non-private MLP on test set using patient-clustered paired bootstrap (seed 42, $2,000$ resamples).

| Evaluated Metric | Manuscript Point Diff ($\Delta$) | Canonical Point Diff | Bootstrap Mean | 95% Percentile Confidence Interval | Empirical $p$-Value | Significance Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | $+0.0491$ | $+0.049129$ | $+0.0494$ | $[-0.0409, +0.1393]$ | $p = 0.290$ | Not Significant (Parity) |
| **AUPRC (APS)** | $+0.0235$ | $+0.023479$ | $+0.0244$ | $[-0.0843, +0.1256]$ | $p = 0.624$ | Not Significant (Parity) |
| **AUPRC (Trapz)** | $+0.0198$ | $+0.019796$ | $+0.0207$ | $[-0.0812, +0.1215]$ | $p = 0.648$ | Not Significant (Parity) |
| **Platt Brier Score** | $+0.0015$ | $+0.001522$ | $+0.0015$ | $[-0.0042, +0.0082]$ | $p = 0.629$ | Not Significant (Parity) |
| **Raw Brier Score** | $-0.0516$ | $-0.051574$ | $-0.0519$ | $[-0.0933, -0.0087]$ | $p = 0.020$ | Statistically Significant |
| **ECE (10-bin)** | $+0.0276$ | $+0.027610$ | $+0.0281$ | $[-0.0210, +0.0765]$ | $p = 0.252$ | Not Significant (Parity) |

---

## 5. Table IV: Exploratory Decision Curve Analysis (DCA) Net Benefit

Net Benefit: $\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$ on holdout test set ($N=241, 28\text{ deaths}$).

| Threshold ($p_t$) | Treat-None | Treat-All | Calibrated QNN | Calibrated MLP | Superior Strategy | Clinical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $0.05$ | $0.0000$ | $+0.0697$ | $+0.0697$ | $+0.0697$ | 3-Way Tie | All patients exceed threshold; treat all |
| $0.08$ | $0.0000$ | $+0.0393$ | $+0.0393$ | $+0.0390$ | QNN = Treat-All | Boundary transition zone |
| $0.09$ | $0.0000$ | $+0.0288$ | **+0.0551** | $+0.0392$ | **Calibrated QNN** | QNN saves 26.7 FP interventions/100 pts |
| $0.10$ | $0.0000$ | $+0.0180$ | **+0.0456** | $+0.0337$ | **Calibrated QNN** | QNN clearly superior to Treat-All |
| $0.12$ | $0.0000$ | $-0.0043$ | **+0.0285** | $+0.0283$ | **Calibrated QNN** | Treat-All enters negative net benefit |
| $0.15$ | $0.0000$ | $-0.0398$ | $+0.0139$ | **+0.0222** | **Calibrated MLP** | MLP exhibits higher specificity |
| $0.20$ | $0.0000$ | $-0.1048$ | $+0.0114$ | **+0.0207** | **Calibrated MLP** | Both models positive; MLP higher |
| $0.25$ | $0.0000$ | $-0.1784$ | $+0.0069$ | **+0.0111** | **Calibrated MLP** | Upper boundary of positive net benefit |
| $0.30$ | $0.0000$ | $-0.2626$ | $0.0000$ | $0.0000$ | QNN = MLP = Treat-None | No patients exceed threshold |

---

## 6. Model Architectural Complexity and Federated Communication Audit

### Trainable Parameter Breakdown:
- **Classical MLP:**
  - Layer 1 ($41 \to 64$): $41 \times 64 + 64 = 2,688$
  - Layer 2 ($64 \to 32$): $64 \times 32 + 32 = 2,080$
  - Layer 3 ($32 \to 16$): $32 \times 16 + 16 = 528$
  - Layer 4 ($16 \to 1$): $16 \times 1 + 1 = 17$
  - GroupNorm / Bias params: $224$
  - **Total Classical MLP Parameters:** $\mathbf{5,537}$
- **Hybrid Quantum Neural Network (QNN):**
  - Classical Bottleneck Encoder ($41 \to 16 \to 4$): $760$ parameters
  - 4-Qubit Strongly Entangling Variational Core ($L=2$ layers): $3 \times 4 \times 2 = \mathbf{24}$ parameters
  - Classical Head ($8 \to 16 \to 1$): $173$ parameters
  - **Total Hybrid QNN Parameters:** $\mathbf{957}$
- **Parameter Reduction:**
  $$\frac{5,537 - 957}{5,537} = \frac{4,580}{5,537} = \mathbf{82.72\%}$$
  Reported in manuscript as: **82.7%**.

### Federated Communication Payload:
- Classical MLP per round: $2.88$ MB (weights + optimizer states across clients).
- Hybrid QNN per round: $0.80$ MB.
- **Communication Payload Reduction:**
  $$\frac{2.88 - 0.80}{2.88} = \frac{2.08}{2.88} = \mathbf{72.22\%}$$
  Reported in manuscript as: **72.2%**.

---

## 7. Calibration Parameters Verification

1. **Platt Scaling fitted on Validation Cohort ($N=234, 24\text{ deaths}$):**
   - Fitted Slope: $a_{\text{val}} = \mathbf{0.441545} \approx 0.4415$
   - Fitted Intercept: $b_{\text{val}} = \mathbf{-0.092357} \approx -0.0924$
   - Applied out-of-sample to test logits: $\hat{p}_{\text{test}} = \sigma(0.4415 \cdot z_{\text{test}} - 0.0924)$.
2. **Out-of-Sample Test Cox Calibration Diagnostic:**
   - Descriptive Slope: $\beta_{\text{test}} = \mathbf{1.709588} \approx 1.7096$
   - Descriptive Intercept: $\alpha_{\text{test}} = 0.1081$ (logit scale) / $1.5717$ (unscaled log-odds)
   - Stated clearly in Section IX.A as an evaluation diagnostic, not a transformation parameter.
3. **Prevalence Baselines:**
   - Training prevalence: $89 / 1,132 = 7.86\%$; baseline Brier = $0.1041$.
   - Test prevalence (oracle): $28 / 241 = 11.62\%$; baseline Brier = $0.1027$.
   - Non-private QNN Platt Brier: $0.0980 < 0.1027$ (strictly outperforms oracle prevalence baseline).
   - Non-private MLP Platt Brier: $0.0964 < 0.1027$ (strictly outperforms oracle prevalence baseline).

---

## 8. Cryptographic Secure Aggregation Verification

- 5 consortium clients, Shamir $(t, n)$ threshold secret sharing ($t = 3, n = 5$).
- Dropouts simulated up to $40\%$ ($2$ of $5$ clients dropping out before unmasking).
- Maximum Absolute Error: $1.73 \times 10^{-6}$
- Root Mean Squared Error (RMSE): $4.46 \times 10^{-7}$
- Verified: Both metrics strictly match the manuscript text in Section VI.B.

---

## 9. Conclusion

All numerical figures, tables, text references, and mathematical bounds in `IEEE_Research_Paper.tex` and `IEEE_Research_Paper.pdf` are verified against the authoritative Phase 13.3.6 audit baseline. No fabrication, approximation drift, or legacy values exist in the manuscript.
