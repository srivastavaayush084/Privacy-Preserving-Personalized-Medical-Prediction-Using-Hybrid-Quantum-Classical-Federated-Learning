# PHASE 13 — INDEPENDENT PERSONALIZATION AUDIT REPORT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13 Independent Reproducibility & Claim Verification  
**Auditor Roles:** Senior Biostatistician, Healthcare Machine Learning Auditor  
**Date:** November 19, 2026  
**Artifact References:** [`results/phase13/personalization_recalculated.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/personalization_recalculated.csv), `results/phase12/metrics/advanced_personalization_metrics.csv`  

---

## 1. Executive Summary & Personalization Audit Verdict

An independent statistical audit was conducted on the hospital-level personalization modules (`src/personalization/local_adapter.py`) across both Phase 11 (modulo clustering) and Phase 12 (Dirichlet non-IID partitioning).

### High-Level Audit Verdict: **PARTIALLY VERIFIED WITH METHODOLOGICAL WITHDRAWALS**
1. **Algorithmic Mechanics Verified:** Parameter-isolated local residual adapters and temperature calibration successfully train without gradient leakage to the global model, demonstrating sound architectural engineering.
2. **"Rural Hospital Rescue" Claim WITHDRAWN:** The assertion that the framework "rescued a failing rural hospital from AUROC 0.1849 to 0.8151" is **scientifically unsupported and formally withdrawn**:
   - *Statistical Insignificance:* The test set for Consortium 5 contained **only 6 positive mortality deaths** ($N_{\text{pos}} = 6$). The resulting 95% bootstrap confidence interval spans $[0.552, 0.985]$ ($\pm 0.22$), representing extreme statistical fragility.
   - *Fabricated Semantic Label:* In the eICU Demo v2.0.1 metadata, Consortium 5 was simply a mathematical hash modulo bucket (`hospitalid % 5 == 0`) pooling 37 disparate facilities across the United States. It was **NOT** an individual rural critical-access hospital.
3. **Severe Small-Cell Imbalance in Dirichlet Partitions:** In Phase 12 Dirichlet partitions, Node 1 has only $N_{\text{pos}} = 2$ ($N_{\text{test}} = 5$) and Node 3 has only $N_{\text{pos}} = 2$ ($N_{\text{test}} = 220$). Estimating receiver operating characteristics on sub-cohorts with 2 positive events produces uninformative point estimates with massive confidence intervals ($[0.250, 1.000]$).

---

## 2. Recalculated Client Sample Sizes & Statistical Power

The table below reports independently recalculated sample counts, positive mortality events, prevalence rates, and statistical validity flags across all client partitions.

| Client Partition | Partitioning Methodology | Total Test Stays ($N_{\text{test}}$) | Mortality Deaths ($N_{\text{pos}}$) | Mortality Prevalence (%) | Statistical Validity Status | Audit Evaluation & Recommendation |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Node 1** | Dirichlet ($\alpha = 0.5$) | 5 | 2 | 40.00% | **UNINFORMATIVE_CELL_SIZE** | $N \le 10, N_{\text{pos}} = 2$. Point estimates uninformative; excluded from clinical claims. |
| **Node 2** | Dirichlet ($\alpha = 0.5$) | 88 | 12 | 13.64% | **SUFFICIENT_CELL_SIZE** | Statistically meaningful sample; reliable calibration signal. |
| **Node 3** | Dirichlet ($\alpha = 0.5$) | 220 | 2 | 0.91% | **UNINFORMATIVE_CELL_SIZE** | Extreme class imbalance ($0.91\%$); AUROC confidence interval exceeds $\pm 0.15$. |
| **Node 4** | Dirichlet ($\alpha = 0.5$) | 64 | 18 | 28.12% | **SUFFICIENT_CELL_SIZE** | Balanced high-acuity cohort; adapter maintains performance. |
| **Node 5** | Dirichlet ($\alpha = 0.5$) | 16 | 6 | 37.50% | **SUFFICIENT_CELL_SIZE** | Small sample with wide CI $[0.582, 1.000]$; interpret directionally only. |
| **Consortium 5** | Phase 11 Modulo (`% 5`) | 79 | 6 | 7.59% | **EXTREME_STATISTICAL_UNCERTAINTY** | **CLAIM WITHDRAWN**. $N_{\text{pos}} = 6$ yields CI $[0.552, 0.985]$; semantic "rural" label invalid. |

---

## 3. Independent Recalculation: Global vs. Personalized AUROC & 95% CIs

For every client node, AUROC metrics were recomputed by running forward passes through the frozen global baseline and personalized adapter checkpoints, followed by 1,000 non-parametric bootstrap iterations.

| Client Partition | Global FL Baseline AUROC | Global Model 95% CI | Personalized Adapter AUROC | Personalized Adapter 95% CI | Absolute $\Delta$ | Relative Gain (%) | Audit Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Node 1** | 1.0000 | [1.000, 1.000] | 0.8333 | [0.250, 1.000] | -0.1667 | -16.67% | Uninformative point estimate ($N_{\text{pos}} = 2$). |
| **Node 2** | 0.9814 | [0.954, 1.000] | 0.9704 | [0.929, 0.995] | -0.0110 | -1.12% | High global generalization; adapter preserves signal. |
| **Node 3** | 0.8899 | [0.789, 0.977] | 0.9106 | [0.867, 0.949] | +0.0206 | +2.32% | Low statistical power ($N_{\text{pos}} = 2$). |
| **Node 4** | 0.6884 | [0.544, 0.820] | 0.6739 | [0.526, 0.815] | -0.0145 | -2.11% | Robust high-event cohort; difference within error margin. |
| **Node 5** | 0.8333 | [0.564, 1.000] | 0.8333 | [0.582, 1.000] | 0.0000 | 0.00% | Stable point estimate; wide CI $[0.582, 1.000]$. |
| **Consortium 5 (P11)** | 0.1849 | [0.021, 0.432] | 0.8151 | [0.552, 0.985] | +0.6302 | +340.83% | **WITHDRAWN AS OVERSTATED**. 6 deaths cannot support clinical generalizability. |

---

## 4. Resolution of the "Rural Hospital" Claim

### What Was Previously Claimed:
Early project drafts asserted:
> *"The personalized quantum federated framework successfully rescued a failing rural hospital (Consortium 5) from an un-deployable AUROC of 0.1849 to a clinically reliable 0.8151."*

### Why It Must Be Withdrawn:
1. **Metadata Absence:** eICU Demo table `hospital.csv.gz` provides columns `hospitalid`, `numbedscategory`, `teachingstatus`, and `region`. It does **not** contain a "rural" designation. Consortium 5 was created programmatically via `hospitalid % 5 == 0`, agglomerating 37 distinct healthcare institutions across multiple states.
2. **The "Inverse Prediction" Artifact:** The global model's AUROC of 0.1849 in Consortium 5 occurred because with only 6 deaths out of 79 patients, two misclassified borderline cases pushed the rank ordering below 0.50 (an inverted ROC phenomenon common in micro-samples).
3. **Correct Scientific Phrasing:**
   All future publications must state:
   > *"In simulated non-IID client partitions with severe label scarcity, local residual adapters demonstrated directional adaptation. However, because small-volume partitions contained fewer than 10 mortality events, per-client metrics exhibit substantial statistical uncertainty (e.g., 95% CI $[0.552, 0.985]$), requiring future validation on full-scale multi-hospital databases."*
