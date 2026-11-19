# PHASE 13.3 AUDIT: STATISTICAL REPRODUCIBILITY & CLIENT-LEVEL AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3 — Final Source-Code Verification, DP Accounting Reproducibility and Experiment Consistency Audit  
**Author:** Independent Healthcare Biostatistician & Clinical Audit Specialist  
**Date:** November 2026  
**Status:** Audited & Statistically Hardened  

---

## 1. Executive Summary & Audit Scope

This document provides a rigorous statistical audit of:
1. The **patient-clustered paired bootstrap** comparing Hybrid DP-QNN vs. Classical DP-MLP.
2. The **federated client-level evaluation** across the 5 synthetic Dirichlet consortium nodes.
3. The validity and boundaries of the reported **hospital personalization gains**.

**Key Statistical Findings:**
- The patient-clustered paired bootstrap ($B = 1,000$ iterations) is methodologically sound: it resamples patient clusters, computes paired metric differences on identical resamples, and properly excludes degenerate single-class resamples.
- Across all 5 privacy budgets ($\epsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$), the 95% confidence intervals for $\Delta\text{AUROC}$ and $\Delta\text{AUPRC}$ cross zero ($p > 0.05$). **The null hypothesis of equal performance cannot be rejected. No quantum advantage is demonstrated.**
- In federated client evaluation, test set sample sizes and positive case counts are acutely constrained on certain nodes (Node 2 has $N_{\text{pos}} = 1$; Node 4 has $N_{\text{pos}} = 1$). The reported AUROC of 1.0000 on Node 2 is an artifact of a single positive test case and must **never** be cited as clinical efficacy.
- Personalization fine-tuning demonstrates a modest gain on Node 5 (+0.0152 AUROC on an adequately powered test set of $N = 102$ with 19 deaths) without test-set leakage, but shows negative or neutral transfer on other nodes. **No universal personalization benefit can be claimed.**

---

## 2. Audit of the Patient-Clustered Paired Bootstrap

### 2.1 Cluster-Level Resampling Methodology
In the landmark test set ($N = 241$ encounters), there are 203 unique patients (`uniquepid`). Multiple admissions from the same patient violate sample independence.

In `experiments/phase13_2/run_phase13_2_experiments.py` (lines 52–104):
1. **Cluster Unit:** Resampling is performed over the 203 unique patient IDs with replacement:
   ```python
   sampled_pts = rng.choice(unique_pts, size=len(unique_pts), replace=True)
   idx = np.concatenate([pt_to_idx[pt] for pt in sampled_pts])
   ```
2. **Paired Evaluation:** In each replicate $b$, the exact same resampled index array `idx` is used to index both `probs_mlp` and `probs_qnn`, guaranteeing an exact paired comparison:
   $$\Delta \text{Metric}^{(b)} = \text{Metric}(y[\text{idx}], p_{\text{QNN}}[\text{idx}]) - \text{Metric}(y[\text{idx}], p_{\text{MLP}}[\text{idx}])$$
3. **Degeneracy Handling:** If all sampled records in replicate $b$ belong to class 0 (or class 1), the iteration is skipped (`if len(np.unique(y_b)) < 2: continue`).
4. **Interval Construction:** 95% Confidence Intervals are constructed using the empirical 2.5th and 97.5th percentiles of the replicate differences.
5. **Two-Sided $p$-value:** Computed as $2 \times \min(\mathbb{P}(\Delta \le 0), \mathbb{P}(\Delta \ge 0))$.

### 2.2 Paired Bootstrap Summary Table

The table below reflects the verified paired bootstrap results from `results/phase13_2/paired_statistical_tests.csv`:

| Budget $\epsilon$ | Metric | Classical MLP | Hybrid QNN | Point $\Delta$ ($\text{QNN}-\text{MLP}$) | Bootstrap Mean $\Delta$ | Bootstrap SE | 95% Confidence Interval | $p$-value | Significant at $\alpha=0.05$ | Audit Finding |
|---|---|---|---|---|---|---|---|---|---|---|
| **1.0** | $\Delta\text{AUROC}$ | 0.5857 | 0.5904 | +0.0047 | +0.0045 | 0.0554 | **[-0.1028, +0.1135]** | 0.920 | No | CIs cross 0 ($p = 0.920$) |
| 1.0 | $\Delta\text{AUPRC}$ | 0.1772 | 0.1382 | -0.0390 | -0.0401 | 0.0441 | **[-0.1522, +0.0194]** | 0.210 | No | CIs cross 0 ($p = 0.210$) |
| 1.0 | $\Delta\text{Brier}$ | 0.1144 | 0.1139 | -0.0005 | -0.0005 | 0.0002 | **[-0.0008, -0.0002]** | 0.000 | Yes (Minor) | Negligible difference ($< 0.001$) |
| **2.0** | $\Delta\text{AUROC}$ | 0.5778 | 0.6184 | +0.0406 | +0.0416 | 0.0531 | **[-0.0570, +0.1489]** | 0.416 | No | CIs cross 0 ($p = 0.416$) |
| 2.0 | $\Delta\text{AUPRC}$ | 0.1898 | 0.1522 | -0.0377 | -0.0400 | 0.0519 | **[-0.1693, +0.0332]** | 0.312 | No | CIs cross 0 ($p = 0.312$) |
| **4.0** | $\Delta\text{AUROC}$ | 0.6016 | 0.6410 | +0.0394 | +0.0396 | 0.0505 | **[-0.0583, +0.1404]** | 0.396 | No | CIs cross 0 ($p = 0.396$) |
| 4.0 | $\Delta\text{AUPRC}$ | 0.1945 | 0.1675 | -0.0270 | -0.0289 | 0.0511 | **[-0.1574, +0.0427]** | 0.476 | No | CIs cross 0 ($p = 0.476$) |
| **8.0** | $\Delta\text{AUROC}$ | 0.6281 | 0.6648 | +0.0367 | +0.0367 | 0.0468 | **[-0.0553, +0.1279]** | 0.424 | No | CIs cross 0 ($p = 0.424$) |
| 8.0 | $\Delta\text{AUPRC}$ | 0.2006 | 0.1817 | -0.0189 | -0.0197 | 0.0510 | **[-0.1399, +0.0604]** | 0.630 | No | CIs cross 0 ($p = 0.630$) |
| **$\infty$** | $\Delta\text{AUROC}$ | 0.7166 | 0.7658 | +0.0491 | +0.0487 | 0.0464 | **[-0.0424, +0.1397]** | 0.310 | No | CIs cross 0 ($p = 0.310$) |
| $\infty$ | $\Delta\text{AUPRC}$ | 0.2722 | 0.2957 | +0.0235 | +0.0210 | 0.0543 | **[-0.0868, +0.1237]** | 0.608 | No | CIs cross 0 ($p = 0.608$) |

**Conclusion:** In all 5 evaluation settings, the 95% confidence intervals span zero, and all p-values exceed 0.20. There is no evidence of quantum advantage.

---

## 3. Audit of Federated Client Evaluation & Small-Sample Artifacts

### 3.1 Client Demographic Reality
The 5 clients in `results/phase13_2/federated_client_identity.csv` are synthetic Dirichlet partitions ($\alpha = 0.5$) of the eICU landmark dataset:
- Node 1: $N_{\text{train}} = 137, \ N_{\text{test}} = 29, \ N_{\text{pos, test}} = 5$ (Prevalence: 17.2%)
- Node 2: $N_{\text{train}} = 85, \ N_{\text{test}} = 21, \ N_{\text{pos, test}} = 1$ (Prevalence: 4.8%)
- Node 3: $N_{\text{train}} = 661, \ N_{\text{test}} = 82, \ N_{\text{pos, test}} = 2$ (Prevalence: 2.4%)
- Node 4: $N_{\text{train}} = 116, \ N_{\text{test}} = 7, \ N_{\text{pos, test}} = 1$ (Prevalence: 14.3%)
- Node 5: $N_{\text{train}} = 133, \ N_{\text{test}} = 102, \ N_{\text{pos, test}} = 19$ (Prevalence: 18.6%)

### 3.2 The Single-Positive-Case Artifact (Node 2 and Node 4)
- **Node 2 Observation:** `Global_FedAvg_AUROC = 1.0000` and `Personalized_FineTuned_AUROC = 1.0000`.
- **Audit Explanation:** Node 2 test set contains exactly **1 positive case** out of 21 test encounters. In ROC analysis with a single positive instance, if the model predicts any probability for that single positive instance that happens to exceed the probabilities of the 20 negative instances, the true positive rate reaches 1.0 with zero false positives at that threshold, yielding an AUROC of 1.0000.
- **Scientific Rule:** An AUROC calculated on $N_{\text{pos}} = 1$ has zero statistical reliability (standard error is undefined or unbounded). The manuscript must explicitly disclose this artifact and **never** claim that the model achieved "perfect prediction" on Node 2.
- **Node 4 Observation:** Node 4 has only 7 test samples and 1 positive case. Its AUROC (0.6667) is similarly constrained by sample scarcity.

### 3.3 Audit of Node 5 Personalization
- **Node 5 Characteristics:** $N_{\text{train}} = 133$ (6 deaths), $N_{\text{test}} = 102$ (19 deaths). Node 5 has the only test holdout with sufficient positive cases to support meaningful statistical evaluation.
- **Personalization Outcome:**
  - Local Model Only: AUROC = 0.6056, PRAUC = 0.2785
  - Global FedAvg: AUROC = 0.6703, PRAUC = 0.3333
  - Personalized Fine-Tuning: AUROC = 0.6855, PRAUC = 0.3453
  - Personalization Gain: $\Delta\text{AUROC} = +0.0152$ (+1.5 percentage points)
- **Data Integrity Verification:**
  - Training used exclusively `client.X_train` and `client.y_train` for 3 epochs with learning rate 0.001.
  - Test set `client.X_test` was evaluated strictly after training was complete.
  - Zero test leakage occurred.
- **Scientific Boundary:** While personalization provided a modest benefit on Node 5 (+0.0152), it produced neutral transfer on Node 4 (+0.0000) and negative transfer on Node 1 (-0.1000) and Node 3 (-0.1125) due to small local training set overfitting. Therefore, **no claim of universal or guaranteed personalization advantage can be made**.

---

## 4. Manuscript Recommendations

1. **Rejection of Quantum Advantage:** State clearly that under patient-clustered bootstrap testing, hybrid quantum-classical networks achieve predictive parity with classical baselines, but do not provide statistically significant improvements.
2. **Explicit Small-Sample Disclosure:** Report the number of positive test cases alongside every client-level metric ($N_{\text{pos}} = 1$ for Nodes 2 and 4).
3. **Nuanced Personalization Narrative:** Frame personalization as an adaptive mechanism that can yield modest gains on high-prevalence sites (Node 5), but which is prone to over-adaptation or negative transfer when local sample sizes are severely restricted.
