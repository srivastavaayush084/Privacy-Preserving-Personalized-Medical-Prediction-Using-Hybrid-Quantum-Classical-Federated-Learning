# Patient-Clustered Statistical Uncertainty & Bootstrap Audit

**Phase:** 13.3.3  
**Task:** Clustered Bootstrap Estimation (B=2,000) Accounting for Multi-Encounter Patients  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Biostatistical Methodologist  

---

## 1. Executive Summary

Standard independent and identically distributed (i.i.d.) resampling is invalid when individual patients contribute multiple admissions to an observational clinical cohort. In the frozen test split, **198 unique patients** account for **241 ICU stays**. Because intra-patient clinical outcomes and physiological traits are correlated, naive encounter-level bootstrapping underestimates standard errors and produces falsely narrow confidence intervals.

This audit establishes:
1. **Patient-Clustered Bootstrap Implementation:** 198 unique patients were resampled with replacement over **$B = 2,000$ independent replicates**. Whenever a patient was selected, all of their ICU encounters were included as an intact cluster.
2. **Identical Sample Pairing:** Identical patient-resampled datasets were evaluated across all competing models on each replicate to compute exact paired differences ($\Delta = \text{QNN} - \text{MLP}$).
3. **No Statistically Credible Superiority:**
   - **$\Delta$ AUROC:** $+0.0494$, $95\%$ Clustered CI: **$[-0.0409, 0.1393]$**, two-sided **$p = 0.290$**.
   - **$\Delta$ AUPRC:** $+0.0244$, $95\%$ Clustered CI: **$[-0.0843, 0.1256]$**, two-sided **$p = 0.624$**.
   - **$\Delta$ Brier (Platt):** $+0.0015$, $95\%$ Clustered CI: **$[-0.0042, 0.0082]$**, two-sided **$p = 0.629$**.
   - **$\Delta$ F1 (Youden):** $+0.0174$, $95\%$ Clustered CI: **$[-0.0596, 0.0940]$**, two-sided **$p = 0.667$**.
4. **Methodological Finding:** The apparent point-estimate margin of the Hybrid QNN over Classical MLP is **statistically indistinguishable from random sampling variability** ($p > 0.20$ across all discriminative and clinical decision metrics).

---

## 2. Test Cohort Clustering Structure

The distribution of ICU admissions per unique patient in the test set ($N=241$ encounters, $198$ patients) is:

| Stays per Patient | Unique Patient Count | Total Encounter Count | Percentage of Encounters |
| :---: | :---: | :---: | :---: |
| 1 Stay | 164 | 164 | 68.05% |
| 2 Stays | 29 | 58 | 24.07% |
| 3 Stays | 2 | 6 | 2.49% |
| 4 Stays | 2 | 8 | 3.32% |
| 5 Stays | 1 | 5 | 2.07% |
| **Total** | **198 Unique Patients** | **241 Stays** | **100.00%** |

Nearly one-third of all test encounters ($31.95\%$) belong to patients with readmissions. Failure to cluster at the `uniquepid` level violates the independence assumption of the empirical bootstrap.

---

## 3. Clustered Bootstrap Algorithm

For replicate $b = 1, \dots, B$ (with $B = 2,000$ and seed $42$):
1. Draw $198$ unique patients with replacement from $\{pt_1, \dots, pt_{198}\}$.
2. Concatenate all ICU encounter records belonging to the selected patients to form bootstrap sample $\mathcal{D}_b^*$. The resulting encounter size $N_b^*$ fluctuates naturally around $241$.
3. If $\mathcal{D}_b^*$ contains only survivors or only deaths (zero outcome variance), the replicate is discarded and redrawn.
4. Evaluate competing models simultaneously on $\mathcal{D}_b^*$ to maintain strict pairing.
5. Compute percentile $95\%$ confidence intervals $[Q_{0.025}, Q_{0.975}]$ and empirical two-sided p-values:
   $$p = 2 \times \min\left(\frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b \le 0), \frac{1}{B}\sum_{b=1}^B \mathbb{I}(\Delta_b \ge 0)\right)$$

---

## 4. Patient-Clustered Bootstrap Metrics Across Models

Table 2 presents the bootstrap mean, median, and $95\%$ confidence intervals across all 10 trained models:

| Model Architecture | $\epsilon$ | Metric | Point Estimate | Bootstrap Mean | Bootstrap Median | 95% Clustered CI | CI Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid DP-QNN** | $\infty$ | AUROC | $0.7658$ | $0.7663$ | $0.7679$ | **$[0.6807, 0.8449]$** | $0.1642$ |
| **Hybrid DP-QNN** | $\infty$ | AUPRC | $0.2957$ | $0.3129$ | $0.3015$ | **$[0.1779, 0.4866]$** | $0.3087$ |
| **Hybrid DP-QNN** | $\infty$ | Brier (Raw) | $0.1106$ | $0.1105$ | $0.1098$ | $[0.0701, 0.1542]$ | $0.0841$ |
| **Hybrid DP-QNN** | $\infty$ | Brier (Platt) | $0.0980$ | $0.0980$ | $0.0975$ | $[0.0666, 0.1331]$ | $0.0665$ |
| **Hybrid DP-QNN** | $\infty$ | Sens (Youden) | $0.7500$ | $0.7483$ | $0.7500$ | $[0.5833, 0.8948]$ | $0.3114$ |
| **Hybrid DP-QNN** | $\infty$ | Spec (Youden) | $0.6479$ | $0.6483$ | $0.6482$ | $[0.5837, 0.7091]$ | $0.1254$ |
| **Hybrid DP-QNN** | $\infty$ | F1 (Youden) | $0.3387$ | $0.3374$ | $0.3361$ | $[0.2314, 0.4463]$ | $0.2149$ |
| **Classical DP-MLP** | $\infty$ | AUROC | $0.7166$ | $0.7169$ | $0.7186$ | **$[0.6033, 0.8200]$** | $0.2167$ |
| **Classical DP-MLP** | $\infty$ | AUPRC | $0.2722$ | $0.2885$ | $0.2741$ | **$[0.1605, 0.4778]$** | $0.3173$ |
| **Classical DP-MLP** | $\infty$ | Brier (Raw) | $0.1621$ | $0.1624$ | $0.1621$ | $[0.1227, 0.2038]$ | $0.0811$ |
| **Classical DP-MLP** | $\infty$ | Brier (Platt) | $0.0964$ | $0.0965$ | $0.0960$ | $[0.0653, 0.1300]$ | $0.0647$ |
| **Classical DP-MLP** | $\infty$ | Sens (Youden) | $0.7500$ | $0.7495$ | $0.7500$ | $[0.5789, 0.9000]$ | $0.3211$ |
| **Classical DP-MLP** | $\infty$ | Spec (Youden) | $0.6150$ | $0.6150$ | $0.6146$ | $[0.5446, 0.6806]$ | $0.1360$ |
| **Classical DP-MLP** | $\infty$ | F1 (Youden) | $0.3206$ | $0.3200$ | $0.3182$ | $[0.2080, 0.4296]$ | $0.2216$ |
| **Hybrid DP-QNN** | $8.0$ | AUROC | $0.6648$ | $0.6648$ | $0.6677$ | $[0.5630, 0.7588]$ | $0.1958$ |
| **Hybrid DP-QNN** | $4.0$ | AUROC | $0.6410$ | $0.6397$ | $0.6421$ | $[0.5369, 0.7381]$ | $0.2012$ |
| **Hybrid DP-QNN** | $2.0$ | AUROC | $0.6184$ | $0.6184$ | $0.6193$ | $[0.5186, 0.7136]$ | $0.1950$ |
| **Hybrid DP-QNN** | $1.0$ | AUROC | $0.5904$ | $0.5910$ | $0.5912$ | $[0.4919, 0.6884]$ | $0.1965$ |
| **Classical DP-MLP** | $8.0$ | AUROC | $0.6281$ | $0.6284$ | $0.6300$ | $[0.5058, 0.7483]$ | $0.2425$ |
| **Classical DP-MLP** | $4.0$ | AUROC | $0.6016$ | $0.6016$ | $0.6022$ | $[0.4795, 0.7247]$ | $0.2452$ |
| **Classical DP-MLP** | $2.0$ | AUROC | $0.5778$ | $0.5786$ | $0.5794$ | $[0.4566, 0.7061]$ | $0.2495$ |
| **Classical DP-MLP** | $1.0$ | AUROC | $0.5857$ | $0.5878$ | $0.5888$ | $[0.4566, 0.7176]$ | $0.2610$ |

---

## 5. Paired Quantum-Classical Difference Analysis

Table 3 details the exact paired difference distributions ($\text{QNN}_{\text{inf}} - \text{MLP}_{\text{inf}}$) across the $2,000$ paired clustered replicates:

| Difference Metric | Mean Difference | Median Difference | 95% Clustered CI | Two-Sided Empirical $p$-Value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$\Delta$ AUROC** | $+0.0494$ | $+0.0486$ | **$[-0.0409, 0.1393]$** | **$0.290$** | **No** |
| **$\Delta$ AUPRC** | $+0.0244$ | $+0.0237$ | **$[-0.0843, 0.1256]$** | **$0.624$** | **No** |
| **$\Delta$ Brier (Raw)** | $-0.0519$ | $-0.0523$ | **$[-0.0933, -0.0087]$** | **$0.020$** | **Yes** (Due to uncalibrated MLP overconfidence) |
| **$\Delta$ Brier (Platt)** | $+0.0015$ | $+0.0015$ | **$[-0.0042, 0.0082]$** | **$0.629$** | **No** (Parity upon recalibration) |
| **$\Delta$ Sensitivity** | $-0.0011$ | $0.0000$ | **$[-0.2000, 0.1905]$** | **$1.000$** | **No** |
| **$\Delta$ Specificity** | $+0.0333$ | $+0.0330$ | **$[-0.0193, 0.0876]$** | **$0.239$** | **No** |
| **$\Delta$ F1 (Youden)** | $+0.0174$ | $+0.0164$ | **$[-0.0596, 0.0940]$** | **$0.667$** | **No** |

---

## 6. Biostatistical Interpretation

1. **Why Overlapping Marginal CIs Are Insufficient:**  
   In naive reviews, auditors sometimes check whether confidence intervals overlap. While non-overlapping CIs guarantee significance, overlapping CIs do not prove non-significance for paired data. However, our **paired bootstrap analysis** explicitly tests the paired difference vector:
   $$\mathbb{P}(\text{AUROC}_{\text{QNN}} \le \text{AUROC}_{\text{MLP}}) = 14.5\% \implies p = 2 \times 0.145 = 0.290$$
   This proves that in nearly $1$ out of every $7$ resampled patient cohorts, the Classical MLP actually achieves a higher AUROC than the Hybrid QNN.
2. **The Raw Brier Artifact:**  
   The only metric where QNN significantly differs from MLP is raw Brier score ($p = 0.020$). As proven in Section 4 of the Calibration Audit, this is driven entirely by Classical MLP's extreme logit dispersion. As soon as both models are calibrated, the difference drops to a negligible $+0.0015$ ($p = 0.629$).
3. **Scientific Publication Finding:**  
   The manuscript cannot claim that the Hybrid QNN outperforms the Classical MLP with statistical significance. The findings must be reported as **comparable discrimination within wide patient-clustered confidence intervals**.
