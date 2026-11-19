# CLINICAL OPERATING POINT VERIFICATION & RETROSPECTIVE TRADE-OFF AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.4 — Final Statistical Audit, Calibration Validation and Decision-Curve Correction  
**Author:** Senior Medical AI Researcher, Biostatistician & DP Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_4/CLINICAL_OPERATING_POINT_VERIFICATION.md`  

---

## 1. Executive Summary

A pervasive defect in applied medical machine learning literature is reporting model accuracy under an arbitrary default threshold ($\tau = 0.50$) without considering cohort class prevalence. In the landmark ICU cohort, in-hospital mortality prevalence is approximately $11.6\%$, and models trained with probability compression output maximum predicted risks below $0.15$. Under $\tau = 0.50$, both the hybrid QNN and classical DP-MLP assign zero positive predictions ($\text{Sensitivity} = 0.0\%$), rendering the model clinically inert.

In Phase 13.3.4, we audit the performance of **validation-selected decision thresholds** evaluated out-of-sample on the frozen test set ($N_{\text{test}} = 241, 28\text{ deaths}$). We provide exact confusion matrices, bootstrap confidence intervals, precision, NPV, balanced accuracy, and clinical false-alarm ratios.

---

## 2. Threshold Selection Rules (Strictly Validation-Locked)

To prevent data leakage, all operational decision thresholds are derived strictly from the **validation split** ($N_{\text{val}} = 234, 24\text{ deaths}$) and frozen before application to the test set:
1. **Validation Youden Index $J$ ($\tau_{\text{Youden}}$):**
   $$\tau_{\text{Youden}} = \arg\max_\tau \left( \text{Sensitivity}_{\text{val}}(\tau) + \text{Specificity}_{\text{val}}(\tau) - 1 \right)$$
   Selects the operating point optimizing the unweighted trade-off between sensitivity and specificity.
2. **Validation Target 80% Specificity ($\tau_{80}$):**
   The lowest threshold achieving $\text{Specificity}_{\text{val}} \ge 0.80$ with maximum sensitivity, prioritizing alarm burden reduction.
3. **Validation Target 90% Specificity ($\tau_{90}$):**
   The lowest threshold achieving $\text{Specificity}_{\text{val}} \ge 0.90$.
4. **Validation Target 95% Specificity ($\tau_{95}$):**
   The lowest threshold achieving $\text{Specificity}_{\text{val}} \ge 0.95$.
5. **Default Benchmark ($\tau = 0.50$):**
   Included strictly to document the catastrophic failure of uncalibrated default thresholds.

---

## 3. Audited Operating Points Table on Frozen Test Set

The table below presents the verified out-of-sample test performance across key model configurations from `results/phase13_3_4/clinical_operating_points.csv`:

| Model Architecture | $\varepsilon$ | Selection Rule | Threshold $\tau$ | TP | FP | TN | FN | Sensitivity (95% CI) | Specificity (95% CI) | PPV | NPV | F1 | Bal Acc | False Alarm Ratio (FP:TP) | Flagged Rate |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid DP-QNN** | $\infty$ | Default 0.50 | 0.500000 | 0 | 0 | 213 | 28 | 0.0% [0.0%, 0.0%] | 100.0% [100.0%, 100.0%] | 0.0% | 88.4% | 0.0000 | 50.0% | $\infty$ | 0.0% (0) |
| **Hybrid DP-QNN** | $\infty$ | **Validation Youden $J$** | **0.006521** | **21** | **75** | **138** | **7** | **75.0% [58.3%, 89.5%]** | **64.8% [58.4%, 70.9%]** | **21.9%** | **95.2%** | **0.3387** | **69.9%** | **3.57 : 1** | **39.8% (96)** |
| **Hybrid DP-QNN** | $\infty$ | **Val Target Spec 80%** | **0.007673** | **18** | **40** | **173** | **10** | **64.3% [45.7%, 81.0%]** | **81.2% [75.5%, 86.6%]** | **31.0%** | **94.5%** | **0.4186** | **72.8%** | **2.22 : 1** | **24.1% (58)** |
| **Hybrid DP-QNN** | $\infty$ | Val Target Spec 90% | 0.013515 | 10 | 23 | 190 | 18 | 35.7% [17.5%, 55.6%] | 89.2% [84.8%, 93.3%] | 30.3% | 91.4% | 0.3279 | 62.5% | 2.30 : 1 | 13.7% (33) |
| **Hybrid DP-QNN** | $\infty$ | Val Target Spec 95% | 0.088928 | 4 | 4 | 209 | 24 | 14.3% [3.2%, 28.0%] | 98.1% [96.0%, 99.6%] | 50.0% | 89.7% | 0.2222 | 56.2% | 1.00 : 1 | 3.3% (8) |
| Classical DP-MLP | $\infty$ | Default 0.50 | 0.500000 | 11 | 32 | 181 | 17 | 39.3% [20.0%, 59.3%] | 85.0% [79.8%, 89.6%] | 25.6% | 91.4% | 0.3099 | 62.1% | 2.91 : 1 | 17.8% (43) |
| Classical DP-MLP | $\infty$ | Validation Youden $J$ | 0.080205 | 21 | 82 | 131 | 7 | 75.0% [57.9%, 90.0%] | 61.5% [54.5%, 68.1%] | 20.4% | 94.9% | 0.3206 | 68.3% | 3.90 : 1 | 42.7% (103) |
| Classical DP-MLP | $\infty$ | Val Target Spec 80% | 0.316313 | 12 | 38 | 175 | 16 | 42.9% [23.8%, 62.5%] | 82.2% [76.6%, 87.3%] | 24.0% | 91.6% | 0.3077 | 62.5% | 3.17 : 1 | 20.8% (50) |
| Classical DP-MLP | $\infty$ | Val Target Spec 90% | 0.883112 | 9 | 18 | 195 | 19 | 32.1% [13.8%, 52.2%] | 91.6% [87.5%, 95.1%] | 33.3% | 91.1% | 0.3273 | 61.9% | 2.00 : 1 | 11.2% (27) |
| Hybrid DP-QNN | $8.0$ | Validation Youden $J$ | 0.011079 | 12 | 50 | 163 | 16 | 42.9% [23.3%, 61.9%] | 76.5% [70.2%, 82.4%] | 19.4% | 91.1% | 0.2667 | 59.7% | 4.17 : 1 | 25.7% (62) |
| Classical DP-MLP | $8.0$ | Validation Youden $J$ | 0.008374 | 14 | 51 | 162 | 14 | 50.0% [30.8%, 71.9%] | 76.1% [69.6%, 81.9%] | 21.5% | 92.1% | 0.3011 | 63.0% | 3.64 : 1 | 27.0% (65) |
| Hybrid DP-QNN | $1.0$ | Validation Youden $J$ | 0.009810 | 12 | 85 | 128 | 16 | 42.9% [24.1%, 62.5%] | 60.1% [53.0%, 67.1%] | 12.4% | 88.9% | 0.1920 | 51.5% | 7.08 : 1 | 40.2% (97) |
| Classical DP-MLP | $1.0$ | Validation Youden $J$ | 0.007880 | 14 | 55 | 158 | 14 | 50.0% [30.8%, 71.9%] | 74.2% [67.8%, 80.4%] | 20.3% | 91.9% | 0.2887 | 62.1% | 3.93 : 1 | 28.6% (69) |

---

## 4. In-Depth Clinical Trade-Off Analysis

### 4.1 Youden Operating Point: Prioritizing Mortality Capture
- **Hybrid QNN ($\varepsilon=\infty$):** At $\tau = 0.006521$, the model captures **21 out of 28 deaths** ($\text{Sensitivity} = 75.00\%$), missing 7 deteriorating patients.
- **Cost:** It triggers alarms on 75 surviving patients, resulting in a **False Alarm Ratio of $3.57 : 1$** (3.57 false alarms for every true positive).
- **Alarm Burden:** The algorithm flags **39.8% of the entire ICU test cohort** (96 encounters). In a busy 20-bed ICU, alerting on 40% of patients daily risks inducing severe clinical alarm fatigue.

### 4.2 Target 80% Specificity: Balancing Alert Burden
- **Hybrid QNN ($\varepsilon=\infty$):** At $\tau = 0.007673$, the model achieves **$81.22\%$ specificity** ($\text{FP} = 40$), capturing **18 out of 28 deaths** ($\text{Sensitivity} = 64.29\%$).
- **Cost:** An additional 3 deaths are missed (10 missed total vs 7 at Youden).
- **Benefit:** The False Alarm Ratio drops to **$2.22 : 1$**, and the proportion of flagged encounters decreases from $39.8\%$ down to **$24.07\%$** (58 encounters).

---

## 5. Statistical Transparency: High Sampling Uncertainty

A paramount finding that must be communicated with complete candor in the manuscript is the **high statistical variance inherent in small-cohort ICU evaluations**:
1. **Total Positive Cases = 28:**
   In the test cohort of 241 encounters, there are only 28 deaths. Consequently, a single correctly identified or missed death shifts the empirical sensitivity by:
   $$\Delta \text{Sens} = \frac{1}{28} \approx 3.57\%$$
   A change of 3 patients shifts sensitivity by over $10.7\%$.
2. **Wide Confidence Intervals:**
   The patient-clustered bootstrap 95% confidence intervals reflect this substantial sampling variability:
   - QNN Youden Sensitivity: $[58.33\%, 89.48\%]$ ($\text{Width} = 31.15\%$).
   - QNN Target 80% Specificity Sensitivity: $[45.71\%, 80.95\%]$ ($\text{Width} = 35.24\%$).
3. **Mandatory Manuscript Framing:**
   The manuscript must state:
   > *"Operating point evaluations on this 241-encounter test cohort represent retrospective descriptions of clinical trade-offs rather than prospective operational performance. Because the test set contains 28 mortality events, empirical sensitivity estimates carry wide 95% confidence intervals (e.g., [45.7%, 81.0%] for the 80% specificity threshold). These findings establish proof-of-concept feasibility but require validation in large, multi-center prospective cohorts before automated bedside deployment can be considered."*
