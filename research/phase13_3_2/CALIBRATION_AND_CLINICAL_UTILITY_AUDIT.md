# Calibration and Clinical Utility Audit

**Phase:** 13.3.2  
**Task:** Calibration Metrics, Clinical Operating Points, and Retrospective Scope Audit  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Clinical Informatics Specialist  

---

## 1. Executive Summary

This audit assesses the probabilistic calibration, clinical operating points, and translational utility of the hybrid quantum-classical and classical neural network models trained with differential privacy on the eICU landmark cohort ($N=241$ test encounters, $11.62\%$ mortality prevalence).

Key findings:
1. **Calibration Performance:** The uncalibrated non-private Hybrid QNN achieves a Brier score of **$0.1106$** and Expected Calibration Error (ECE, 10 bins) of **$0.1027$**, outperforming the uncalibrated classical MLP (Brier: **$0.1621$**, ECE: **$0.1232$**).
2. **Clinical Decision Thresholds:** At a high-specificity operating point ($\text{Specificity} \ge 80\%$), the non-private QNN detects **$64.29\%$** of subsequent in-hospital deaths ($\text{Sensitivity} = 0.6429$), compared to **$46.43\%$** for the classical MLP.
3. **Prevalence Constraints:** Under an $11.62\%$ test mortality rate, positive predictive value (precision) is inherently constrained across all models ($0.12 - 0.26$).
4. **Regulatory & Scope Boundary:** All reported results derive from retrospective observational data. They demonstrate algorithmic feasibility and risk stratification potential, but do **not** constitute prospective clinical validation or diagnostic readiness.

---

## 2. Clinical Operating Points & Calibration Table

Table 3 details the operating characteristics and calibration metrics across all 10 trained models evaluated on the locked test cohort:

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

---

## 3. Analysis of Clinical Utility Metrics

### 3.1 High-Specificity Triage Capability
In an intensive care unit, false positive mortality alerts generate alarm fatigue and clinical disruption. High-specificity thresholds are therefore critical:
- **At 80% Specificity ($\text{FPR} \le 20\%$):**
  - The non-private Hybrid QNN detects **$64.29\%$** of all in-hospital mortality events ($18$ out of $28$ deaths), generating at most $42$ false alarms among $213$ survivors.
  - The classical MLP detects **$46.43\%$** of deaths ($13$ out of $28$ deaths).
- **At 90% Specificity ($\text{FPR} \le 10\%$):**
  - QNN detects **$35.71\%$** of deaths ($10$ out of $28$).
  - MLP detects **$39.29\%$** of deaths ($11$ out of $28$).
- **At 95% Specificity ($\text{FPR} \le 5\%$):**
  - Both QNN and MLP capture **$21.43\%$** of deaths ($6$ out of $28$) with fewer than $11$ false alarms across the entire test set.

### 3.2 High-Sensitivity Screening Capability
When the clinical objective is ruling out high-risk patients (screening):
- **At 80% Sensitivity ($\ge 80\%$ of deaths captured):**
  - Non-private QNN achieves **$52.58\%$** specificity (ruling out over half of non-dying patients with $\le 5$ missed deaths).
  - Classical MLP achieves **$59.62\%$** specificity.
- **At 90% Sensitivity ($\ge 90\%$ of deaths captured):**
  - Non-private QNN achieves **$43.66\%$** specificity.
  - Classical MLP specificity degrades sharply to **$15.02\%$**.

---

## 4. Calibration Assessment (Brier Score and ECE)

### 4.1 Expected Calibration Error (ECE)
ECE measures the average discrepancy between predicted probabilities and observed empirical frequencies across 10 equal-width bins in $[0, 1]$:
$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **DP-Trained Models:** Across all DP levels ($\epsilon = 1.0$ to $8.0$), ECE remains tightly bounded between **$0.1057$** and **$0.1091$**. Because predictions are compressed in $[0.005, 0.045]$, the vast majority of samples fall into the first bin ($[0.0, 0.1]$), where the empirical death rate is $11.62\%$, yielding an expected calibration gap of $\approx |0.010 - 0.116| \approx 0.106$.
- **Non-Private Comparison:** Non-private QNN has an ECE of **$0.1027$**, whereas non-private MLP has an ECE of **$0.1232$** and a substantially worse Brier score of **$0.1621$** (reflecting overconfident predictions on negative cases).

---

## 5. Formal Clinical Limitations & Scope Boundaries

The manuscript must clearly articulate the following clinical constraints:
1. **Retrospective Benchmark:** All analyses are conducted on historical observational records from the eICU Collaborative Research Database. The cohort represents patients who survived the first 24 hours of ICU admission; findings cannot be generalized to hyper-acute admission mortality ($< 24$ hours).
2. **Missing Data Imputation:** Missing clinical variables were imputed using training-split medians. In practice, missingness patterns often carry informative prognostic signals that are discarded by static imputation.
3. **No Prospective Interventional Evidence:** The models have not been evaluated in active clinical workflows. High AUROC or sensitivity does not imply that algorithmic intervention improves patient survival or alters physician decision-making.
4. **Prevalence Dependence:** Predictive values (precision) depend heavily on local ICU mortality prevalence. If deployed in units with lower mortality (e.g., intermediate care units with $3\%$ mortality), precision will decrease accordingly.
