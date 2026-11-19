# Clinical Operating-Point Validation & False-Alarm Audit

**Phase:** 13.3.3  
**Task:** Independent Verification of Out-of-Sample Clinical Decision Thresholds and ICU Workload Burden  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Critical Care Informatician  

---

## 1. Executive Summary

In critical care medicine, model evaluation cannot be confined to threshold-free summary metrics like AUROC. Intensive Care Unit (ICU) clinicians act on binary decisions (e.g., palliative care consult, high-intensity hemodynamic monitoring, escalation of therapy). 

This audit evaluates clinical decision thresholds under strict out-of-sample isolation:
1. **Zero Test-Tuning Rule:** All decision thresholds ($\tau^*$) were identified **strictly on the independent validation split** ($N=234, 24\text{ deaths}$) and locked prior to evaluating the test set ($N=241, 28\text{ deaths}$).
2. **Default Threshold Pathology:** Across 9 of the 10 models (all DP models and non-private QNN), the default threshold ($\tau = 0.50$) identifies **zero high-risk patients** ($\text{Sensitivity} = 0.0000$, $\text{F1} = 0.0000$).
3. **Validation Youden's J Threshold:** Unlocks high clinical sensitivity:
   - **Non-private QNN ($\tau^* = 0.006521$):** Sensitivity $= \mathbf{75.00\%}$ ($21/28$ deaths), Specificity $= \mathbf{64.79\%}$ ($138/213$ survivors), Balanced Accuracy $= \mathbf{69.89\%}$, F1 $= \mathbf{0.3387}$.
   - **Classical MLP ($\tau^* = 0.080205$):** Sensitivity $= \mathbf{75.00\%}$ ($21/28$ deaths), Specificity $= \mathbf{61.50\%}$ ($131/213$ survivors), Balanced Accuracy $= \mathbf{68.25\%}$, F1 $= \mathbf{0.3206}$.
4. **Target-Specificity Operating Points:**
   - At a **Validation 80% Specificity Target** ($\tau^* = 0.007673$), the QNN achieves **$81.22\%$ test specificity** and **$64.29\%$ test sensitivity** ($18/28$ deaths captured with $40$ false alarms).
   - At a **Validation 90% Specificity Target** ($\tau^* = 0.013515$), test specificity is **$89.20\%$**, but sensitivity falls sharply to **$35.71\%$** ($18$ missed deaths).
5. **False-Positive Burden Analysis:** Due to an $11.62\%$ baseline mortality prevalence, achieving high sensitivity ($75\%$) requires accepting a **Positive Predictive Value (PPV) of $\approx 22\%$**, generating roughly **$3.6$ false alarms for every true positive mortality prediction**.

---

## 2. Validation-Locked Operating Points Table

Table 4 details the classification metrics and patient-clustered bootstrap $95\%$ confidence intervals across all evaluation rules:

| Model | $\epsilon$ | Operating Rule | Threshold $\tau^*$ | TP | FP | TN | FN | Test Sens [95% CI] | Test Spec [95% CI] | PPV | F1 | Bal. Acc | Pred Pos |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hybrid QNN** | $\infty$ | Default $\tau=0.50$ | $0.500000$ | 0 | 0 | 213 | 28 | $0.0000$ [$0.00, 0.00$] | $1.0000$ [$1.00, 1.00$] | $0.0000$ | $0.0000$ | $0.5000$ | 0 (0.0%) |
| **Hybrid QNN** | $\infty$ | Youden's J | $0.006521$ | 21 | 75 | 138 | 7 | **$0.7500$** [$0.58, 0.89$] | **$0.6479$** [$0.58, 0.71$] | $0.2188$ | $0.3387$ | $0.6989$ | 96 (39.8%) |
| **Hybrid QNN** | $\infty$ | Target Spec 80% | $0.007673$ | 18 | 40 | 173 | 10 | **$0.6429$** [$0.46, 0.81$] | **$0.8122$** [$0.75, 0.87$] | $0.3103$ | $0.4186$ | $0.7275$ | 58 (24.1%) |
| **Hybrid QNN** | $\infty$ | Target Spec 90% | $0.013515$ | 10 | 23 | 190 | 18 | **$0.3571$** [$0.18, 0.56$] | **$0.8920$** [$0.85, 0.93$] | $0.3030$ | $0.3279$ | $0.6246$ | 33 (13.7%) |
| **Hybrid QNN** | $\infty$ | Target Spec 95% | $0.088928$ | 4 | 4 | 209 | 24 | **$0.1429$** [$0.03, 0.28$] | **$0.9812$** [$0.96, 1.00$] | $0.5000$ | $0.2222$ | $0.5620$ | 8 (3.3%) |
| **Classical MLP** | $\infty$ | Default $\tau=0.50$ | $0.500000$ | 11 | 32 | 181 | 17 | $0.3929$ [$0.20, 0.59$] | $0.8498$ [$0.80, 0.90$] | $0.2558$ | $0.3099$ | $0.6213$ | 43 (17.8%) |
| **Classical MLP** | $\infty$ | Youden's J | $0.080205$ | 21 | 82 | 131 | 7 | **$0.7500$** [$0.58, 0.90$] | **$0.6150$** [$0.54, 0.68$] | $0.2039$ | $0.3206$ | $0.6825$ | 103 (42.7%) |
| **Classical MLP** | $\infty$ | Target Spec 80% | $0.316313$ | 12 | 38 | 175 | 16 | **$0.4286$** [$0.24, 0.62$] | **$0.8216$** [$0.77, 0.87$] | $0.2400$ | $0.3077$ | $0.6251$ | 50 (20.8%) |
| **Classical MLP** | $\infty$ | Target Spec 90% | $0.883112$ | 9 | 18 | 195 | 19 | **$0.3214$** [$0.14, 0.52$] | **$0.9155$** [$0.88, 0.95$] | $0.3333$ | $0.3273$ | $0.6185$ | 27 (11.2%) |
| **Classical MLP** | $\infty$ | Target Spec 95% | $0.921949$ | 7 | 11 | 202 | 21 | **$0.2500$** [$0.08, 0.44$] | **$0.9484$** [$0.92, 0.98$] | $0.3889$ | $0.3043$ | $0.5992$ | 18 (7.5%) |
| **Hybrid QNN** | $8.0$ | Youden's J | $0.011079$ | 12 | 50 | 163 | 16 | $0.4286$ [$0.23, 0.62$] | $0.7653$ [$0.70, 0.83$] | $0.1935$ | $0.2667$ | $0.5969$ | 62 (25.7%) |
| **Hybrid QNN** | $4.0$ | Youden's J | $0.007749$ | 20 | 97 | 116 | 8 | $0.7143$ [$0.52, 0.87$] | $0.5446$ [$0.47, 0.62$] | $0.1709$ | $0.2759$ | $0.6294$ | 117 (48.6%) |
| **Hybrid QNN** | $2.0$ | Youden's J | $0.010198$ | 15 | 63 | 150 | 13 | $0.5357$ [$0.34, 0.72$] | $0.7042$ [$0.64, 0.77$] | $0.1923$ | $0.2830$ | $0.6200$ | 78 (32.4%) |
| **Hybrid QNN** | $1.0$ | Youden's J | $0.009810$ | 12 | 85 | 128 | 16 | $0.4286$ [$0.24, 0.62$] | $0.6009$ [$0.53, 0.67$] | $0.1237$ | $0.1920$ | $0.5148$ | 97 (40.3%) |
| **Classical MLP** | $8.0$ | Youden's J | $0.008374$ | 14 | 51 | 162 | 14 | $0.5000$ [$0.31, 0.70$] | $0.7606$ [$0.70, 0.82$] | $0.2154$ | $0.3011$ | $0.6303$ | 65 (27.0%) |
| **Classical MLP** | $4.0$ | Youden's J | $0.007849$ | 14 | 50 | 163 | 14 | $0.5000$ [$0.30, 0.69$] | $0.7653$ [$0.70, 0.82$] | $0.2188$ | $0.3043$ | $0.6326$ | 64 (26.6%) |
| **Classical MLP** | $2.0$ | Youden's J | $0.007110$ | 15 | 71 | 142 | 13 | $0.5357$ [$0.33, 0.72$] | $0.6667$ [$0.60, 0.73$] | $0.1744$ | $0.2632$ | $0.6012$ | 86 (35.7%) |
| **Classical MLP** | $1.0$ | Youden's J | $0.007880$ | 14 | 55 | 158 | 14 | $0.5000$ [$0.31, 0.72$] | $0.7418$ [$0.68, 0.80$] | $0.2029$ | $0.2887$ | $0.6209$ | 69 (28.6%) |

---

## 3. Clinical Implications of Threshold Selection

### 3.1 The Low-Threshold / High-Sensitivity Regime (Youden's J)
- **Clinical Role:** Rapid response screening, proactive bedside review, non-invasive telemetry check.
- **Performance:** Non-private QNN captures **$75.00\%$ of deaths** ($21/28$), with only $7$ missed deaths out of $241$ encounters.
- **The False-Alarm Toll:** To capture 21 deaths, the model flags **$96$ patients** as high risk ($39.83\%$ of the entire ICU). Among these, **$75$ are false positives** (patients who survive to discharge).
- **Workload Impact:** Nurses and intensivist teams would face $3.57$ false alerts for every dying patient identified. In a busy unit, such alert frequency degrades clinician trust and induces alarm fatigue.

### 3.2 The High-Specificity Triage Regime ($\text{Specificity} \ge 80\%$)
- **Clinical Role:** Resource allocation, palliative care consultation, invasive hemodynamic monitoring.
- **Performance:** At validation-selected $\tau^* = 0.007673$, the QNN achieves **$81.22\%$ test specificity** while maintaining **$64.29\%$ sensitivity** ($18/28$ deaths).
- **False Alarms:** False alarms drop from $75$ down to **$40$**, while total flagged patients drop from $96$ to **$58$** ($24.07\%$ of the ICU).
- **Comparison with Classical MLP:** At the same $80\%$ validation specificity target ($\tau^* = 0.316313$), the Classical MLP achieves **$82.16\%$ test specificity**, but captures only **$42.86\%$ of deaths** ($12/28$, missing $16$ deaths). Here, the QNN provides an absolute $+21.43\%$ sensitivity advantage ($18$ vs. $12$ captured deaths).

### 3.3 The Ultra-High Specificity Regime ($\text{Specificity} \ge 95\%$)
- **Performance:** At validation $\tau^* = 0.088928$, QNN achieves **$98.12\%$ specificity** ($4$ false positives among $213$ survivors), yielding a precision of **$50.00\%$** ($4$ true deaths, $4$ false alarms).
- **Clinical Trade-Off:** The price of high precision is unacceptable under-triage: **$85.71\%$ of deaths are missed** ($24/28$). This threshold is unsuitable for life-critical risk alerts.

---

## 4. Formal Audit Conclusion

1. **Threshold Transparency:** The audit confirms that every reported operating point in Phase 13.3.3 was **derived strictly from the independent validation split**. Zero test labels were accessed during threshold optimization.
2. **Clinical Trade-Off Reality:** Under an $11.62\%$ ICU mortality prevalence, there is no magic operating point that simultaneously eliminates false alarms and captures the vast majority of deaths. High-sensitivity triage requires managing false-alarm ratios of $\approx 3:1$.
3. **Publication Guidance:** The manuscript must present both sensitivity and false-alarm burden across multiple clinical operating points, refraining from claiming that high AUROC alone translates into clinical effectiveness.
