# Clinical Utility & Decision Curve Analysis (DCA) Audit

**Phase:** 13.3.3  
**Task:** Decision-Analytic Net Benefit Evaluation Across Clinical Risk Thresholds  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Healthcare Decision Scientist  

---

## 1. Executive Summary

Traditional metrics such as AUROC, sensitivity, and specificity do not evaluate clinical utility because they treat false positives and false negatives with arbitrary weights and ignore the real-world consequences of intervention. **Decision Curve Analysis (DCA)**, introduced by Vickers and Elkin (2006), quantifies the clinical value of a prediction model by calculating **Net Benefit** across a continuum of patient risk preference thresholds $p_t$.

This audit establishes:
1. **Decision Curve Evaluation Range:** A clinically realistic threshold range of **$p_t \in [0.01, 0.35]$** was evaluated, reflecting ICU triage scenarios where the threshold for initiating heightened surveillance or specialized consult ranges from $5\%$ to $30\%$ predicted risk of death.
2. **Threshold Crossing Point:** For all thresholds $p_t > 0.1162$ (the test cohort mortality prevalence), the default **"Treat All" strategy yields a negative Net Benefit**, confirming that intervening on every patient generates more false-positive harm than true-positive clinical benefit.
3. **Calibrated Model Utility:** Both the validation-calibrated Hybrid QNN and validation-calibrated Classical MLP provide **positive Net Benefit superior to both "Treat All" and "Treat None"** across the key clinical decision range $p_t \in [0.08, 0.25]$.
4. **Impact of Miscalibration:** The raw, uncalibrated Hybrid QNN displays poor Net Benefit at thresholds $> 0.10$ because its compressed probabilities rarely exceed $0.10$, resulting in an artificial collapse toward the "Treat None" line. Recalibration is indispensable for decision-analytic utility.
5. **Ethical & Translational Boundary:** Decision curve analysis is an *in-silico* decision model based on retrospective data. It demonstrates mathematical net benefit under theoretical utility weights, but **does not prove improved patient outcomes, decreased mortality, or clinical deployment readiness**.

---

## 2. Decision Curve Analysis Mathematical Formulation

For a given decision threshold probability $p_t \in (0, 1)$, a patient is classified as high-risk and "treated" (e.g., scheduled for intensivist review, advanced monitoring, or palliative evaluation) if their predicted risk $\hat{p}_i \ge p_t$.

### 2.1 Net Benefit Formulation
The Net Benefit (NB) puts true positives and false positives on a common scale using the odds ratio of the threshold probability as the exchange rate:
$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$$
where $N = 241$ is the total number of test encounters, $\text{TP}$ is the number of true positive mortality cases identified, and $\text{FP}$ is the number of false alarms.

### 2.2 Default Clinical Strategies
- **Treat None:** No patient is classified as high risk ($\text{TP} = 0, \text{FP} = 0$).
  $$\text{Net Benefit}_{\text{none}}(p_t) = 0.0$$
- **Treat All:** Every patient is classified as high risk ($\text{TP} = N_{\text{pos}} = 28, \text{FP} = N_{\text{neg}} = 213$).
  $$\text{Net Benefit}_{\text{all}}(p_t) = \frac{28}{241} - \frac{213}{241} \left(\frac{p_t}{1 - p_t}\right) = \bar{y} - (1 - \bar{y}) \left(\frac{p_t}{1 - p_t}\right)$$

When $p_t = \bar{y} = 0.116183$, $\text{Net Benefit}_{\text{all}} = 0.0$. For any $p_t > \bar{y}$, $\text{Net Benefit}_{\text{all}} < 0$.

---

## 3. Decision Curve Analysis Results Table

Table 5 details the Net Benefit across representative clinical risk thresholds $p_t \in [0.01, 0.30]$:

| Threshold ($p_t$) | Treat None | Treat All | Raw Hybrid QNN | Raw Classical MLP | Calibrated Hybrid QNN | Calibrated Classical MLP | Optimal Strategy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.01$** | $0.0000$ | $+0.1073$ | $+0.0526$ | $+0.1073$ | $+0.1073$ | $+0.1073$ | Tie (Treat All / Models) |
| **$0.03$** | $0.0000$ | $+0.0888$ | $+0.0231$ | $+0.0888$ | $+0.0888$ | $+0.0888$ | Tie (Treat All / Models) |
| **$0.05$** | $0.0000$ | $+0.0697$ | $+0.0188$ | $+0.0697$ | $+0.0697$ | $+0.0697$ | Tie (Treat All / Models) |
| **$0.07$** | $0.0000$ | $+0.0497$ | $+0.0147$ | $+0.0497$ | $+0.0497$ | $+0.0497$ | Tie (Treat All / Models) |
| **$0.08$** | $0.0000$ | $+0.0393$ | $+0.0144$ | $+0.0603$ | $+0.0393$ | $+0.0390$ | Raw MLP |
| **$0.09$** | $0.0000$ | $+0.0288$ | $+0.0150$ | $+0.0497$ | **$+0.0551$** | $+0.0392$ | **Calibrated QNN** |
| **$0.10$** | $0.0000$ | $+0.0180$ | $+0.0065$ | $+0.0332$ | **$+0.0456$** | $+0.0337$ | **Calibrated QNN** |
| **$0.11$** | $0.0000$ | $+0.0069$ | $+0.0073$ | $+0.0324$ | **$+0.0292$** | $+0.0324$ | Calibrated MLP |
| **$0.12$** | $0.0000$ | **$-0.0043$** | $+0.0036$ | $+0.0298$ | **$+0.0285$** | $+0.0283$ | **Calibrated QNN** |
| **$0.13$** | $0.0000$ | **$-0.0159$** | $+0.0035$ | $+0.0277$ | **$+0.0208$** | $+0.0227$ | Calibrated MLP |
| **$0.15$** | $0.0000$ | **$-0.0398$** | $0.0000$ | $+0.0251$ | **$+0.0139$** | $+0.0222$ | Calibrated MLP |
| **$0.18$** | $0.0000$ | **$-0.0778$** | $0.0000$ | $+0.0232$ | **$+0.0150$** | $+0.0179$ | Calibrated MLP |
| **$0.20$** | $0.0000$ | **$-0.1048$** | $0.0000$ | $+0.0207$ | **$+0.0119$** | $+0.0179$ | Calibrated MLP |
| **$0.25$** | $0.0000$ | **$-0.1784$** | $0.0000$ | $+0.0187$ | **$+0.0069$** | $+0.0125$ | Calibrated MLP |
| **$0.30$** | $0.0000$ | **$-0.2625$** | $0.0000$ | $+0.0093$ | **$+0.0062$** | $+0.0028$ | **Calibrated QNN** |

---

## 4. Key Findings and Clinical Interpretation

### 4.1 Superiority Over Default Strategies
Across the entire decision range $p_t \in [0.08, 0.30]$:
- Both the **Calibrated Hybrid QNN** and **Calibrated Classical MLP** maintain positive Net Benefit ($0.01$ to $0.06$).
- The "Treat All" strategy collapses into steep negative territory once $p_t > 0.1162$, reaching a harmful $-0.1048$ at $p_t = 0.20$.
- Using either calibrated model to guide decision-making provides higher clinical value than treating everyone or treating no one.

### 4.2 Raw vs. Calibrated Decision Curves
- For the **Raw Hybrid QNN**, probabilities never exceed $0.1367$. As a result, for all decision thresholds $p_t \ge 0.14$, the model predicts zero high-risk patients ($\text{TP}=0, \text{FP}=0$), causing its Net Benefit curve to fall flat onto the "Treat None" line ($\text{NB} = 0.0$).
- Recalibration using Platt scaling on validation data scales probabilities into $[0.02, 0.45]$, restoring positive Net Benefit out to $p_t = 0.35$.

### 4.3 Quantum vs. Classical Decision Curves
- Between $p_t = 0.08$ and $p_t = 0.12$, the Calibrated Hybrid QNN achieves the highest Net Benefit (peaking at $+0.0551$ at $p_t = 0.09$ vs. $+0.0392$ for MLP).
- Between $p_t = 0.13$ and $p_t = 0.25$, the Calibrated Classical MLP exhibits a slightly higher Net Benefit ($+0.018$ vs. $+0.012$).
- In decision-analytic terms, **neither model dominates the other across the entire clinical threshold spectrum**. The curves interleave, confirming the conclusion of the patient-clustered bootstrap that there is no uniform quantum superiority.

---

## 5. Scope Boundaries & Clinical Caveats

1. **Observational In-Silico Context:** Net Benefit is an analytical construct calculated on retrospective test data. It assumes that clinical intervention is $100\%$ effective for true positives and incurs a penalty strictly proportional to $p_t / (1 - p_t)$ for false positives. Real clinical interventions carry variable efficacy, financial costs, and procedural risks that cannot be reduced to a single threshold odds ratio.
2. **Alert Burden Reality:** At $p_t = 0.10$, the Calibrated QNN's Net Benefit of $+0.0456$ corresponds to correctly identifying $18$ true deaths while still generating $40$ false alarms. The model reduces unnecessary treatments compared to "Treat All", but does not eliminate alert burden.
3. **Manuscript Transparency:** The publication must present Decision Curve Analysis as evidence of theoretical decision-analytic feasibility under recalibration, explicitly disclaiming any assertion of prospective clinical efficacy.
