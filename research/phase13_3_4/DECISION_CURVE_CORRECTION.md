# DECISION CURVE ANALYSIS CORRECTION & CLINICAL DECISION-ANALYTIC AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.4 — Final Statistical Audit, Calibration Validation and Decision-Curve Correction  
**Author:** Senior Medical AI Researcher, Biostatistician & DP Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_4/DECISION_CURVE_CORRECTION.md`  

---

## 1. Executive Summary & Defect Correction

In the Phase 13.3.3 audit report, the narrative asserted:
> *"Validation-calibrated models outperform Treat-All and Treat-None across the entire decision threshold range of $p_t \in [0.08, 0.25]$."*

An independent numerical re-verification of the decision-analytic curves reveals that this statement was **partially inaccurate at the lower boundary**:
- At threshold probability **$p_t = 0.08$**:
  - The Treat-All strategy achieves a Net Benefit of **$0.039329$**.
  - Calibrated $\text{QNN}_{\varepsilon=\infty}$ achieves an identical Net Benefit of **$0.039329$** (an exact tie to 6 decimal places).
  - Calibrated $\text{MLP}_{\varepsilon=\infty}$ achieves a Net Benefit of **$0.038968$** (which is **inferior** to Treat-All).
- The true crossover point where both calibrated models strictly outperform Treat-All and Treat-None is **$p_t \ge 0.09$** (spanning $p_t \in [0.09, 0.25]$).

In this report, we formally correct the decision curve narrative, provide the exact numerical Net Benefit table, detail the clinical decision context, and establish the mandatory publication framing.

---

## 2. Mathematical Formulation of Decision Curve Analysis (DCA)

Decision Curve Analysis (Vickers & Elkin, 2006) evaluates clinical utility by calculating the **Standardized Net Benefit** across a spectrum of decision threshold probabilities $p_t \in (0, 1)$:
$$\text{Net Benefit}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left( \frac{p_t}{1 - p_t} \right)$$
where:
- $\text{TP}$ is the number of true positive predictions at threshold $p_t$.
- $\text{FP}$ is the number of false positive predictions at threshold $p_t$.
- $N = 241$ is the total number of encounters in the test cohort.
- $w(p_t) = \frac{p_t}{1 - p_t}$ is the **odds weight**, representing the relative clinical harm of a false positive compared to the benefit of a true positive.

### 2.1 Default Clinical Strategies
1. **Treat None:** Assumes no patient receives the intervention ($\text{TP} = 0, \text{FP} = 0$):
   $$\text{NB}_{\text{None}}(p_t) = 0.0 \quad \forall p_t$$
2. **Treat All:** Assumes every patient receives the intervention ($\text{TP} = N_{\text{pos}} = 28, \text{FP} = N - N_{\text{pos}} = 213$):
   $$\text{NB}_{\text{All}}(p_t) = \text{Prev} - (1 - \text{Prev}) \left( \frac{p_t}{1 - p_t} \right) = 0.116183 - 0.883817 \left( \frac{p_t}{1 - p_t} \right)$$

---

## 3. Audited Net Benefit Table Across Critical Thresholds

The table below presents the verified Net Benefit across representative threshold probabilities extracted from `results/phase13_3_4/decision_curve_analysis_corrected.csv`:

| Threshold $p_t$ | Odds Weight $w(p_t)$ | Treat None | Treat All | Raw QNN ($\infty$) | Raw MLP ($\infty$) | Platt QNN ($\infty$) | Platt MLP ($\infty$) | Optimal Strategy | Clinical Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $0.01$ | $0.0101$ | $0.0000$ | **$0.1073$** | $0.0526$ | **$0.1073$** | **$0.1073$** | **$0.1073$** | Treat All / Calibrated | Tolerates 100 FPs per TP |
| $0.05$ | $0.0526$ | $0.0000$ | **$0.0697$** | $0.0188$ | **$0.0697$** | **$0.0697$** | **$0.0697$** | Treat All / Calibrated | Tolerates 20 FPs per TP |
| **$0.08$** | **$0.0870$** | **$0.0000$** | **$0.0393$** | **$0.0144$** | **$0.0603$** | **$0.0393$** | **$0.0390$** | **Treat All $\approx$ Platt QNN** | **Crossover Boundary: Treat All ties QNN, beats MLP** |
| **$0.09$** | **$0.0989$** | **$0.0000$** | **$0.0288$** | **$0.0150$** | **$0.0497$** | **$0.0551$** | **$0.0392$** | **Platt QNN** | **Strict Dominance Begins ($p_t \ge 0.09$)** |
| $0.10$ | $0.1111$ | $0.0000$ | $0.0180$ | $0.0065$ | $0.0332$ | **$0.0456$** | $0.0337$ | Platt QNN | Models strictly beat Treat All |
| $0.12$ | $0.1364$ | $0.0000$ | -0.0043 | $0.0036$ | $0.0298$ | **$0.0285$** | $0.0283$ | Platt QNN | Treat All becomes harmful (NB < 0) |
| $0.15$ | $0.1765$ | $0.0000$ | -0.0398 | $0.0000$ | $0.0251$ | $0.0139$ | **$0.0222$** | Platt MLP | Moderate ICU risk threshold |
| $0.20$ | $0.2500$ | $0.0000$ | -0.1048 | $0.0000$ | $0.0083$ | $0.0114$ | **$0.0207$** | Platt MLP | Tolerates 4 FPs per TP |
| $0.25$ | $0.3333$ | $0.0000$ | -0.1784 | $0.0000$ | -0.0041 | $0.0069$ | **$0.0111$** | Platt MLP | High-specificity decision threshold |
| $0.30$ | $0.4286$ | **$0.0000$** | -0.2626 | $0.0000$ | -0.0178 | **$0.0000$** | **$0.0000$** | Treat None | Threshold exceeds calibrated risk distribution |

---

## 4. Clinical Context & Threshold Justification

In Decision Curve Analysis, the threshold probability $p_t$ is not an arbitrary hyperparameter; it directly parameterizes the clinician's **decision-analytic trade-off** between the harm of an unnecessary intervention ($\text{FP}$) and the harm of failing to treat a deteriorating patient ($\text{FN}$):
$$\frac{\text{Harm}(\text{FP})}{\text{Benefit}(\text{TP})} = \frac{p_t}{1 - p_t}$$

### 4.1 Hypothetical Clinical Interventions at the 24-Hour Landmark
In an ICU workflow at 24 hours post-admission, automated risk predictions can drive specific non-invasive or low-to-moderate-risk clinical actions:
1. **Delayed ICU Step-Down Discharge:** Delaying transfer to a general medical-surgical ward for 24 hours of continued surveillance.
2. **Senior Intensivist Consultation:** Scheduling a planned multidisciplinary bedside review of organ support and lactate trends.
3. **Invasive Hemodynamic Monitoring:** Placement of an arterial line or continuous cardiac output monitoring.

### 4.2 Why the Range $p_t \in [0.09, 0.25]$ is Clinically Meaningful
- **Lower Bound ($p_t = 0.09$):** Corresponds to an odds weight of $\approx 1:10$. The intensivist considers missing a preventable death to be 10 times worse than a false alarm. In this regime, intervening on patients with $>9\%$ predicted mortality prevents deaths while avoiding the indiscriminate resource waste of treating 100% of the unit.
- **Upper Bound ($p_t = 0.25$):** Corresponds to an odds weight of $\approx 1:3$. The intensivist tolerates only 3 false alarms per true death prevented (e.g., when the intervention carries non-trivial procedural risks, such as central venous cannulation).
- **Why Not $p_t > 0.25$?** In a population with $11.6\%$ baseline mortality, setting $p_t > 0.25$ requires the model to identify extreme outliers. Because true in-hospital deaths are distributed across a broad spectrum of moderate severity scores at 24 hours, setting $p_t > 0.25$ suppresses sensitivity, causing the model to default to the "Treat None" line.

---

## 5. Mandatory Manuscript Framing: Exploratory Decision Analysis

To ensure scientific integrity and prevent reviewer pushback regarding clinical efficacy claims, Phase 14 manuscript drafting must strictly enforce the following language:

> [!CAUTION]
> **Prohibited Claim:** The manuscript must NOT state that the model "improves patient outcomes," "reduces ICU mortality," or "is clinically validated."

> [!NOTE]
> **Prescribed Manuscript Phrasing:**  
> *"Decision Curve Analysis was conducted as an exploratory decision-analytic assessment to quantify potential clinical net benefit under hypothetical triage policies. In this retrospective simulation, validation-calibrated hybrid quantum and classical models demonstrated positive net benefit over default 'treat-all' and 'treat-none' strategies across decision thresholds between 9% and 25%. However, this decision-analytic evaluation does not constitute clinical trial evidence, and prospective interventional studies are necessary to evaluate whether algorithm-guided clinical actions impact clinical outcomes."*
