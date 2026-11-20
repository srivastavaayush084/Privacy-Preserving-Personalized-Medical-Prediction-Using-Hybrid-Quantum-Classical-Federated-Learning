# Decision Curve Analysis (DCA), Net Benefit Interpretation, and Clinical Calibration Audit

**Author:** Senior Biostatistician, Clinical Machine Learning Auditor, Decision Science Specialist  
**Date:** November 20, 2026  
**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Status:** Certified Audit Document — Phase 14.1  

---

## 1. Executive Summary

This audit examines the mathematical formulation, clinical interpretation, and reporting of **Decision Curve Analysis (DCA)** and **calibration metrics** across the research project and the IEEE research manuscript.

### Key Audit Findings & Corrective Actions
1. **Retraction of the Literal "26.7 Interventions" Claim:**
   - Earlier project drafts asserted that the calibrated QNN *"avoided 26.7 false-positive interventions per 100 encounters without missing true positive deaths"*.
   - This assertion has been **completely retracted**. It confused a theoretical algebraic conversion of net benefit with an empirical prospective clinical finding.
2. **Adoption of Authoritative, Evidence-Based Wording:**
   - In `IEEE_Research_Paper_CORRECTED.tex` (Abstract, Section VIII-B), the text was replaced with the strictly verified formulation:
     > *"At a decision threshold of 0.09, the calibrated QNN achieved a net benefit of 0.0551, compared with 0.0288 for the Treat-All strategy. This represents an exploratory improvement in net benefit of approximately 0.0264. The analysis does not establish clinical effectiveness or a reduction in actual interventions."*
3. **Prevalence Baseline Nomenclature Formalized:**
   - The test-set prevalence baseline ($p_{\text{baseline}} = 28 / 241 = 11.62\%$, Brier score $0.1027$) is explicitly designated as an **oracle retrospective reference**, because computing it requires prior knowledge of the test set outcome distribution.
   - The training prevalence baseline ($p_{\text{train}} = 132 / 1,132 = 11.66\%$, Brier score $0.1041$) represents the deployable prior.
   - Out-of-sample Platt-scaled models (QNN Brier $0.0980$, MLP Brier $0.0964$) strictly outperform both prevalence baselines.
4. **Resolution of Calibration Slope Nomenclature:**
   - The validation Platt scaling transformation parameter ($a_{\text{val}} = 0.4415, b_{\text{val}} = -0.0924$) is clearly distinguished from the out-of-sample test Cox calibration diagnostic slope ($\beta_{\text{test}} = 1.7096, \alpha_{\text{test}} = 0.1081$).

---

## 2. Mathematical Deconstruction of the Withdrawn Claim

### 2.1 The Net Benefit Formulation (Vickers & Elkin, 2006)
Decision Curve Analysis quantifies clinical utility under a continuum of decision thresholds $p_t$:
$$\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$$

Where:
- $\text{TP}/N$ is the true positive rate in the population (benefit of detection).
- $\text{FP}/N$ is the false positive rate in the population (harm of unnecessary intervention).
- $w = \frac{p_t}{1 - p_t}$ is the exchange rate weighing false positives against true positives.

For the default strategies:
$$\text{NB}_{\text{Treat-None}}(p_t) = 0$$
$$\text{NB}_{\text{Treat-All}}(p_t) = \text{Prevalence} - (1 - \text{Prevalence}) \left(\frac{p_t}{1 - p_t}\right)$$

### 2.2 Numerical Trajectory on Frozen Test Cohort ($N=241, 28$ Deaths)
At threshold $p_t = 0.09$:
$$\text{Prevalence} = \frac{28}{241} \approx 0.11618$$
$$\text{NB}_{\text{Treat-All}}(0.09) = 0.11618 - (1 - 0.11618) \left(\frac{0.09}{0.91}\right) = 0.11618 - 0.88382 \times 0.09890 = \mathbf{0.02877} \approx 0.0288$$
$$\text{NB}_{\text{QNN}}(0.09) = \mathbf{0.05514} \approx 0.0551$$
$$\Delta \text{NB} = 0.05514 - 0.02877 = \mathbf{+0.02637} \approx +0.0264$$

### 2.3 Why the "26.7 Avoided Interventions" Metric Was Invalid
In decision science literature, an algebraic quantity called *"reduction in net interventions per 100 patients"* is sometimes computed as:
$$\Delta \text{Interventions} = \Delta \text{NB} \times \left(\frac{1 - p_t}{p_t}\right) \times 100 = 0.02637 \times 10.111 \times 100 \approx \mathbf{26.66 \approx 26.7}$$

However, presenting this as an empirical medical finding suffered from three fatal clinical and statistical flaws:
1. **Unequal True Positives:** The conversion formula mathematically requires that the model identifies *the exact same number of true cases* as Treat-All. But Treat-All identifies all 28 cases ($100\%$ sensitivity), whereas a model operating at threshold $p_t = 0.09$ selects a risk stratum and inevitably misses a portion of true deaths. Claiming "without missing cases" was factually untrue.
2. **Observational vs Interventional Setting:** The study analyzed retrospective observational records from the eICU database. No clinician acted on these model scores, and no clinical interventions were actually withheld or administered.
3. **Misleading Clinical Guidance:** In an intensive care unit, claiming 26.7 avoided interventions per 100 patients could encourage clinicians to prematurely de-escalate monitoring without prospective safety trial validation.

---

## 3. Decision Curve Analysis Summary Table

| Threshold ($p_t$) | Exchange Rate ($w$) | Treat-None | Treat-All | Calibrated QNN | Calibrated MLP | Preferred Strategy |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 0.0526 | 0.0000 | +0.0697 | +0.0697 | +0.0697 | 3-Way Tie (All above threshold) |
| **0.08** | 0.0870 | 0.0000 | +0.0393 | +0.0393 | +0.0390 | QNN = Treat-All |
| **0.09** | 0.0989 | 0.0000 | +0.0288 | **+0.0551** | +0.0392 | **Calibrated QNN** ($\Delta \text{NB} = +0.0264$) |
| **0.10** | 0.1111 | 0.0000 | +0.0180 | **+0.0456** | +0.0337 | **Calibrated QNN** ($\Delta \text{NB} = +0.0276$) |
| **0.12** | 0.1364 | 0.0000 | -0.0043 | **+0.0285** | +0.0283 | **Calibrated QNN** |
| **0.15** | 0.1765 | 0.0000 | -0.0398 | +0.0139 | **+0.0222** | **Calibrated MLP** |
| **0.20** | 0.2500 | 0.0000 | -0.1048 | **+0.0083** | +0.0083 | Tie |
| **0.25** | 0.3333 | 0.0000 | -0.1784 | **+0.0041** | +0.0041 | Tie |
| **0.30** | 0.4286 | 0.0000 | -0.2626 | 0.0000 | 0.0000 | Collapses to Treat-None |

### Key Clinical Observations
- **Clinical Utility Window:** Both calibrated models demonstrate positive net benefit superior to both Treat-All and Treat-None in the threshold range $p_t \in [0.09, 0.25]$.
- **Safe Collapse:** At decision thresholds $p_t \ge 0.30$, both models predict no patients exceeding threshold, safely falling back to the Treat-None baseline without negative net benefit.

---

## 4. Calibration Hierarchy and Baseline Nomenclature

| Baseline / Model Strategy | Description | Brier Score | Expected Calibration Error (ECE) | Operational Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Training Prevalence Prior** | Predicts constant $\bar{y}_{\text{train}} = 132/1,132 \approx 0.1166$ | 0.1041 | 0.0139 | Deployable unconditional empirical prior |
| **Oracle Test Prevalence** | Predicts constant $\bar{y}_{\text{test}} = 28/241 \approx 0.1162$ | 0.1027 | 0.0000 | **Retrospective theoretical oracle baseline** |
| **Raw Uncalibrated QNN** | Sigmoid of raw network logit output | 0.1106 | 0.0573 | Bounded outputs, but miscalibrated |
| **Raw Uncalibrated MLP** | Sigmoid of raw network logit output | 0.1621 | 0.1364 | Severe overconfidence on high-risk tails |
| **Calibrated Hybrid QNN** | Out-of-sample validation Platt scaling | **0.0980** | **0.0445** | Superior to both prevalence baselines |
| **Calibrated Classical MLP** | Out-of-sample validation Platt scaling | **0.0964** | **0.0286** | Superior to both prevalence baselines |

---

## 5. Audit Conclusion

The Decision Curve Analysis and calibration evidence are now reported with rigorous, responsible scientific terminology. All unjustified claims of avoided interventions have been expunged, and the test prevalence baseline is accurately identified as an oracle retrospective benchmark.
