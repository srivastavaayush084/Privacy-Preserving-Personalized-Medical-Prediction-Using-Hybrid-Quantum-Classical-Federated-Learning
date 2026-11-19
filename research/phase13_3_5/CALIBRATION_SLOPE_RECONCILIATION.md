# CALIBRATION COEFFICIENT RECONCILIATION & STATISTICAL QUANTITY AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.5 — Final Source-Level Verification and Audit Closure  
**Lead Auditor:** Senior Biostatistician & Medical AI Reproducibility Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_5/CALIBRATION_SLOPE_RECONCILIATION.md`  

---

## 1. Executive Summary & Root Cause Analysis

During the Phase 13.3.4 review, an apparent numerical conflict was identified across three reported quantities associated with the calibration of the non-private hybrid quantum neural network ($\text{QNN}_{\varepsilon=\infty}$):

1. **Reported Value A:** *"QNN validation Platt scaling slope = 1.7096"* (in executive summary notes).
2. **Reported Value B:** *"QNN validation-resampling slope 95% CI: [-0.1176, 0.8572]"* (in `platt_scaling_stability.csv`).
3. **Reported Value C:** *"QNN calibrated test calibration slope = 1.7096"* (in `prevalence_baseline_comparison.csv`).

### Audit Finding
An exhaustive inspection of the underlying Python source code (`experiments/phase13_3_4/run_phase13_3_4_audit.py`) reveals that **these values represent two completely different statistical quantities fitted on two different datasets with different input variables**:

- **Quantity 1: The Platt Transformation Parameter ($a_{\text{val}}$):**  
  The actual slope parameter fitted on the **validation dataset** ($N_{\text{val}} = 234, 24\text{ deaths}$) is **$a_{\text{val}} = 0.4415$** (with validation bootstrap 95% CI: **$[-0.1176, 0.8572]$**).
- **Quantity 2: The Test-Set Recalibration Slope ($\beta_{\text{test}}$ / Cox Calibration Slope):**  
  The slope of the logistic regression of true **test labels** ($y_{\text{test}}$) on test calibrated log-odds ($\text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$) is **$\beta_{\text{test}} = 1.7096$** (with calibration intercept $\alpha_{\text{test}} = 0.1081$).

In Phase 13.3.4's narrative discussion, the label *"validation Platt scaling slope"* was erroneously applied to the test-set recalibration slope ($1.7096$). The underlying saved prediction arrays, checkpoint weights, and generated CSV tables are mathematically sound and unaltered. This report provides the definitive mathematical reconciliation, source-code mapping, and corrected documentation.

---

## 2. Mathematical Definitions & Source-Code Mapping

```
                                      STAGE 1: MODEL TRAINING
                                      (Training Data: N = 1,132)
                                                 │
                                                 ▼
                                     Raw Logits Produced: z
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             STAGE 2: CALIBRATOR FITTING                       STAGE 3: CALIBRATION AUDIT
             (Validation Data: N = 234)                        (Test Data: N = 241)
                        │                                                 │
      Fit Platt Logistic Regression:                    Compute Test Calibration Slope:
        logit(P(y_val=1)) = a * z_val + b                logit(P(y_test=1)) = alpha + beta * logit(p_cal)
                        │                                                 │
         Fitted Slope: a_val = 0.4415                      Test Slope: beta_test = 1.7096
      (Resampling 95% CI: [-0.118, 0.857])              (Test Intercept: alpha_test = 0.1081)
```

### 2.1 Quantity 1: Platt Scaling Calibrator Parameter ($a_{\text{val}}$)
- **Mathematical Formulation:**
  Let $z_{i, \text{val}} \in \mathbb{R}$ denote the raw scalar logit output from the neural network or quantum circuit for validation encounter $i$. Platt scaling models the posterior probability of mortality via a 1-dimensional logistic regression:
  $$\hat{p}_{i, \text{val}}^{\text{platt}} = \sigma(a_{\text{val}} z_{i, \text{val}} + b_{\text{val}}) = \frac{1}{1 + \exp(-(a_{\text{val}} z_{i, \text{val}} + b_{\text{val}}))}$$
  where $(a_{\text{val}}, b_{\text{val}})$ minimize the L2-regularized binary cross-entropy on $\mathcal{D}_{\text{val}}$.
- **Source-Code Location:**
  `experiments/phase13_3_4/run_phase13_3_4_audit.py`, lines 210–213:
  ```python
  # Fit Platt Calibrator strictly on Validation Split
  platt = LogisticRegression(solver="lbfgs", C=1.0)
  platt.fit(l_v.reshape(-1, 1), y_val)
  slope_a = float(platt.coef_[0][0])
  intercept_b = float(platt.intercept_[0])
  ```
- **Audited Values for $\text{QNN}_{\varepsilon=\infty}$:**
  - `slope_a` ($a_{\text{val}}$): **$0.4415$**
  - `intercept_b` ($b_{\text{val}}$): **$-0.0924$**

### 2.2 Quantity 2: Validation-Bootstrap Distribution of Platt Parameter ($a_{\text{val}}^{(b)}$)
- **Mathematical Formulation:**
  Evaluates the parameter estimation stability of $a_{\text{val}}$ under $B = 1,000$ patient-clustered bootstrap resamples of $\mathcal{D}_{\text{val}}$:
  $$a_{\text{val}}^{(b)} = \arg\max_a \mathcal{L}_{\text{val}}^{(b)}(a, b), \quad b = 1, \dots, 1000$$
- **Source-Code Location:**
  `experiments/phase13_3_4/run_phase13_3_4_audit.py`, lines 248–261:
  ```python
  clf_b = LogisticRegression(solver="lbfgs", C=1.0)
  clf_b.fit(l_v_b.reshape(-1, 1), y_val_b)
  a_b = float(clf_b.coef_[0][0])
  b_b = float(clf_b.intercept_[0])
  ```
- **Audited Values for $\text{QNN}_{\varepsilon=\infty}$:**
  - Mean $a_{\text{val}}$: **$0.4155$**
  - Standard Deviation: **$0.2513$**
  - 95% Percentile Confidence Interval: **$[-0.1176, 0.8572]$**
  - Mean Intercept $b_{\text{val}}$: **$-0.2420$** (95% CI: **$[-2.7032, 1.8989]$**)

### 2.3 Quantity 3: Test-Set Recalibration Slope ($\beta_{\text{test}}$ / Cox Calibration Slope)
- **Mathematical Formulation:**
  Standard biostatistical assessment of calibration on an independent test cohort (Cox, 1958; Steyerberg et al., 2010). True test outcomes $y_{\text{test}}$ are regressed on the log-odds of the calibrated test predictions:
  $$\text{logit}(P(y_{i, \text{test}} = 1)) = \alpha_{\text{test}} + \beta_{\text{test}} \cdot \text{logit}(\hat{p}_{i, \text{test}}^{\text{platt}})$$
  - An ideal model on the test set achieves **$\alpha_{\text{test}} = 0.0$** (calibration-in-the-large / intercept) and **$\beta_{\text{test}} = 1.0$** (calibration slope).
  - A slope $\beta_{\text{test}} > 1.0$ indicates that the predictions are slightly under-dispersed on the test set (relative to the validation set), whereas $\beta_{\text{test}} < 1.0$ indicates over-dispersion.
- **Source-Code Location:**
  `experiments/phase13_3_4/run_phase13_3_4_audit.py`, lines 77–103, 278–279:
  ```python
  def compute_calibration_intercept_slope(probs: np.ndarray, labels: np.ndarray):
      probs_clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
      lp = np.log(probs_clipped / (1.0 - probs_clipped))
      ...
      clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
      clf.fit(lp.reshape(-1, 1), labels)
      cal_slope = float(clf.coef_[0][0])
      cal_intercept_full = float(clf.intercept_[0])
  ```
- **Audited Values for $\text{QNN}_{\varepsilon=\infty}$:**
  - `cal_slope` ($\beta_{\text{test}}$): **$1.7096$**
  - `cal_intercept` ($\alpha_{\text{test}}$): **$0.1081$**

---

## 3. Authoritative Reconciliation Table Across All Models

The table below reconciles all slope and intercept parameters across both model families and all evaluated privacy budgets, extracted directly from the verified CSV outputs (`results/phase13_3_4/platt_scaling_stability.csv` and `results/phase13_3_4/prevalence_baseline_comparison.csv`):

| Model Architecture | $\varepsilon$ | Validation Platt Slope $a_{\text{val}}$ | Validation Platt Slope 95% CI | Validation Platt Intercept $b_{\text{val}}$ | Validation Platt Intercept 95% CI | Raw Test Calib Slope $\beta_{\text{raw}}$ | Calibrated Test Calib Slope $\beta_{\text{test}}$ | Calibrated Test Calib Intercept $\alpha_{\text{test}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical DP-MLP | $1.0$ | -0.0994 | [-1.0040, 0.8293] | -2.6567 | [-7.0732, 1.9491] | 1.3359 | -13.4414 | 0.1393 |
| Classical DP-MLP | $2.0$ | 0.2670 | [-0.6657, 1.2264] | -0.8481 | [-5.5485, 3.7830] | 1.5676 | 5.8719 | 0.1423 |
| Classical DP-MLP | $4.0$ | 0.6513 | [-0.2951, 1.6034] | 1.0078 | [-3.8647, 5.7407] | 1.9183 | 2.9430 | 0.1440 |
| Classical DP-MLP | $8.0$ | 0.8588 | [-0.0785, 1.7997] | 1.9569 | [-2.7501, 6.6194] | 2.2831 | 2.6586 | 0.1422 |
| **Classical DP-MLP** | $\infty$ | **0.3223** | **[0.0760, 0.5480]** | **-1.7377** | **[-2.4469, -1.2544]** | **0.3377** | **1.0494** | **0.0766** |
| Hybrid DP-QNN | $1.0$ | 0.7563 | [-0.2703, 1.8273] | 1.3100 | [-3.6124, 6.2966] | 0.2400 | 0.3145 | 0.1166 |
| Hybrid DP-QNN | $2.0$ | 1.2717 | [0.2640, 2.3017] | 3.7453 | [-1.1845, 8.6351] | 0.6830 | 0.5357 | 0.0900 |
| Hybrid DP-QNN | $4.0$ | 1.3057 | [0.4296, 2.1549] | 3.9896 | [-0.2834, 7.9197] | 0.8519 | 0.6519 | 0.0646 |
| Hybrid DP-QNN | $8.0$ | 1.1334 | [0.3800, 1.8511] | 3.2503 | [-0.4895, 6.6190] | 0.8404 | 0.7404 | 0.0489 |
| **Hybrid DP-QNN** | $\infty$ | **0.4415** | **[-0.1176, 0.8572]** | **-0.0924** | **[-2.7032, 1.8989]** | **0.7526** | **1.7096** | **0.1081** |

---

## 4. Formal Narrative Correction for Manuscript & Audits

### Defective Phrasing (in Phase 13.3.4 Draft Summary):
> *"For QNN inf, fitted Platt parameters on validation: $a = 1.7096$ (slope), $b = +0.1081$ (intercept)... with slope 95% CI [-0.1176, 0.8572]."*

### Corrected, Authoritative Statement:
> *"For the non-private hybrid QNN ($\varepsilon = \infty$), the Platt scaling parameters estimated on the validation cohort ($N_{\text{val}} = 234, 24\text{ deaths}$) are slope $a_{\text{val}} = 0.4415$ and intercept $b_{\text{val}} = -0.0924$. Under 1,000 patient-clustered validation bootstrap resamples, the estimated slope demonstrates moderate sampling variability with a 95% confidence interval of $[-0.1176, 0.8572]$ (standard deviation $0.2513$). When the validation-frozen calibrator is applied out-of-sample to the test cohort ($N_{\text{test}} = 241, 28\text{ deaths}$), the resulting test calibration assessment yields a calibration intercept of $\alpha_{\text{test}} = 0.1081$ and a test recalibration slope of $\beta_{\text{test}} = 1.7096$."*

### Conclusion:
The apparent discrepancy is completely resolved. No numerical errors, code bugs, or label misalignments exist in the underlying data or prediction pipelines. The issue was purely a nomenclature collision between the **calibrator parameter $a_{\text{val}}$** and the **evaluative test recalibration slope $\beta_{\text{test}}$**, which is now permanently corrected in project documentation.
