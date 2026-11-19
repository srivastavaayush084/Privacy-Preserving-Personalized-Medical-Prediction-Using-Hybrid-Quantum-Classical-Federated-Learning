# CLINICAL PROBABILITY CALIBRATION PROTOCOL & INFERENCE GOVERNANCE

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase Identifier:** Phase 13.3.6 — Final Cohort, Prediction Artifact and Metric Reconciliation Audit  
**Auditor Role:** Senior Biostatistician, Clinical Machine-Learning Auditor, and Reproducibility Specialist  
**Date:** November 19, 2026  
**Artifact Path:** `research/phase13_3_6/CALIBRATION_PROTOCOL_FINAL.md`  

---

## 1. Executive Summary & Calibration Governance

Probability calibration in clinical risk prediction models requires strict partition hygiene to prevent information leakage and optimistic bias.

This document verifies the two-stage calibration and evaluation protocol implemented in the research baseline, confirming that:
1. **Platt scaling calibrators are fitted strictly on validation data.**
2. **All calibration parameters are frozen prior to test-set evaluation.**
3. **Test-set data is never used to fit, tune, or select calibrator parameters.**
4. **Validation Platt scaling slope ($a_{\text{val}} = 0.4415$) is mathematically and conceptually distinguished from test Cox recalibration slope ($\beta_{\text{test}} = 1.7096$).**
5. **Test Cox recalibration statistics are reported solely as out-of-sample descriptive diagnostic evaluations, never as model training or tuning.**

---

## 2. Formal Two-Stage Mathematical Protocol

### Stage 1: Validation-Split Calibrator Estimation (Training the Calibrator)

On the independent validation split ($\mathcal{D}_{\text{val}}$, $N=234$ encounters, $198$ patients, $24$ deaths):
1. Raw model logits are computed: $z_{\text{val}} = f_{\theta}(X_{\text{val}})$.
2. A univariate logistic calibration model is fitted via maximum likelihood:
   $$\hat{p}_{\text{val}}^{\text{platt}} = \sigma(a_{\text{val}} \cdot z_{\text{val}} + b_{\text{val}}) = \frac{1}{1 + \exp\left(-(a_{\text{val}} \cdot z_{\text{val}} + b_{\text{val}})\right)}$$
3. **Fitted Point Estimates on $\mathcal{D}_{\text{val}}$:**
   - **Hybrid DP-QNN ($\infty$):** $a_{\text{val}} = \mathbf{0.441545}$, $b_{\text{val}} = \mathbf{-0.092357}$
   - **Classical DP-MLP ($\infty$):** $a_{\text{val}} = \mathbf{0.322251}$, $b_{\text{val}} = \mathbf{-1.737721}$
4. **Validation Bootstrap Distribution ($B=1,000$ Resamples):**
   - For QNN, $a_{\text{val}}^{*(b)}$ exhibited mean $= 0.4155$, $\text{SD} = 0.2513$, and $95\%$ percentile CI: $\mathbf{[-0.1176, 0.8572]}$.
   - The wide confidence interval reflecting zero reflects the small number of mortality events ($24$ deaths) in the validation cohort under patient-level clustering.
5. **Parameter Freezing:** The scalars $(a_{\text{val}}, b_{\text{val}})$ are locked and permanently saved into calibration state records.

---

### Stage 2: Out-of-Sample Test Evaluation (Applying the Calibrator)

On the frozen test partition ($\mathcal{D}_{\text{test}}$, $N=241$ encounters, $198$ patients, $28$ deaths):
1. Raw test logits are computed: $z_{\text{test}} = f_{\theta}(X_{\text{test}})$.
2. Calibrated probabilities are evaluated strictly out-of-sample using the frozen validation coefficients:
   $$\hat{p}_{\text{test}}^{\text{platt}} = \sigma(a_{\text{val}} \cdot z_{\text{test}} + b_{\text{val}})$$
   - For QNN: $\hat{p}_{\text{test}}^{\text{platt}} = \sigma(0.441545 \cdot z_{\text{test}} - 0.092357)$
   - Zero test data labels ($y_{\text{test}}$) are observed or utilized during this transformation.
3. Test-set calibrated metrics are computed:
   - QNN Platt Brier: **$0.097960$** (outperforms training prevalence baseline $0.104095$ and test prevalence reference $0.102684$).
   - QNN Platt 10-bin ECE: **$0.044539$**.

---

### Stage 3: Out-of-Sample Calibration Quality Assessment (Descriptive Diagnostic Only)

To evaluate whether the out-of-sample probabilities $\hat{p}_{\text{test}}^{\text{platt}}$ are clinically reliable, standard Cox calibration analysis is performed on the test set:
$$\text{logit}(P(Y_{\text{test}} = 1)) = \alpha_{\text{test}} + \beta_{\text{test}} \cdot \text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$$

- **Calibration Intercept ($\alpha_{\text{test}}$):** Measures overall calibration-in-the-large. Ideal value: $0.0$.
  - QNN: $\alpha_{\text{test}} = \mathbf{0.1081}$ (indicating slight underestimation of average risk).
- **Calibration Slope ($\beta_{\text{test}}$):** Measures spread of probabilities (discrimination and shrinkage). Ideal value: $1.0$.
  - QNN: $\beta_{\text{test}} = \mathbf{1.709588}$ ($\approx \mathbf{1.7096}$, $95\%$ CI: $[0.9576, 2.5976]$).
  - Values $> 1.0$ indicate that the validation-fitted transformation slightly over-shrunk the predictions towards the mean.

---

## 3. Clear Distinction Between Slopes

| Property | Validation Platt Slope ($a_{\text{val}}$) | Test Cox Recalibration Slope ($\beta_{\text{test}}$) |
| :--- | :--- | :--- |
| **Value (QNN $\infty$)** | **$0.4415$** ($0.441545$) | **$1.7096$** ($1.709588$) |
| **95% Confidence Interval** | $[-0.1176, 0.8572]$ (Validation Bootstrap) | $[0.9576, 2.5976]$ (Test Logistic Standard Error) |
| **Fitting Dataset** | Validation Split ($\mathcal{D}_{\text{val}}, N=234, 24\text{ deaths}$) | Test Split ($\mathcal{D}_{\text{test}}, N=241, 28\text{ deaths}$) |
| **Input Feature ($X$)** | Raw Validation Logit ($z_{\text{val}}$) | Calibrated Test Logit ($\text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$) |
| **Target Label ($Y$)** | Validation Mortality ($y_{\text{val}}$) | Test Mortality ($y_{\text{test}}$) |
| **Role in Pipeline** | **Model Parameter** (Used to compute $\hat{p}$) | **Evaluation Diagnostic** (Used to evaluate calibration) |
| **Leakage Risk** | Zero (Fitted exclusively on validation set) | Zero (Descriptive diagnostic only; not used to modify model) |

---

## 4. Scientific Protocol Verification

- **Verification Finding:** The pipeline strictly prevents target leakage.
- **Reporting Requirement:** In the final manuscript, $\beta_{\text{test}} = 1.7096$ must be presented strictly as a descriptive evaluation metric of calibration quality, and must never be characterized as model training, fitting, or optimization.
