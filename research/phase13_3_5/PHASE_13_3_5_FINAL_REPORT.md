# PHASE 13.3.5 — FINAL SOURCE-LEVEL VERIFICATION AND AUDIT CLOSURE REPORT

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Evaluation Role:** Senior Biostatistician, Medical AI Auditor, Differential Privacy Reviewer, and Reproducibility Specialist  
**Phase Identifier:** Phase 13.3.5  
**Date of Audit Closure:** November 19, 2026  
**Status:** Audit Completed — Baseline Formally Frozen — **DECISION A**

---

## 1. EXECUTIVE SUMMARY & AUDIT OBJECTIVE

Phase 13.3.5 was initiated to conduct an independent, source-code-level verification of the Phase 13.3.4 statistical analysis, differential privacy accounting, calibration reconciliation, validation resampling distributions, patient-clustered test bootstrapping, and decision-curve analysis before freezing the scientific research baseline.

In accordance with strict audit governance:
- **No models were retrained**, no cohorts were regenerated, no clinical labels were modified, no data splits were altered, and no prior empirical results were overwritten.
- All source files, checkpoints, test predictions, and scripts were inspected directly and tested using reproducible commands.
- All discrepancies noted in Phase 13.3.4 documentation were systematically reconciled against raw execution traces and python scripts.

### Summary of Major Audit Findings

1. **Calibration Slope Reconciliation:** The reported values $1.7096$ and $95\%\ \text{CI}\ [-0.1176, 0.8572]$ represent entirely different statistical quantities from distinct phases of the calibration pipeline. $a_{\text{val}} = 0.4415$ (with bootstrap $95\%\ \text{CI}\ [-0.1176, 0.8572]$) is the Platt transformation parameter fitted on the validation cohort ($N=234$), whereas $\beta_{\text{test}} = 1.7096$ is the post-calibration Cox calibration slope evaluated on the frozen test set ($N=241$). The underlying mathematics and test predictions are completely correct and reproducible.
2. **Validation Resampling Integrity:** All 1,000 validation bootstrap resamples were drawn without dropping, discarding, or encountering single-class degeneracies. The previously noted "10,000 refits" corresponds to 10 distinct model configurations refit across 1,000 resamples each ($10 \times 1,000 = 10,000$), all of which converged cleanly with zero numerical failures.
3. **Patient-Clustered Test Bootstrap Audit:** All 2,000 test bootstrap iterations resampled intact patient clusters ($198$ unique patients across $241$ encounters). Paired hypothesis testing confirmed statistical parity between QNN and MLP on AUROC ($p = 0.290$), AUPRC ($p = 0.624$), and Platt Brier score ($p = 0.629$). No statistically significant superiority exists for either architecture.
4. **Complete Test Suite Execution:** The entire project test suite was executed from root: **70 passed, 0 failed, 0 skipped** (Exit Code: `0`, Duration: 104.56s). The earlier Exit Code 1 reported in Phase 13.3.4 was traced to PowerShell failing to locate `pytest.exe` directly on the Windows user PATH, whereas `python -m pytest` executes seamlessly with zero errors.
5. **Decision Curve Analysis (DCA) Verification:** Net Benefit was independently verified across threshold probabilities $p_t \in [0.05, 0.30]$. At $p_t = 0.05$, QNN, MLP, and Treat-All tie exactly at $0.0697$. QNN achieves net benefit superiority over Treat-All and Treat-None for $p_t \in [0.09, 0.25]$, with maximal net benefit of $0.0551$ at $p_t = 0.09$. DCA is formally classified as an exploratory secondary analysis.

---

## 2. CALIBRATION COEFFICIENT RECONCILIATION

The apparent paradox identified in Phase 13.3.4 narrative notes involved three values:
- Validation Platt slope point estimate: $a_{\text{val}} = 0.4415$ (narratively conflated with test slope $1.7096$ in one section).
- Validation-resampling slope 95% CI: $[-0.1176, 0.8572]$.
- Test-set Cox calibration slope: $\beta_{\text{test}} = 1.7096$.

Direct inspection of `experiments/phase13_3_4/run_phase13_3_4_audit.py` (lines 351–427 and 600–710) resolved this discrepancy definitively.

### Mathematical Formulation of the Pipeline

1. **Stage 1: Validation Platt Scaling Fit**  
   On the validation cohort ($\mathcal{D}_{\text{val}}$, $N=234$, $24$ deaths), raw validation logits $z_{\text{val}} = \text{logit}(p_{\text{raw}})$ are transformed via logistic regression:
   $$\hat{p}_{\text{val}}^{\text{platt}} = \sigma(a_{\text{val}} \cdot z_{\text{val}} + b_{\text{val}})$$
   - Point estimates fitted on the full validation set:
     $$a_{\text{val}} = 0.4415, \quad b_{\text{val}} = -0.0924$$
   - Over 1,000 validation bootstrap draws ($\mathcal{D}_{\text{val}}^{*(b)}$), the distribution of $a_{\text{val}}^{*(b)}$ yields:
     $$\text{Mean } a = 0.4155, \quad \text{SD } = 0.2513, \quad 95\%\ \text{Percentile CI} = [-0.1176, 0.8572]$$
     Because the $95\%$ CI spans zero, validation calibration parameters under small-sample clustering ($24$ events) exhibit significant sampling variance.

2. **Stage 2: Out-of-Sample Calibration Application**  
   The frozen parameters $(a_{\text{val}} = 0.4415, b_{\text{val}} = -0.0924)$ are applied out-of-sample to raw test logits $z_{\text{test}}$ without refitting:
   $$\hat{p}_{\text{test}}^{\text{platt}} = \sigma(0.4415 \cdot z_{\text{test}} - 0.0924)$$

3. **Stage 3: Test-Set Cox Calibration Assessment**  
   To evaluate whether the resulting test probabilities $\hat{p}_{\text{test}}^{\text{platt}}$ are well-calibrated, a standard Cox calibration regression is fit on the test set ($\mathcal{D}_{\text{test}}$, $N=241$, $28$ deaths):
   $$\text{logit}(P(Y_{\text{test}} = 1)) = \alpha_{\text{test}} + \beta_{\text{test}} \cdot \text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$$
   - Calibration intercept: $\alpha_{\text{test}} = 0.1081$ (ideal: $0.0$)
   - Calibration slope: $\beta_{\text{test}} = 1.7096$ (ideal: $1.0$)

### Calibration Reconciliation Table

| Metric / Parameter | Cohort / Stage | Sample Size & Events | Point Estimate | 95% Confidence Interval | Method / Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Platt Slope ($a_{\text{val}}$)** | Validation | $N=234$ ($24$ deaths) | $\mathbf{0.4415}$ | $[-0.1176, 0.8572]$ | Logistic reg. of $y_{\text{val}}$ on $z_{\text{val}}$ (1,000 bootstraps) |
| **Platt Intercept ($b_{\text{val}}$)** | Validation | $N=234$ ($24$ deaths) | $\mathbf{-0.0924}$ | $[-0.6698, 0.4568]$ | Logistic reg. of $y_{\text{val}}$ on $z_{\text{val}}$ (1,000 bootstraps) |
| **Test Cox Slope ($\beta_{\text{test}}$)** | Frozen Test | $N=241$ ($28$ deaths) | $\mathbf{1.7096}$ | $[0.9576, 2.5976]$ | Logistic reg. of $y_{\text{test}}$ on $\text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$ |
| **Test Cox Intercept ($\alpha_{\text{test}}$)** | Frozen Test | $N=241$ ($28$ deaths) | $\mathbf{0.1081}$ | $[-0.3807, 0.5843]$ | Logistic reg. of $y_{\text{test}}$ on $\text{logit}(\hat{p}_{\text{test}}^{\text{platt}})$ |
| **Uncalibrated Test Cox Slope** | Frozen Test | $N=241$ ($28$ deaths) | $\mathbf{0.7548}$ | $[0.4228, 1.1469]$ | Logistic reg. of $y_{\text{test}}$ on raw $z_{\text{test}}$ |

**Conclusion:** The three values are fully reconciled. The Phase 13.3.4 narrative typo accidentally reported $\beta_{\text{test}} = 1.7096$ in the field intended for $a_{\text{val}} = 0.4415$. The calculated values, predictions, and files are completely intact and accurate. Detailed derivation: [CALIBRATION_SLOPE_RECONCILIATION.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_5/CALIBRATION_SLOPE_RECONCILIATION.md).

---

## 3. VALIDATION RESAMPLING AUDIT

A granular verification of the validation resampling procedure was performed to audit stability, convergence, and clustering.

### Resampling Verification Summary

- **Total Bootstrap Resamples:** 1,000.
- **Resampling Unit:** Patient cluster (`uniquepid`).
- **Resamples Attempted:** 1,000.
- **Resamples Successfully Fitted:** 1,000 ($100.0\%$).
- **Resamples Discarded / Dropped:** 0 ($0.0\%$).
- **Single-Class Degeneracies:** 0.
- **Convergence Failures (`ConvergenceWarning`):** 0.
- **Cluster Integrity:** Grouping intact; mean encounters per resample = $234.08$ (range: $215 - 257$), mean deaths per resample = $23.95$ (range: $11 - 39$).
- **The "10,000 Refits" Explanation:** 10 model configurations were evaluated across 1,000 validation bootstrap draws ($10 \times 1,000 = 10,000$ logistic regression refits). All $10,000$ refits converged with zero numerical divergence.

Detailed audit report: [VALIDATION_RESAMPLING_VERIFICATION.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_5/VALIDATION_RESAMPLING_VERIFICATION.md).

---

## 4. PATIENT-CLUSTERED TEST BOOTSTRAP AUDIT

The test-set bootstrap evaluation was audited against 7 strict methodological criteria.

### Criteria Verification Matrix

| # | Audit Criterion | Verification Status | Code Evidence / Operational Realization |
| :-: | :--- | :---: | :--- |
| 1 | Resampling by patient identifier (`uniquepid`) | **VERIFIED** | `patient_ids = np.random.choice(unique_patients, size=n_patients, replace=True)` |
| 2 | Intact clustering of repeated encounters | **VERIFIED** | Encounter indices mapped via precomputed dictionary `pt_to_indices[pt]`, concatenated intact |
| 3 | Paired evaluation on identical bootstrap samples | **VERIFIED** | Both QNN and MLP evaluated within the exact same loop iteration $b$ on sample indices `boot_idx` |
| 4 | Duplicates in patient draws correctly handled | **VERIFIED** | Repeated patient draws append multiple encounter blocks via list concatenation (not set union) |
| 5 | Single-class bootstrap sample monitoring | **VERIFIED** | Condition `len(np.unique(y_boot)) < 2` monitored; occurred $0$ times across 2,000 iterations |
| 6 | Confidence interval calculation method | **VERIFIED** | Percentile bootstrap: `np.percentile(diffs, [2.5, 97.5])` |
| 7 | Empirical two-sided p-value calculation | **VERIFIED** | `p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))` |

### Paired Comparison Results (2,000 Patient-Clustered Bootstrap Iterations)

$$\mathcal{D}_{\text{test}}: N = 241\text{ encounters}, \quad 198\text{ unique patients}, \quad 28\text{ deaths } (11.62\%)$$

| Metric | DP-QNN (Point Est.) | DP-MLP (Point Est.) | Difference ($\Delta = \text{QNN} - \text{MLP}$) | 95% Bootstrap CI | Empirical Two-Sided $p$-value | Statistical Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AUROC** | $0.7651$ | $0.7157$ | $+0.0494$ | $[-0.0409, +0.1393]$ | $p = 0.290$ | Parity (No significant difference) |
| **AUPRC** | $0.2798$ | $0.2554$ | $+0.0244$ | $[-0.0772, +0.1309]$ | $p = 0.624$ | Parity (No significant difference) |
| **Raw Brier** | $0.1068$ | $0.1118$ | $-0.0050$ | $[-0.0224, +0.0125]$ | $p = 0.584$ | Parity (No significant difference) |
| **Platt Brier** | $0.0984$ | $0.0969$ | $+0.0015$ | $[-0.0049, +0.0078]$ | $p = 0.629$ | Parity (No significant difference) |
| **ECE (10 bins)** | $0.0534$ | $0.0498$ | $+0.0036$ | $[-0.0475, +0.0543]$ | $p = 0.892$ | Parity (No significant difference) |

Detailed audit report: [PATIENT_BOOTSTRAP_SOURCE_AUDIT.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_5/PATIENT_BOOTSTRAP_SOURCE_AUDIT.md).

---

## 5. COMPLETE TEST SUITE VERIFICATION

The full test suite was executed in PowerShell inside `C:\Users\ayush\Desktop\New folder`.

### Execution Record

```text
Command: python -m pytest -v
Execution Start: 2026-11-19T23:19:05.732111
Execution Finish: 2026-11-19T23:20:50.292453
Elapsed Time: 104.56 seconds
Python Version: 3.11.9
Pytest Version: 9.1.1
Exit Code: 0
Status: 70 PASSED, 0 FAILED, 0 SKIPPED (20 warnings)
Output Log: results/phase13_3_5/pytest_output.txt
```

### Module Breakdown

- `tests/test_audit_fixes.py`: 8 passed
- `tests/test_cohort_and_split.py`: 7 passed
- `tests/test_dp_accounting.py`: 12 passed
- `tests/test_formal_dp_engine.py`: 15 passed
- `tests/test_model_pipeline.py`: 11 passed
- `tests/test_patient_disjoint_split.py`: 6 passed
- `tests/test_statistical_eval.py`: 11 passed

### Resolution of Earlier Exit Code 1

In Phase 13.3.4, an execution record noted exit code 1. Audit revealed that the subshell command attempted `pytest -v` directly, but `pytest.exe` had not been appended to the Windows system user `PATH` environment variable. The system raised a command-not-found / path-resolution exception. When invoked via the Python interpreter module (`python -m pytest`), the test suite runs cleanly and all 70 tests pass without failures.

Detailed test log and analysis: [FINAL_TEST_EXECUTION_EVIDENCE.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_5/FINAL_TEST_EXECUTION_EVIDENCE.md) and [pytest_output.txt](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13_3_5/pytest_output.txt).

---

## 6. DECISION CURVE ANALYSIS (DCA) VERIFICATION

Decision Curve Analysis was audited across the clinically plausible range $p_t \in [0.05, 0.30]$.

### Net Benefit Equation
$$\text{NB}(p_t) = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \left(\frac{p_t}{1 - p_t}\right)$$

### Verification Table

| Threshold ($p_t$) | Odds Weight $\left(\frac{p_t}{1-p_t}\right)$ | Treat-None | Treat-All | Calibrated QNN (Platt) | Calibrated MLP (Platt) | Superior Strategy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.05$** | $0.052632$ | $0.000000$ | $0.069666$ | $0.069666$ | $0.069666$ | **Exact 3-way Tie** |
| **$0.08$** | $0.086957$ | $0.000000$ | $0.039329$ | $0.039329$ | $0.038968$ | **Tie (QNN = Treat-All)** |
| **$0.09$** | $0.098901$ | $0.000000$ | $0.028772$ | $0.055127$ | $0.039214$ | **Calibrated QNN** |
| **$0.10$** | $0.111111$ | $0.000000$ | $0.017981$ | $0.045643$ | $0.033656$ | **Calibrated QNN** |
| **$0.12$** | $0.136364$ | $0.000000$ | $-0.004338$ | $0.028480$ | $0.028291$ | **Calibrated QNN** |
| **$0.15$** | $0.176471$ | $0.000000$ | $-0.039785$ | $0.013913$ | $0.022211$ | **Calibrated MLP** |
| **$0.20$** | $0.250000$ | $0.000000$ | $-0.104772$ | $0.011411$ | $0.020747$ | **Calibrated MLP** |
| **$0.25$** | $0.333333$ | $0.000000$ | $-0.178423$ | $0.006916$ | $0.011065$ | **Calibrated MLP** |
| **$0.30$** | $0.428571$ | $0.000000$ | $-0.262596$ | $0.000000$ | $0.000000$ | **Tie (QNN = MLP = Treat-None)** |

### Clinical Utility Insights
- At low thresholds ($p_t = 0.05$), all patients receive treatment predictions under calibrated probabilities ($\hat{p} \ge 0.05$), yielding net benefit identical to Treat-All.
- Between $p_t = 0.09$ and $p_t = 0.12$, Calibrated QNN achieves the highest net benefit, peaking at $\Delta \text{NB} = +0.0264$ over Treat-All at $p_t = 0.09$ (equivalent to avoiding $\sim 26.7$ false positives per 100 encounters with zero missed deaths).
- Between $p_t = 0.15$ and $p_t = 0.25$, Calibrated MLP provides higher net benefit.
- At $p_t \ge 0.30$, both models predict no encounters above threshold, collapsing to Treat-None.
- **Classification:** DCA is officially maintained as an exploratory secondary analysis. Detailed report: [DCA_FINAL_VERIFICATION.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_5/DCA_FINAL_VERIFICATION.md).

---

## 7. STRICT RESEARCH CLAIMS & PRIVACY BOUNDARIES

To ensure scientific rigor in the IEEE manuscript, the research baselines and boundaries are formally locked:

| Research Boundary / Claim | Audited Status | Required Scientific Phrasing for Publication |
| :--- | :---: | :--- |
| **Quantum Advantage** | **REFUTED** | State computational feasibility and statistical parity with classical baselines. Do *not* claim quantum superiority or supremacy. |
| **Differential Privacy Scope** | **RECORD-LEVEL** | Formally describe as record-level (per-encounter) DP ($|\mathcal{D} \Delta \mathcal{D}'| = 1$). Do *not* claim patient-level DP ($198$ patients across $241$ encounters). |
| **Federated DP Scope** | **LOCAL-SIMULATION** | Formal DP accounting is mathematically verified for centralized local training models. Cross-silo federated DP is evaluated under simulated aggregation. |
| **Secure Aggregation Scope** | **SIMULATED** | SecAgg is verified algorithmically via simulated pairwise additive masking. Do *not* claim production cryptographic network protocols. |
| **Generalizability / Deployment** | **PRE-CLINICAL** | Single-center retrospective cohort (eICU-derived). Do *not* claim external multicenter validation, production deployment readiness, or improved clinical outcomes. |

---

## 8. REPRODUCIBILITY & CRYPTOGRAPHIC VERIFICATION

All Phase 13.3.5 deliverables, test outputs, and primary experiment code have been hashed and indexed in the verification manifest:
- **Manifest:** [verification_manifest.json](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13_3_5/verification_manifest.json)
- **Checksums:** [checksums.sha256](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13_3_5/checksums.sha256)
- **Test Output:** [pytest_output.txt](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13_3_5/pytest_output.txt)
- **Git Repository:** [GitHub Remote](https://github.com/srivastavaayush084/Privacy-Preserving-Personalized-Medical-Prediction-Using-Hybrid-Quantum-Classical-Federated-Learning) (all core code and Phase 13 artifacts tracked on `main`).

---

## 9. FINAL AUDIT DECISION

Based on complete source-level verification, mathematical reconciliation of calibration slopes, validation resampling audits, patient-clustered bootstrap verifications, decision curve analysis, and 100% test suite passing:

### **AUDIT DECISION: DECISION A**
**The research baseline is fully verified, mathematically consistent, statistically robust, and ready to be frozen.**

---

## 10. PROTOCOL ON PHASE 14 (MANUSCRIPT PREPARATION)

In accordance with strict operating instructions:
> **Phase 14 (Final IEEE Research Manuscript Preparation and Evidence Integration) will NOT begin automatically.**
> The research baseline is frozen. Phase 14 will commence only upon explicit user instruction and authorization.
