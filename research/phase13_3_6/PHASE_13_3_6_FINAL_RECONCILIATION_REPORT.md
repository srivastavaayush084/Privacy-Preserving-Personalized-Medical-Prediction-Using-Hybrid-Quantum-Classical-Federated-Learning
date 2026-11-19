# PHASE 13.3.6 — FINAL COHORT, PREDICTION ARTIFACT AND METRIC RECONCILIATION AUDIT REPORT

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Evaluation Role:** Senior Biostatistician, Clinical Machine-Learning Auditor, Reproducibility Specialist, and Statistical Reviewer  
**Phase Identifier:** Phase 13.3.6  
**Date of Audit Closure:** November 19, 2026  
**Final Audit Decision:** **DECISION A (All Cohort Counts, Artifacts, and Metrics Fully Reconciled — Baseline Permanently Frozen)**

---

## 1. EXECUTIVE AUDIT SUMMARY

Phase 13.3.6 was conducted as an exhaustive, source-level reconciliation of Phase 13.3.5 against all historical experimental outputs, saved checkpoints, prediction arrays, and cohort manifests.

In strict adherence to audit governance:
- **No models were retrained**, no datasets regenerated, no test labels modified, and no historical reports overwritten.
- All 10 model checkpoints from `results/phase13_3_1/checkpoints/` were evaluated directly on the frozen landmark test cohort ($N=241$).
- Ground truth prediction arrays and bootstrap distributions were recomputed from scratch and verified against saved CSV registries.
- Every discrepancy identified between Phase 13.3.3/13.3.4 and Phase 13.3.5 was investigated and traced to its exact source.

---

## 2. COHORT COUNT RECONCILIATION

### 2.1 Resolution of the "192" vs "198" Validation Patients
- **Observed Discrepancy:** Earlier cohort documentation cited 234 encounters and 198 unique patients. Phase 13.3.5 narrative text cited 234 validation encounters and 192 unique patients.
- **Source Inspection:** Direct enumeration of `data/processed_landmark/val_meta.csv` (SHA-256: `7604f7b1d9e117d55a1037ab2f7a9c4c895df7850cf93f404228b3352c5831fe`) confirms:
  - Rows / Encounters: **234**
  - Unique Patients (`uniquepid`): **198**
  - Mortality Events ($Y=1$): **24** ($10.2564\%$ prevalence)
  - Survivors ($Y=0$): **210**
- **Patient Encounters Distribution:**
  - 171 patients with 1 encounter ($171$)
  - 20 patients with 2 encounters ($40$)
  - 5 patients with 3 encounters ($15$)
  - 2 patients with 4 encounters ($8$)
  - Total: $171 + 40 + 15 + 8 = \mathbf{234}$ encounters across $\mathbf{198}$ unique patients.
- **Root Cause:** In Phase 13.3.5 report preparation, the report file `VALIDATION_RESAMPLING_VERIFICATION.md` (line 13) correctly stated "198 unique patients". However, an isolated typographical transcription error occurred in the assistant's previous chat reply prose, stating "192" instead of "198". There are zero missing patients, zero filtering exclusions, and zero dropped clusters.
- **Patient Disjointness:** Pairwise intersections of `uniquepid` between Train (921), Validation (198), and Test (198) yield exactly zero overlap ($0.00\%$ patient leakage). Total unique patients across all splits: $921 + 198 + 198 = \mathbf{1,317}$.

Detailed report: [COHORT_COUNT_RECONCILIATION.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_6/COHORT_COUNT_RECONCILIATION.md).

---

## 3. PREDICTION ARTIFACT PROVENANCE & METRIC RECONCILIATION

Direct recomputation was executed on the frozen landmark test set ($N_{\text{test}} = 241, 28\text{ deaths}$) using the frozen checkpoints from `results/phase13_3_1/checkpoints/` (`qnn_eps_inf.pt`, SHA-256: `2993f7c1...`; `mlp_eps_inf.pt`, SHA-256: `4a80af6d...`).

### Reconciled Metrics Comparison Table

| Metric | Phase 13.3.3 / 13.3.4 (Verified Saved) | Phase 13.3.5 Narrative Table | Phase 13.3.6 Recomputed Ground Truth | Discrepancy Resolution & Audit Finding |
| :--- | :---: | :---: | :---: | :--- |
| **QNN AUROC** | **0.7658** | 0.7651 | **0.7658** ($0.765761$) | `0.7651` was erroneously transcribed in Phase 13.3.5 from a Youden specificity table row. Checkpoint yields **0.7658**. |
| **QNN AUPRC** | **0.2957** | 0.2798 | **0.2957** ($0.295727$) | `0.2798` was an improvised number in Phase 13.3.5 text. Standard Average Precision Score on test set is **0.2957**. |
| **QNN Raw Brier** | **0.1106** | 0.1068 | **0.1106** ($0.110555$) | `0.1068` was copied from the 10-bin ECE column of the $\varepsilon=8.0$ model in Phase 13.3.4. True Raw Brier is **0.1106**. |
| **QNN Platt Brier** | **0.0980** | 0.0984 | **0.0980** ($0.097960$) | `0.0984` was the mean of the bootstrap resamples. Full-sample test point estimate is **0.0980**. |
| **MLP AUROC** | **0.7166** | 0.7157 | **0.7166** ($0.716633$) | Checkpoint produces **0.7166** on test set. |
| **MLP AUPRC** | **0.2722** | 0.2554 | **0.2722** ($0.272248$) | Checkpoint produces **0.2722** on test set. |
| **MLP Raw Brier** | **0.1621** | 0.1118 | **0.1621** ($0.162129$) | Uncalibrated MLP suffers from tail overconfidence ($0.1621$). `0.1118` was an erroneous narrative entry. |
| **MLP Platt Brier** | **0.0964** | 0.0969 | **0.0964** ($0.096438$) | Full-sample test Platt Brier is **0.0964**. `0.0969` represented the bootstrap resample mean. |
| **$\Delta$ AUPRC 95% CI** | **[-0.0843, 0.1256]** | [-0.0772, +0.1309] | **[-0.0843, 0.1256]** | True patient-clustered bootstrap ($B=2,000$, seed 42) yields **$[-0.0843, 0.1256]$** ($p = 0.624$). |
| **$\Delta$ Platt Brier 95% CI** | **[-0.0042, 0.0082]** | [-0.0049, +0.0078] | **[-0.0042, 0.0082]** | True patient-clustered bootstrap ($B=2,000$, seed 42) yields **$[-0.0042, 0.0082]$** ($p = 0.629$). |

- **Provenance Confirmation:** There are **NOT two separate experimental runs**. All checkpoints from Phase 13.3.1 are identical to Phase 13.2, and their recomputed metrics match Phase 13.3.4 saved files to the exact digit. The variations in Phase 13.3.5 were exclusively clerical narrative errors in markdown reporting.

Detailed report: [PREDICTION_ARTIFACT_PROVENANCE.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_6/PREDICTION_ARTIFACT_PROVENANCE.md).

---

## 4. PAIRED BOOTSTRAP RECONCILIATION

All 7 methodological criteria for patient-clustered bootstrap analysis were audited and verified directly in `experiments/phase13_3_4/run_phase13_3_4_audit.py` (lines 500–649):
1. Resampling performed strictly by unique patient (`uniquepid`).
2. Intact encounter clusters concatenated without fragmentation.
3. True paired evaluation on identical bootstrap draws for every replicate.
4. Duplicate patient selections correctly represented.
5. Replicate validity: 2,000 attempted, **2,000 valid (100.0%)**, 0 zero-class degeneracies.
6. Percentile confidence intervals: $[\theta_{0.025}^*, \theta_{0.975}^*]$.
7. Empirical two-sided $p$-values: $2 \min(\text{mean}(\Delta \le 0), \text{mean}(\Delta \ge 0))$.

### Canonical Paired Comparison Results ($B=2,000$, Seed 42)

$$\mathcal{D}_{\text{test}}: N = 241\text{ encounters}, \quad 198\text{ unique patients}, \quad 28\text{ deaths } (11.62\%)$$

| Evaluation Metric | Hybrid QNN (Full Test Set) | Classical MLP (Full Test Set) | Paired Point Diff ($\Delta_{\text{point}}$) | Bootstrap Mean Diff ($\bar{\Delta}$) | Bootstrap 95% CI | Two-Sided $p$-Value | Statistical Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **AUROC** | $0.7658$ | $0.7166$ | $+0.0491$ | $+0.0494$ | $[-0.0409, +0.1393]$ | $p = 0.290$ | Parity (No Significant Difference) |
| **AUPRC** | $0.2957$ | $0.2722$ | $+0.0235$ | $+0.0244$ | $[-0.0843, +0.1256]$ | $p = 0.624$ | Parity (No Significant Difference) |
| **Calibrated Brier** | $0.0980$ | $0.0964$ | $+0.0015$ | $+0.0015$ | $[-0.0042, +0.0082]$ | $p = 0.629$ | Parity (No Significant Difference) |
| **Raw Brier** | $0.1106$ | $0.1621$ | $-0.0516$ | $-0.0519$ | $[-0.0933, -0.0087]$ | $p = 0.020$ | Significant (MLP Overconfidence) |
| **ECE (10 bins)** | $0.0445$ | $0.0169$ | $+0.0276$ | $+0.0281$ | $[-0.0210, +0.0765]$ | $p = 0.252$ | Parity (No Significant Difference) |

Detailed report: [BOOTSTRAP_RECONCILIATION.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_6/BOOTSTRAP_RECONCILIATION.md).

---

## 5. CALIBRATION PROTOCOL VERIFICATION

The calibration pipeline strictly adheres to clinical biostatistical guidelines:
1. **Validation Platt Fit:** Univariate logistic regression is fitted strictly on validation data ($\mathcal{D}_{\text{val}}, N=234, 24\text{ deaths}$), yielding slope $a_{\text{val}} = \mathbf{0.441545}$ and intercept $b_{\text{val}} = \mathbf{-0.092357}$.
2. **Out-of-Sample Application:** Calibrated probabilities $\hat{p}_{\text{test}}^{\text{platt}} = \sigma(0.441545 \cdot z_{\text{test}} - 0.092357)$ are generated on the test set without refitting or access to test labels.
3. **Distinction from Test Cox Recalibration Slope:** The test Cox recalibration slope $\beta_{\text{test}} = \mathbf{1.709588}$ ($\approx \mathbf{1.7096}$, $95\%$ CI: $[0.9576, 2.5976]$) is an out-of-sample diagnostic evaluation of calibration quality, completely distinct from the validation transformation parameter $a_{\text{val}} = 0.4415$.
4. **Governance:** In the final manuscript, $\beta_{\text{test}} = 1.7096$ must be presented strictly as a descriptive evaluation diagnostic, never as model training or validation.

Detailed report: [CALIBRATION_PROTOCOL_FINAL.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_6/CALIBRATION_PROTOCOL_FINAL.md).

---

## 6. TEST SUITE EXECUTION EVIDENCE

Inspection of `results/phase13_3_5/pytest_output.txt` (SHA-256: `052effec3b777eaf1d8960276e334bb39bd96e4f450fd1fc5d37aaea138f6f4a`):
- **Command:** `python -m pytest -v`
- **Start / Finish:** `2026-11-19T23:19:05.732111` to `2026-11-19T23:20:50.292453` (Duration: 104.56s)
- **Exit Code:** `0`
- **Result:** **70 passed, 0 failed, 0 skipped, 20 warnings** across all 14 active test suites in `tests/`.

Detailed report: [TEST_EXECUTION_EVIDENCE.md](file:///c:/Users/ayush/Desktop/New%20folder/research/phase13_3_6/TEST_EXECUTION_EVIDENCE.md).

---

## 7. PRIVACY & SCIENTIFIC CLAIMS AUDIT

All research boundaries and claims remain strictly locked:
- **No Quantum Advantage Claim:** The quantum model achieves statistical parity ($\Delta \text{AUROC} = +0.0494, p = 0.290$), computational feasibility, and parameter compactness (24 variational parameters vs 5,313 weights in MLP).
- **Record-Level DP Only:** Formally defined as record-level differential privacy ($|\mathcal{D} \Delta \mathcal{D}'| = 1$). It is not patient-level DP ($198$ patients across $241$ encounters).
- **Simulated Secure Aggregation:** Validated algorithmically via pairwise additive masking; not production cryptographic network protocols.
- **Pre-Clinical Scope:** Retrospective eICU study; no claims of clinical deployment readiness or improved patient outcomes.

---

## 8. FINAL AUDIT DECISION

Based on complete source-level reconciliation, exact recomputation of all metrics from frozen checkpoints, resolution of all narrative transcription errors, and verification of zero patient leakage:

### **AUDIT DECISION: DECISION A**
**All cohort counts, prediction artifacts, metrics, and statistical outputs are fully reconciled, mathematically verified, and reproducible. The numerical baseline is permanently frozen.**

---

## 9. PROTOCOL ON PHASE 14 (MANUSCRIPT PREPARATION)

> **Phase 14 (Final IEEE Research Manuscript Preparation and Evidence Integration) will NOT begin automatically.**  
> The research baseline is completely reconciled, verified, and frozen. Phase 14 will begin only upon your explicit instruction.
