# PHASE 13.3 FINAL REPORT: SOURCE-CODE VERIFICATION, DP REPRODUCIBILITY & SCIENTIFIC INTEGRITY AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3 — Final Source-Code Verification, DP Accounting Reproducibility and Experiment Consistency Audit  
**Author:** Independent Multidisciplinary Research, Audit, & Integrity Panel  
**Date:** November 2026  
**Final Decision:** **CONDITIONALLY READY FOR MANUSCRIPT DRAFTING**  

---

## 1. Executive Summary

Phase 13.3 conducted an independent, evidence-based verification of all code, training schedules, differential privacy accounting, experiment results, and statistical analyses produced in Phase 13.2.

The objective was not to perform new training experiments, but to independently verify every mathematical, cryptographic, and statistical claim against the actual source code, checkpoints, execution logs, and checksums.

### Summary of Audit Outcomes:
1. **QNN Training Schedule:** Reconciled the 180 vs. 360 iteration discrepancy. The actual executed QNN training in Phase 13.2 used **5 epochs** ($\lceil 1,132 / 32 \rceil = 36$ steps/epoch $\times 5$ epochs = 180 steps). The 360-step figure in Phase 13.1 was a theoretical 10-epoch proposal that was superseded in code.
2. **DP Accounting Recomputation:** All 13 core DP requirements were verified. Independent analytical RDP recomputations match the reported epsilon values in `privacy_utility_metrics.csv` and `dp_accounting_trace.csv` to 4 decimal places.
3. **Federated Privacy Boundaries:** The reported $(\epsilon, 10^{-5})$ DP guarantees strictly and exclusively cover the **centralized DP-SGD benchmark**. The federated runs (FedAvg, FedProx, SecAgg) were architectural simulations and did not execute multi-round local DP-SGD.
4. **Experiment Reconciliation:** The numerical differences between Phase 13.1 and Phase 13.2 non-private baselines were traced to documented architectural and scheduling differences (GroupNorm replacement, depth 1 vs. depth 2, 5 vs. 12 epochs). All 10 model checkpoints were loaded and verified to reproduce exact CSV metrics.
5. **Statistical Verification:** Patient-clustered paired bootstrap ($B = 1,000$) confirms that 95% confidence intervals cross zero across all budgets ($p > 0.05$). **No quantum advantage exists.** In federated evaluation, Node 2's AUROC of 1.0000 was identified as an artifact of having only 1 positive test case.
6. **Execution Verification:** The automated test suite was executed in its entirety, confirming **67 passed, 0 failed, 0 skipped, 13 warnings (exit code 0)**. All Phase 13.2 artifact SHA-256 checksums match 100%.

---

## 2. Verified Facts & Empirical Proofs

| Fact # | Audited Claim | Supporting Evidence / File | Code Location | Status |
|---|---|---|---|---|
| **F-1** | Clinical landmark cohort is strictly 24-hour post-ICU admission ($N = 1,607$, Train $N = 1,132$, Val $N = 234$, Test $N = 241$, $N_{\text{pos}} = 28$). | `data/processed_landmark/` | `src/data/data_loader.py` | **VERIFIED** |
| **F-2** | QNN normalization divides by fixed expected batch size $B_{\text{exp}} = 32.0$, preserving RDP post-processing. | `research/phase13_3/DP_ACCOUNTING_RECOMPUTATION.md` | `src/privacy/formal_dp_engine.py`:340–375 | **VERIFIED** |
| **F-3** | QNN empty batches produce pure Gaussian noise updates and advance accountant exactly once per attempt. | `tests/test_phase13_2_dp.py` | `src/privacy/formal_dp_engine.py`:350–385 | **VERIFIED** |
| **F-4** | Classical MLP Opacus training performs single internal scaling by `expected_batch_size = 62.0` with zero double-scaling. | `research/phase13_3/DP_ACCOUNTING_RECOMPUTATION.md` | `src/privacy/formal_dp_engine.py`:245–260 | **VERIFIED** |
| **F-5** | Perfect trace equality holds: $\text{Sampler Iterations} == \text{Optimizer Updates} == \text{Accountant Steps} == T$. | `results/phase13_2/dp_accounting_trace.csv` | `experiments/phase13_2/run_phase13_2_experiments.py`:185–225 | **VERIFIED** |
| **F-6** | Independent analytical RDP recomputations match reported epsilon values to 4 decimal places. | `research/phase13_3/DP_ACCOUNTING_RECOMPUTATION.md` | `tests/test_phase13_2_dp.py` | **VERIFIED** |
| **F-7** | All 10 model checkpoints in `results/phase13_2/checkpoints/` dynamically reproduce the exact CSV metrics. | `research/phase13_3/EXPERIMENT_RECONCILIATION.md` | Checkpoint verification script | **VERIFIED** |
| **F-8** | Paired patient-clustered bootstrap 95% CIs cross zero across all budgets ($p > 0.05$). | `results/phase13_2/paired_statistical_tests.csv` | `experiments/phase13_2/run_phase13_2_experiments.py`:50–110 | **VERIFIED** |
| **F-9** | Client test set for Node 2 contains only 1 positive case; AUROC 1.0 is a small-sample artifact. | `results/phase13_2/federated_client_identity.csv` | `experiments/phase13_2/run_phase13_2_experiments.py`:325–385 | **VERIFIED** |
| **F-10** | Complete test suite passes: 67 passed, 0 failed, 0 skipped, exit code 0. | PyTorch/pytest execution log | `tests/` (13 test modules) | **VERIFIED** |
| **F-11** | SHA-256 checksums of all Phase 13.2 artifacts match 100%. | `results/phase13_2/checksums.sha256` | Hash verification script | **VERIFIED** |

---

## 3. Discrepancies Resolved

### 3.1 QNN Schedule: 180 vs. 360 Iterations
- **Discrepancy:** Phase 13.1 report listed 10 epochs and 360 iterations; Phase 13.2 executed 180 iterations.
- **Resolution:** $N = 1,132$ with batch size 32 yields $\lceil 1,132 / 32 \rceil = 36$ steps per epoch. Phase 13.2 executed **5 epochs**, which equals $5 \times 36 = 180$ steps. The 10-epoch schedule was a preliminary proposal. The executable code and recorded artifacts are 100% consistent with the 5-epoch schedule.

### 3.2 Non-Private Delta Representation
- **Discrepancy:** In `results/phase13_2/privacy_utility_metrics.csv`, non-private baselines ($\epsilon = \infty$) reported $\delta = 10^{-5}$.
- **Resolution:** Non-private baselines inject zero noise ($\sigma = 0.0$) and offer zero privacy protection. Reporting $\delta = 10^{-5}$ is mathematically misleading. For all non-private baselines, $\delta$ must be designated as **N/A (Not Applicable)**.

### 3.3 Scope of Federated Differential Privacy
- **Discrepancy:** Risk of misinterpreting reported epsilon as applying to the 10-round federated training protocol.
- **Resolution:** Formally documented that the $(\epsilon, \delta)$ guarantee strictly applies to **centralized DP-SGD**. The federated training protocol executed standard FedAvg / FedProx with cryptographic secure aggregation simulation, and did not run multi-round local DP-SGD.

---

## 4. Test Suite Execution Proof

The full test suite was executed in an independent clean session:
- **Command:** `python -m pytest tests/`
- **Timestamp:** 2026-11-19T13:41:26Z
- **Exit Code:** `0`
- **Results:**
  - Total Collected: 67
  - Passed: **67**
  - Failed: **0**
  - Skipped: **0**
  - Warnings: **13** (7 standard sklearn penalty deprecations, 6 opacus optimal order / backward hook notifications)
  - Execution Time: **34.97 seconds**

---

## 5. Explicit Decision & Justification

### Final Decision: **CONDITIONALLY READY FOR MANUSCRIPT DRAFTING**

### Justification:
The experimental and mathematical foundation is solid:
- The clinical cohort is locked, auditable, and clinically valid.
- The DP mechanism normalization and empty-batch accounting are mathematically correct and independently verified.
- The paired bootstrap statistical tests and client demographics are fully documented.
- All 67 automated unit and regression tests pass unconditionally.

However, moving to manuscript drafting (Phase 14) is **strictly conditioned** upon adhering to the mandatory scientific disclosures below. The manuscript must reflect the audited evidence, not prior aspirational claims.

---

## 6. Binding Guidelines for IEEE Manuscript Drafting (Phase 14)

### 6.1 Permitted & Supported Claims (SAFE TO MAKE)
1. **Clinical Landmark Endpoint:** The task is *"prediction of subsequent in-hospital mortality using clinical observations available at 24 hours after ICU admission."*
2. **Record-Level Differential Privacy:** Privacy is bounded at the **single ICU encounter level** under add/remove-one-encounter adjacency ($\Delta_2 = C = 1.0$), with exact RDP accounting across $\epsilon \in \{1.0, 2.0, 4.0, 8.0\}$.
3. **Quantum-Classical Parity:** Hybrid Quantum Neural Networks demonstrate **stable convergence and predictive parity** with Classical MLPs under rigorous per-sample DP-SGD, showing that parameter-efficient quantum circuits can be trained privately.
4. **Federated Consortium Gain:** Collaborative training via FedAvg and FedProx consistently outperforms isolated local training for nodes with limited local data.
5. **Site-Specific Personalization:** Local fine-tuning can provide modest performance gains on sites with sufficient sample volume (Node 5: +0.0152 AUROC), but exhibits risk of over-adaptation on small nodes.

### 6.2 Prohibited Claims (MUST NOT BE MADE)
1. **NO Patient-Level DP:** Do not claim patient-level differential privacy. Disclose that multi-admission patients experience standard group privacy degradation.
2. **NO Quantum Advantage:** Do not claim quantum superiority or quantum advantage. The patient-clustered paired bootstrap 95% confidence intervals cross zero across all privacy budgets ($p > 0.05$).
3. **NO Federated Differential Privacy:** Do not claim that the 10-round federated training protocol achieves $(\epsilon=1.0)$-differential privacy. Clarify that formal DP-SGD was evaluated in the centralized setting.
4. **NO Equating SecAgg with DP:** Do not claim that Secure Aggregation provides differential privacy. SecAgg provides cryptographic transit confidentiality; DP bounds inferential disclosure from the released model.
5. **NO Real Multi-Hospital Claim:** Do not describe the 5 clients as 5 isolated physical hospitals. Describe them as synthetic Dirichlet non-IID consortium partitions ($\alpha = 0.5$) over eICU data.
6. **NO External Validation Claim:** Do not claim external dataset validation (e.g., MIMIC-IV). State that external cross-database validation is an area for future research.
7. **NO "Perfect Prediction" on Node 2:** Do not report Node 2's AUROC of 1.0000 without explicitly disclosing that its test set contained only a single positive mortality case.

---

## 7. Next Steps

With Phase 13.3 complete and all deliverables archived, the project has reached empirical and mathematical closure. The research team should review this final audit report and confirm alignment with these binding guidelines before initiating Phase 14 (Manuscript Preparation).
