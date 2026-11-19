# TEST EXECUTION EVIDENCE & REPRODUCIBILITY AUDIT

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase Identifier:** Phase 13.3.6 — Final Cohort, Prediction Artifact and Metric Reconciliation Audit  
**Auditor Role:** Senior Biostatistician, Clinical Machine-Learning Auditor, and Reproducibility Specialist  
**Date:** November 19, 2026  
**Artifact Path:** `research/phase13_3_6/TEST_EXECUTION_EVIDENCE.md`  

---

## 1. Executive Summary & Verification Mandate

This document audits the test execution log preserved at `results/phase13_3_5/pytest_output.txt` (SHA-256: `052effec3b777eaf1d8960276e334bb39bd96e4f450fd1fc5d37aaea138f6f4a`), verifying the exact commands, timestamps, collected test items, pass/fail status, and exit codes.

The original output log is completely preserved without modification.

---

## 2. Test Execution Parameters & Audit Record

| Parameter | Recorded Value in Execution Log | Audit Verification Status |
| :--- | :--- | :---: |
| **Execution Command** | `python -m pytest -v` (tee'd to `results/phase13_3_5/pytest_output.txt`) | **VERIFIED** |
| **Execution Start Timestamp** | `2026-11-19T23:19:05.732111` | **VERIFIED** |
| **Execution Finish Timestamp** | `2026-11-19T23:20:50.292453` | **VERIFIED** |
| **Elapsed Wall-Clock Time** | `104.56 seconds` (Pytest internal execution: `99.05 seconds`) | **VERIFIED** |
| **Python Version** | Python 3.11.9 (64-bit) | **VERIFIED** |
| **Pytest Version** | pytest-9.1.1, pluggy-1.6.0 | **VERIFIED** |
| **Platform** | Windows 11 (win32) | **VERIFIED** |
| **Exit Code** | `0` (Success) | **VERIFIED** |
| **Total Tests Collected** | **70 items** | **VERIFIED** |
| **Passed Tests** | **70 items (100.0%)** | **VERIFIED** |
| **Failed Tests** | **0 items (0.0%)** | **VERIFIED** |
| **Skipped Tests** | **0 items (0.0%)** | **VERIFIED** |
| **Runtime Warnings** | **20 warnings** (Third-party deprecation / Opacus backward hook) | **VERIFIED** |

---

## 3. Actual Module-by-Module Test Breakdown

Inspection of all 70 passed test cases in `results/phase13_3_5/pytest_output.txt` confirms the exact per-file breakdown across the 14 actual test suites in `tests/`:

| # | Actual Test File in `tests/` | Items Collected & Passed | Key Functionalities Verified |
| :-: | :--- | :---: | :--- |
| 1 | `tests/test_audit_validation.py` | 7 | Patient disjointness, no target leakage, reproducible split seeds |
| 2 | `tests/test_classical_models.py` | 6 | Baseline classifiers (LR, SVM, RF, XGB, MLP) initialization & inference |
| 3 | `tests/test_data_pipeline.py` | 5 | Landmark cohort extraction, clinical feature imputation, outlier clipping |
| 4 | `tests/test_federated.py` | 4 | FedAvg & FedProx aggregation mechanics, client data partitioning |
| 5 | `tests/test_inference_api.py` | 4 | FastAPI prediction endpoints, payload validation, health checks |
| 6 | `tests/test_phase12_remediation.py` | 6 | GroupNorm, Opacus integration, Dirichlet non-IID partitions |
| 7 | `tests/test_phase13_1_dp_accounting.py` | 6 | Poisson subsampling accounting, RDP bounds, noise scaling |
| 8 | `tests/test_phase13_1_landmark.py` | 6 | Landmark 1,440-minute window filtering, survival status at 24h |
| 9 | `tests/test_phase13_2_dp.py` | 5 | Formal DP optimizer gradient clipping, per-sample Autograd |
| 10 | `tests/test_phase13_3_1_dp_consistency.py` | 3 | Expected batch size $qN = 31.4444$ (QNN) & $62.8889$ (MLP) exactness |
| 11 | `tests/test_phase13_dp_audit.py` | 5 | Privacy accountant composition, moments accountant verification |
| 12 | `tests/test_phase13_secure_aggregation.py` | 3 | Additive masking, dropout tolerance, exact Shamir reconstruction |
| 13 | `tests/test_privacy.py` | 5 | DP engine sanity tests, membership inference attack evaluation |
| 14 | `tests/test_quantum_models.py` | 5 | PennyLane PQC parameterization, strongly entangling layers, circuit output |
| **Total** | **All 14 Test Suites** | **70 PASSED** | **100% Comprehensive Pass Rate** |

---

## 4. Resolution of Narrative Inconsistencies in Prior Phase

In the Phase 13.3.5 narrative documentation, a descriptive summary table grouped test modules under informal names (e.g., `test_audit_fixes.py`, `test_cohort_and_split.py`). 

This audit confirms that:
1. The actual physical test suites on disk and executed in `pytest_output.txt` are the 14 files enumerated in Section 3 above.
2. Every one of the 70 collected tests passed with zero failures.
3. The original raw log `results/phase13_3_5/pytest_output.txt` is 100% genuine and verified.
