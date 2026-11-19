# FINAL TEST EXECUTION EVIDENCE & REGRESSION AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.5 — Final Source-Level Verification and Audit Closure  
**Lead Auditor:** Senior Biostatistician & Medical AI Reproducibility Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_5/FINAL_TEST_EXECUTION_EVIDENCE.md`  

---

## 1. Executive Summary

This report documents the official execution of the project's regression test suite to verify that all clinical cohort definitions, Opacus DP-SGD implementations, Poisson gradient clipping mechanisms, RDP privacy accountants, Shamir secret-sharing cryptosystems, and statistical validation utilities pass without error or regression.

### Summary of Execution Results
- **Exit Code:** **`0` (Clean Success)**
- **Total Tests Collected:** **`70`**
- **Tests Passed:** **`70 (100.0%)`**
- **Tests Failed:** **`0 (0.0%)`**
- **Tests Skipped:** **`0 (0.0%)`**
- **Warnings Captured:** **`20`** (documented below)
- **Untruncated Output Log:** Saved permanently at `results/phase13_3_5/pytest_output.txt`

---

## 2. Test Environment & Execution Metadata

| Parameter | Recorded Value |
| :--- | :--- |
| **Execution Command** | `python -m pytest -v \| Tee-Object -FilePath "results\phase13_3_5\pytest_output.txt"` |
| **Execution Start Time** | `2026-11-19T23:19:05.732111` |
| **Execution Finish Time** | `2026-11-19T23:20:50.292453` |
| **Total Wall-Clock Time** | `104.56 seconds` (Pytest core duration: `99.05 seconds`) |
| **Python Version** | `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]` |
| **Python Binary Path** | `C:\Program Files\Python311\python.exe` |
| **Pytest Version** | `9.1.1` (Pluggy: `1.6.0`) |
| **Root Directory** | `C:\Users\ayush\Desktop\New folder` |
| **Installed Plugins** | `anyio-4.14.2`, `asyncio-1.4.0` |
| **Operating System** | `Windows 11 (win32)` |

---

## 3. Root Cause Analysis: Earlier Pytest Exit Code 1

During the Phase 13.3.4 execution (Step 3830), an automated command exited with code 1. An audit of the historical logs confirms the exact root cause:

1. **Defective Invocation:** The command executed was `pytest`.
2. **Terminal Output:**
   ```powershell
   pytest : The term 'pytest' is not recognized as the name of a cmdlet, function, script file, or operable program.
   ```
3. **Cause:** On Windows, `pytest.exe` was installed in `C:\Users\ayush\AppData\Roaming\Python\Python311\Scripts`, which was not present on the user's system `PATH` environment variable. PowerShell was unable to locate the executable, generating a `CommandNotFoundException` with exit code 1.
4. **Resolution:** The test suite must always be invoked via the standard Python module entrypoint:
   ```powershell
   python -m pytest
   ```
   When invoked via `python -m pytest`, Python resolves the pytest package directly from `site-packages`, executing with 100% test passage (70/70) and exit code 0.

---

## 4. Module-by-Module Test Breakdown (70/70 Passing)

| Test Module | Subsystem Tested | Tests Collected | Passed | Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/test_audit_validation.py` | Patient-disjointness, feature leakage, consortium partitions | 7 | 7 | 0 | **PASS** |
| `tests/test_classical_models.py` | Classical baselines (LR, RF, SVM, XGBoost, MLP) | 6 | 6 | 0 | **PASS** |
| `tests/test_data_pipeline.py` | Data loader, bounds sanitization, feature engineering | 5 | 5 | 0 | **PASS** |
| `tests/test_federated.py` | Hospital client, FedAvg, Trimmed-Mean, coordinator | 4 | 4 | 0 | **PASS** |
| `tests/test_inference_api.py` | FastAPI REST endpoints, risk scoring, health check | 4 | 4 | 0 | **PASS** |
| `tests/test_phase12_remediation.py` | Shamir secret sharing, SecAgg v2, DP per-sample clipping | 6 | 6 | 0 | **PASS** |
| `tests/test_phase13_1_dp_accounting.py` | Poisson subsampler, noise multiplier monotonicity, empty batches | 6 | 6 | 0 | **PASS** |
| `tests/test_phase13_1_landmark.py` | 24h landmark cohort eligibility, early deaths, shapes | 6 | 6 | 0 | **PASS** |
| `tests/test_phase13_2_dp.py` | Gradient normalization by $B_{\text{exp}}$, Opacus parity trace | 5 | 5 | 0 | **PASS** |
| `tests/test_phase13_3_1_dp_consistency.py` | Float $qN$ exactness, Poisson rate consistency | 3 | 3 | 0 | **PASS** |
| `tests/test_phase13_dp_audit.py` | Non-private delta representation, record-level adjacency | 5 | 5 | 0 | **PASS** |
| `tests/test_phase13_secure_aggregation.py` | Threshold reconstruction, dropout resilience, abort threshold | 3 | 3 | 0 | **PASS** |
| `tests/test_privacy.py` | RDP accountant, gradient engine, membership inference attacks | 5 | 5 | 0 | **PASS** |
| `tests/test_quantum_models.py` | PennyLane QNode execution, circuit profiler, hybrid QNN wrapper | 5 | 5 | 0 | **PASS** |
| **Total** | **All Project Subsystems** | **70** | **70** | **0** | **100% PASS** |

---

## 5. Audit of Runtime Warnings (20 Warnings Captured)

All 20 warnings captured during the test session were audited to ensure none indicate code defects:
1. **SHAP Colormap Deprecations (3 warnings):** `PendingDeprecationWarning` in `shap/plots/colors/_colors.py` regarding future Matplotlib colormap functions (`set_bad`, `set_over`, `set_under`). Upstream library notice.
2. **Scikit-Learn LogisticRegression Penalty (1 warning):** `FutureWarning` in `test_classical_models.py` regarding scikit-learn v1.8 deprecation of `penalty='l2'` in favor of `C` and `l1_ratio`.
3. **Opacus RDP Alpha Range Suggestions (10 warnings):** `UserWarning: Optimal order is the largest alpha...` in `opacus/accountants/analysis/rdp.py`. Advises expanding the $\alpha$ grid for tighter RDP conversion.
4. **Opacus Secure RNG experimentation notice (3 warnings):** `UserWarning: Secure RNG turned off...` in `opacus/privacy_engine.py`. Normal for research test runs.
5. **PyTorch Backward Hook Notice (3 warnings):** `UserWarning: Full backward hook is firing when gradients are computed...` in `formal_dp_engine.py:259`. Expected behavior when intercepting per-sample gradients during QNN autograd hooks.

None of the 20 warnings affect mathematical correctness, numerical stability, or test validity.

---

## 6. Official Verification Certificate

I hereby certify that the project test suite was executed in its entirety on November 19, 2026 at 23:19:05 local time. Exactly 70 out of 70 automated tests passed with exit code 0.
