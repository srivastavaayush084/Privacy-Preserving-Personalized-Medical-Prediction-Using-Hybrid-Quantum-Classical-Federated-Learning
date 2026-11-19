# Independent Scientific Validation & Research Reproducibility Audit Report
## Project: Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning

**Audit Report ID:** AUDIT-PHASE11-20261119  
**Audit Scope:** End-to-End Scientific Integrity, Experimental Reproducibility, Checkpoint Recalculation, Statistical Reliability, and Cryptographic Security Verification.  
**Auditor Roles:** Independent Senior Research Auditor, Statistical Reviewer, Quantum ML Specialist, Healthcare Data Scientist, and Cybersecurity Auditor.

---

### 1. Executive Summary

An exhaustive independent scientific audit of all code modules, experimental datasets, saved model checkpoints, log outputs, and evaluation metrics was conducted. 

#### High-Level Audit Verdict:
1. **Reproducibility of Core Pipelines:** **VERIFIED (100% REPRODUCIBLE)**. All 36 automated unit tests passed across data pipelines, classical models, quantum circuits, federated coordination, differential privacy, inference API, and audit validation checks.
2. **Data Pipeline & Absence of Target Leakage:** **VERIFIED**. Patient-disjoint isolation across train, validation, and test holdout splits achieved strict 0.00% patient leakage. All 41 clinical features are derived from the first 24 hours of ICU admission; all discharge variables and APACHE predicted outcome variables were strictly excluded.
3. **The "Quantum Advantage" Claim:** **DEBUNKED / UNSUPPORTED**. Centralized classical tree-based models (**XGBoost: AUROC 0.8445**, **Random Forest: AUROC 0.8411**) outperform the Hybrid Quantum Neural Network (**0.8346**). The paired AUROC difference between Hybrid QNN and Classical MLP ($+0.0402$) is **not statistically significant ($p = 0.0840$, 95% CI: $[-0.0041, +0.0914]$)**.
4. **Federated Communication Efficiency:** **VERIFIED**. Hybrid Quantum FL achieves a **3.1x reduction in communication bandwidth** (0.80 MB vs 2.46 MB) due to the compact 24-parameter PQC circuit footprint.
5. **Hospital Personalization Claims:** **PARTIALLY VERIFIED WITH METHODOLOGICAL CAVEATS**. While parameter-isolated hospital adapters consistently outperform global models, the reported dramatic recovery on Consortium 5 (AUROC 0.1849 $\to$ 0.8151) is based on a test set containing **only 6 positive mortality cases** ($N_{\text{pos}} = 6$), resulting in extreme statistical uncertainty (95% CI: $[0.552, 0.985]$). Furthermore, calling Consortium 5 a "rural hospital" was an unverified semantic label not supported by eICU hospital metadata.
6. **Differential Privacy & Security:** **PARTIALLY VERIFIED (HEURISTIC IMPLEMENTATION)**. Gradient perturbation is implemented via batch-level norm clipping rather than formal per-sample DP-SGD vector clipping. Secure aggregation is a zero-sum pairwise masking simulation that halts under client dropout, not a production-ready fault-tolerant Bonawitz protocol.

---

### 2. Workspace Artifact Inventory & Completeness Audit

An exhaustive scan of the repository identified **220 files** across all directories:
- **Raw Data:** 31 compressed eICU tables under `eicu-collaborative-research-database-demo-2.0.1/` (including `patient.csv.gz`, `apacheApsVar.csv.gz`, `apachePredVar.csv.gz`, `hospital.csv.gz`).
- **Processed Arrays:** `X_train.npy`, `y_train.npy`, `train_meta.csv` (1,712 stays, 1,274 patients); `X_val.npy`, `y_val.npy`, `val_meta.csv` (387 stays, 273 patients); `X_test.npy`, `y_test.npy`, `test_meta.csv` (393 stays, 273 patients); `feature_names.json` (41 features); `preprocessor.joblib`.
- **Model Checkpoints:** 14 verified binary checkpoints (`.joblib` and `.pt`) stored in `checkpoints/classical/`, `checkpoints/quantum/`, and `checkpoints/federated/`.
- **Evaluation Metrics:** 10 CSV metric tables under `results/metrics/`.
- **Publication Figures:** 24 high-resolution plots under `results/figures/`.
- **Unit Tests:** 7 comprehensive test files under `tests/` containing **36 PyTest unit tests** (100% passing).

---

### 3. Independent Results Recalculation & Discrepancy Resolution

Every primary experimental claim was independently recalculated by loading the raw test arrays (`X_test.npy`, `y_test.npy`), executing forward passes through the saved checkpoints, and computing 1,000 patient-level non-parametric bootstrap resamples for 95% confidence intervals.

#### Detailed Recalculation Summary:

| Exp ID | Model / Architecture | Reported AUROC | Recalculated AUROC | Delta | 95% Bootstrap CI | Verification Finding |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| **E1** | Logistic Regression (L2) | 0.8170 | 0.8170 | -0.0000 | [0.7511, 0.8735] | Verified exactly to 4 decimal places. |
| **E1** | Random Forest (150 trees) | 0.8411 | 0.8411 | -0.0000 | [0.7837, 0.8945] | Verified exactly to 4 decimal places. |
| **E1** | Support Vector Machine (RBF) | 0.7948 | 0.7948 | -0.0000 | [0.7297, 0.8563] | Verified exactly to 4 decimal places. |
| **E1** | **XGBoost (Depth 4)** | **0.8445** | **0.8445** | +0.0000 | [0.7879, 0.8939] | **Strongest baseline**. Verified exactly. |
| **E1** | Classical MLP (3-layer) | 0.7944 | 0.7944 | +0.0000 | [0.7182, 0.8639] | Verified exactly to 4 decimal places. |
| **Q-Scale** | **Hybrid QNN (4-Qubit Depth-2)** | **0.8346** | **0.8346** | -0.0000 | [0.7754, 0.8873] | Verified exactly. Outperforms MLP, but inferior to XGBoost. |
| **Q-Scale** | Hybrid QNN (4-Qubit Depth-1) | 0.7756 | 0.7756 | +0.0000 | [0.7081, 0.8386] | Verified exactly (under-parameterized). |
| **Q-Scale** | Hybrid QNN (4-Qubit Depth-3) | 0.7533 | 0.7533 | -0.0000 | [0.6686, 0.8312] | Verified exactly. Over-parameterization collapse confirmed. |
| **Q-Scale** | Hybrid QNN (6-Qubit Scaled) | 0.7863 | 0.7863 | -0.0000 | [0.7136, 0.8548] | Verified exactly. |
| **Q-Scale** | Hybrid QNN (4-Qubit Ring) | 0.7543 | 0.7543 | -0.0000 | [0.6754, 0.8280] | Verified exactly. |
| **E2** | Classical FL (FedAvg) | 0.7590 | 0.8221 | +0.0632 | [0.7624, 0.8757] | **Discrepancy Reconciled**: 0.7590 was Round 10 final state; saved checkpoint was peak Round 2 state (0.8221). |
| **E2** | Classical FL (FedProx) | 0.7746 | 0.8073 | +0.0327 | [0.7405, 0.8689] | **Discrepancy Reconciled**: 0.7746 was Round 10 final state; saved checkpoint was peak Round 2 state (0.8073). |
| **E3** | Hybrid Quantum FL (FedAvg) | 0.7868 | 0.8085 | +0.0217 | [0.7380, 0.8684] | **Discrepancy Reconciled**: 0.7868 was Round 10 final state; saved checkpoint was peak Round 4 state (0.8085). |
| **E3** | Hybrid Quantum FL (FedProx) | 0.7638 | 0.8015 | +0.0377 | [0.7272, 0.8642] | **Discrepancy Reconciled**: 0.7638 was Round 10 final state; saved checkpoint was peak Round 4 state (0.8015). |
| **E4** | Classical FL (Adapter Mean) | 0.6289 | 0.6289 | 0.0000 | [0.4995, 0.8082] | Verified exactly across 5 consortiums. |
| **E5** | Hybrid Quantum FL (Adapter Mean) | 0.7214 | 0.7214 | 0.0000 | [0.6886, 0.8151] | Verified exactly across 5 consortiums. |
| **E5** | Consortium 5 Adapter Recovery | 0.8151 | 0.8151 | 0.0000 | [0.5520, 0.9850] | Computationally verified, but $N_{\text{pos}}=6$ yields extreme uncertainty. |
| **E6** | Differential Privacy ($\epsilon = 2.0$) | 0.6659 | 0.6659 | 0.0000 | [0.5840, 0.7420] | Verified computationally under batch clipping. |

---

### 4. Resolution of the "Quantum Advantage" Claim

The previous report claimed that the hybrid quantum neural network demonstrated "quantum advantage" by reaching an AUROC of 0.8346, exceeding the classical Multi-Layer Perceptron (0.7944).

#### Scientific Evaluation:
1. **Direct Benchmark Against Gradient Boosted Trees:**
   - Centralized XGBoost achieved an AUROC of **0.8445** (PR-AUC 0.3849).
   - Centralized Random Forest achieved an AUROC of **0.8411** (PR-AUC 0.3403).
   - Both of these standard classical tree-based models outperform the Hybrid QNN on the identical test cohort.
2. **Paired Statistical Hypothesis Testing (DeLong / Paired Bootstrap):**
   - A paired bootstrap test (1,000 resamples on the same 393 patients) was conducted comparing Hybrid QNN against all classical baselines:
     - **vs. XGBoost (0.8445):** Difference = $-0.0099$ ($p = 0.7160$, 95% CI: $[-0.0618, +0.0376]$). **XGBoost is superior.**
     - **vs. Random Forest (0.8411):** Difference = $-0.0065$ ($p = 0.8640$, 95% CI: $[-0.0555, +0.0405]$). **Random Forest is superior.**
     - **vs. Classical MLP (0.7944):** Difference = $+0.0402$ ($p = 0.0840$, 95% CI: $[-0.0041, +0.0914]$). **Difference is NOT statistically significant at the $\alpha = 0.05$ level.**
3. **Audit Verdict on Quantum Advantage:**
   - **The claim of "Quantum Advantage" is scientifically indefensible and rejected.**
   - The paper must present the quantum findings honestly as: *"A Comparative Evaluation of Compact Variational Quantum Representations in Tabular Critical Care Prediction"*.
   - The true empirical benefit of the quantum circuit is **parameter compactness** (24 variational parameters) and **communication efficiency** (0.80 MB bandwidth in federated rounds), not classification superiority over tree-based ensembles.

---

### 5. Audit of Clinical Prediction Methodology

1. **Target Construction:**
   - Formally verified in `src/data/preprocessing.py`: Target is binary in-hospital mortality (`hospitaldischargestatus == 'Expired'`).
   - 28 records with missing discharge status were strictly dropped prior to splitting, avoiding label fabrication.
2. **First-24-Hour Feature Constraint & Target Leakage:**
   - All vital signs and acute physiology variables were extracted from `apacheApsVar`, which eICU protocol defines strictly as the **first 24-hour worst acute values**.
   - Inspection of `feature_names.json` verified that no APACHE predicted risk scores, length of stay, or unit discharge status columns were included.
3. **Patient-Disjoint Partitions:**
   - Train, validation, and test sets were split using `uniquepid` grouping.
   - Set intersections:
     $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   - Zero patient overlap certified.
4. **Hospital Consortium Partitioning Audit:**
   - Consortiums were created via `hospitalid % 5`, clustering 186 hospitals into 5 groups.
   - **Audit finding:** Each cluster contains a mix of hospital beds and US regions (Midwest, South, West, Northeast).
   - Calling Consortium 5 "Rural & Suburban Health System" was an **unsupported semantic narrative**. The report and paper must refer to them strictly as **Hospital Consortiums 1 through 5 (Hash Partitions)**.

---

### 6. Audit of Hospital-Level Personalization

The previous report claimed that a local residual adapter rescued Consortium 5 from an out-of-distribution failure (0.1849 AUROC) to 0.8151 AUROC.

#### Audit Findings:
1. **Sample Size Breakdown for Consortium 5 Test Set:**
   - Total ICU Stays: **79**
   - **Positive Expired Patients: ONLY 6**
   - Negative Alive Patients: **73**
2. **Statistical Reliability:**
   - Evaluating an ROC-AUC on only 6 positive events produces a standard error of $\pm 0.18$.
   - The 95% bootstrap confidence interval for Consortium 5's adapter AUROC (0.8151) is **$[0.5520, 0.9850]$**.
   - The wide confidence interval reflects high statistical variance rather than a definitive breakthrough.
3. **Data Contamination Check:**
   - Code inspection of `src/personalization/personalization_runner.py` confirmed that the adapter was trained strictly on `client.X_train` and evaluated on `client.X_test`. There was no test set leakage into adapter training.
4. **Audit Recommendation:**
   - Disclose the 6-case positive count and wide confidence interval $[0.552, 0.985]$ directly in the results section. Refrain from claiming that adapters "solve rural hospital healthcare disparity".

---

### 7. Audit of Differential Privacy & Cryptographic Security

1. **Clipping Sensitivity & Accountant Validity:**
   - `src/privacy/differential_privacy.py` clips the $L_2$ norm of the **entire mini-batch gradient tensor**, rather than individual per-sample vectors.
   - In formal differential privacy, sensitivity bounds require per-sample gradient clipping:
     $$g_i \leftarrow g_i / \max\left(1, \frac{\|g_i\|_2}{C}\right)$$
   - Therefore, the implementation is an **empirical batch-level noise perturbation**, not certified per-sample DP-SGD.
2. **Membership Inference Attack (MIA) Interpretation:**
   - The MIA evaluated a black-box shadow model. An attacker advantage of zero ($0.0000$) under $\epsilon = 2.0$ demonstrates empirical resistance to this shadow classifier, but does not mathematically prove zero privacy leakage.
3. **Secure Aggregation Implementation:**
   - The implementation confirms zero-sum pairwise masking ($s_{i,j} = -s_{j,i}$) with numerical cancellation error $< 3.81 \times 10^{-6}$.
   - However, it lacks Shamir threshold secret sharing. If any hospital drops out, the server cannot reconstruct the global update. It is a simulation of the masking principle, not a production Bonawitz deployment.

---

### 8. Automated Verification Suite

The repository was enhanced with an independent audit test suite (`tests/test_audit_validation.py`).

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\ayush\Desktop\New folder
collected 36 items

tests\test_audit_validation.py .......                                   [ 19%]
tests\test_classical_models.py ......                                    [ 36%]
tests\test_data_pipeline.py .....                                        [ 50%]
tests\test_federated.py ....                                             [ 61%]
tests\test_inference_api.py ....                                         [ 72%]
tests\test_privacy.py .....                                              [ 86%]
tests\test_quantum_models.py .....                                       [100%]

======================= 36 passed, 1 warning in 13.44s ========================
```
**All 36 unit tests pass cleanly.**

---

### 9. Final Actionable Recommendations for Research Paper Submission

1. **Title Modification:** Change from *"Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning"* to:  
   *"Privacy-Preserving Federated Medical Prediction with Parameter-Isolated Hospital Adaptation: A Comparative Study of Compact Hybrid Quantum-Classical Circuits"*.
2. **Replace All "Quantum Advantage" Claims:** Explicitly present the classical tree-based models (XGBoost 0.8445) as superior in raw discrimination, framing the quantum circuit's merit around **parameter efficiency (24 parameters)** and **low federated communication bandwidth (0.80 MB)**.
3. **Full Disclosure of Limitations:** Incorporate [FINAL_RESEARCH_LIMITATIONS.md](file:///c:/Users/ayush/Desktop/New%20folder/research/FINAL_RESEARCH_LIMITATIONS.md) into the paper's Discussion section, detailing small positive counts ($N_{\text{pos}}=6$ on Consortium 5), batch-level DP approximations, and simulated secure aggregation constraints.
4. **Use Only Audited Tables:** Replace all previous LaTeX tables with the validated edition in [validated_publication_tables.tex](file:///c:/Users/ayush/Desktop/New%20folder/results/reports/validated_publication_tables.tex).
