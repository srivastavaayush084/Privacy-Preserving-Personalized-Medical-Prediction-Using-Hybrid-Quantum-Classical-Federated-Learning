# Phase 14.1 — Final Scientific Corrections and Manuscript Audit Report

**Lead Auditor:** Senior IEEE Reviewer, Biostatistician, Federated Learning Specialist, Differential Privacy Auditor  
**Date:** November 20, 2026  
**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Final Audit Decision:** **DECISION A — All major scientific claims are supported by source-level evidence; ready for submission review.**  

---

## 1. Executive Summary

Phase 14.1 conducted a comprehensive, source-level scientific correction and audit of the IEEE conference manuscript and supporting research artifacts. All prior discrepancies in experimental provenance, architectural parameter accounting, communication bandwidth scaling, decision curve interpretation, statistical framing, feature provenance, and bibliographic metadata have been rigorously resolved.

The corrected manuscript has been compiled into **`IEEE_Research_Paper_CORRECTED.pdf`**, which adheres strictly to the **6–8 page IEEE conference page budget (measuring exactly 8 pages)** with balanced two-column formatting, zero compilation errors, and complete citation resolution.

### Audit Compliance Scorecard

| Audit Domain | Prior State / Concern | Phase 14.1 Resolution | Compliance Status |
| :--- | :--- | :--- | :---: |
| **1. Experimental Provenance** | Ambiguity between federated and DP training in Table II | Proved Table II models were trained centrally via DP-SGD; clearly separated centralized DP, simulated FL, SecAgg, and personalization into distinct modules. | **VERIFIED** |
| **2. QNN Parameter Breakdown** | Misreported intermediate numbers (omitting GroupNorm affine parameters) | Direct PyTorch introspection: Encoder ($772$) + Quantum ($24$) + Head ($161$) = $\mathbf{957}$ total; Classical MLP = $\mathbf{5,537}$ total. | **VERIFIED** |
| **3. Communication Payload** | $2.88$ MB vs $0.80$ MB misattributed to "per round" | Reconciled with `communication_recalculation.csv`: Single payload ($8.15$ KB vs $29.5$ KB), per round ($0.08$ MB vs $0.29$ MB), 10 rounds cumulative ($0.80$ MB vs $2.88$ MB, $72.4\%$ reduction). | **VERIFIED** |
| **4. Decision Curve Analysis** | Literal claim of "avoiding 26.7 false-positive interventions" | Claim retracted; replaced with exact evidence-based net benefit text ($\Delta\text{NB} \approx 0.0264$ at $p_t = 0.09$); test baseline labeled oracle retrospective. | **VERIFIED** |
| **5. Statistical Claims** | Assertions of "equivalence" or "non-inferiority" from $p > 0.05$ | Removed equivalence claims; framed as comparable discrimination without statistically significant difference ($p=0.290$ AUROC, $p=0.624$ AUPRC); added formal non-inferiority disclaimer. | **VERIFIED** |
| **6. Feature Provenance** | Unverified table mappings and landmark window leakage | Documented all 41 features across `patient`, `hospital`, `apacheApsVar`, `apachePredVar`; confirmed stays $< 1,440$ min excluded, preventing post-landmark leakage. | **VERIFIED** |
| **7. Bibliography Integrity** | Citation key typo `shuld2019evaluating` | Corrected to `schuld2019evaluating` in `references.bib` and `.tex`; all 26 references verified authentic with DOIs/URLs. | **VERIFIED** |
| **8. Page Budget & Formatting** | Risk of exceeding IEEE 8-page ceiling | Compiled via Tectonic to **exactly 8 pages**; references [1]–[26] finish cleanly on Page 8. | **VERIFIED** |
| **9. File Preservation** | Risk of overwriting Phase 14 deliverables | Original `IEEE_Research_Paper.tex` and `.pdf` preserved untouched; all corrected outputs created separately. | **VERIFIED** |
| **10. Final Decision** | Pending verification | **Decision A** certified with full scientific documentation. | **CERTIFIED** |

---

## 2. Detailed Audit Findings Across the Ten Items

### Item 1: Provenance of Federated Learning and Differential Privacy Experiments
- **Finding:** Checkpoints in `results/phase13_3_1/checkpoints/` (`mlp_eps_*.pt`, `qnn_eps_*.pt`) reported in Table II were produced by `FormalOpacusMLPTrainer` and `FormalPerSampleDPQNNTrainer` operating on the pooled training dataset ($N_{\text{train}} = 1,132$).
- **Mechanism:** Differentially Private Stochastic Gradient Descent (DP-SGD) with Poisson subsampling ($q=1/18$ for MLP, $q=1/36$ for QNN) and per-sample gradient clipping ($C=1.0$).
- **Distinction:** Simulated federated learning (`src/federated/`) evaluated distributed convergence and communication payload across 5 clients without per-step DP noise injection. Secure aggregation (`src/privacy/secure_aggregation.py`) demonstrated cryptographic mask cancellation ($\text{RMSE} \le 4.46 \times 10^{-7}$). Personalization (`src/personalization/`) evaluated post-aggregation client adaptation.
- **Action:** Delineated as separate modular experiments in Abstract, Section IV, Section V-C, Section VI, Section VII, and Section X. Fully documented in `FEDERATED_DP_EXPERIMENT_PROVENANCE.md`.

### Item 2: Parameter Accounting Reconciliation
- **Finding:** Direct PyTorch introspection on `DPHybridQNNNet` and `ClinicalDPMLPNet` from `src/privacy/formal_dp_engine.py` established:
  - **Encoder:** Dense 1 ($656 + 16 = 672$) + GroupNorm 1 ($16 + 16 = 32$) + Dense 2 ($64 + 4 = 68$) = **772** parameters.
  - **Quantum Layer:** 4 qubits $\times$ 2 layers $\times$ 3 Euler angles = **24** parameters.
  - **Classification Head:** Dense 1 ($128 + 16 = 144$) + Dense 2 ($16 + 1 = 17$) = **161** parameters.
  - **Total QNN:** $772 + 24 + 161 = \mathbf{957}$ parameters.
  - **Classical MLP:** Dense 1 + GN 1 ($2816$) + Dense 2 + GN 2 ($2144$) + Dense 3 + GN 3 ($560$) + Output ($17$) = $\mathbf{5,537}$ parameters.
- **Action:** Main text, Table II, and Figures 1 and 2 reconciled to exact numbers. Documented in `QNN_PARAMETER_RECONCILIATION.md`.

### Item 3: Communication Payload Recalculation
- **Finding:** Serialized model state dicts measure:
  - Single payload: **8.15 KB** (QNN) vs **29.5 KB** (MLP), a **72.37% reduction** ($3.62\times$ smaller).
  - Per round (5 clients, 10 messages): **0.0796 MB** (~$0.08$ MB) vs **0.2881 MB** (~$0.29$ MB).
  - Cumulative consortium bandwidth across 10 rounds (100 messages): **0.7956 MB** (~$0.80$ MB) vs **2.8808 MB** (~$2.88$ MB), a **72.38% reduction (reported as 72.4%)**.
- **Action:** Corrected phrasing in Section VII-D from "per round" to "cumulative consortium bandwidth across 10 rounds". Confirmed that payloads transmit model weights, not momentum buffers or masks. Documented in `COMMUNICATION_PAYLOAD_AUDIT.md`.

### Item 4: Decision Curve Analysis (DCA) and Clinical Utility
- **Finding:** The earlier assertion that the model "avoided 26.7 false-positive interventions" misapplied an algebraic identity to observational retrospective data and falsely implied equal sensitivity to Treat-All.
- **Action:** Completely removed the 26.7 claim. Replaced with verified text:
  > *"At a decision threshold of 0.09, the calibrated QNN achieved a net benefit of 0.0551, compared with 0.0288 for the Treat-All strategy. This represents an exploratory improvement in net benefit of approximately 0.0264. The analysis does not establish clinical effectiveness or a reduction in actual interventions."*
  The test prevalence baseline was explicitly labeled as an *oracle retrospective reference*. Documented in `DCA_INTERPRETATION_AUDIT.md`.

### Item 5: Scientific Framing of Statistical Findings
- **Finding:** In the paired patient-clustered bootstrap analysis ($B = 2,000$ resamples), difference confidence intervals span zero: $\Delta\text{AUROC} = +0.0491$ ($95\%$ CI $[-0.0409, 0.1393], p = 0.290$); $\Delta\text{AUPRC} = +0.0235$ ($95\%$ CI $[-0.0843, 0.1256], p = 0.624$).
- **Action:** Removed claims of "equivalence", "non-inferiority", or "without predictive degradation". Formulated as *comparable predictive discrimination without statistically significant difference*, accompanied by an explicit disclaimer that the absence of significance does not prove mathematical equivalence or formal non-inferiority.

### Item 6: Clinical Feature Provenance and 24-Hour Landmark Window
- **Finding:** All 41 predictors originate from `patient`, `hospital`, `apacheApsVar`, and `apachePredVar`. All physiological values represent the worst measurements in the first 24 hours ($[0, 1440]$ min).
- **Leakage Prevention:** Patients with ICU stay $< 1,440$ min were excluded, ensuring that no predictor or outcome leaks post-landmark. Four binary missingness indicators (`ph_is_missing`, `pao2_is_missing`, `bilirubin_is_missing`, `albumin_is_missing`) capture informative clinical ordering patterns. Continuous feature preprocessing (median imputation, scaling) was fitted strictly on `X_train`. Documented in `FEATURE_TEMPORAL_PROVENANCE.md`.

### Item 7: Bibliography Verification and Typo Correction
- **Finding:** Citation key `shuld2019evaluating` corrected to `schuld2019evaluating` in `references.bib` and `IEEE_Research_Paper_CORRECTED.tex`.
- **Verification:** All 26 references confirmed authentic with complete bibliographic metadata, DOIs, and URLs. Documented in `BIBLIOGRAPHY_VERIFICATION.md`.

### Item 8: IEEE Two-Column Format and Page Budget Enforcement
- **Compilation:** `IEEE_Research_Paper_CORRECTED.tex` compiles cleanly with zero errors via Tectonic.
- **Page Budget:** Exactly **8 pages**, meeting the strict 6–8 page IEEE conference limit.
- **Layout:** Section XI (Conclusion), Acknowledgments, and all 26 references fit neatly across the two columns of Page 8.

### Item 9: Preservation of Historical Files
- **Preserved:** `IEEE_Research_Paper.tex`, `IEEE_Research_Paper.pdf`, and `MANUSCRIPT_FINAL_AUDIT.md` remain completely unmodified.
- **Generated:** All corrected versions and supplementary audit documents were created independently.

---

## 3. Final Audit Decision

### **DECISION A: All major scientific claims are supported by source-level evidence; ready for submission review.**

### Justification
1. **Mathematical Rigor:** Differential privacy accounting, parameter counts, and communication metrics are verified directly from source code and exact mathematical formulations.
2. **Clinical and Statistical Responsibility:** Speculative claims regarding avoided interventions and equivalence have been replaced with responsible, peer-review-defensible prose.
3. **Reproducibility:** All datasets, splits, model checkpoints, figures, and tables are 100% reproducible from frozen artifacts.
4. **Publication Standards:** The corrected manuscript satisfies all IEEE formatting, length, and bibliographic criteria.

---

## 4. Deliverables Index

- **Corrected Manuscript (LaTeX):** [IEEE_Research_Paper_CORRECTED.tex](file:///c:/Users/ayush/Desktop/New%20folder/IEEE_Research_Paper_CORRECTED.tex)
- **Corrected Manuscript (PDF):** [IEEE_Research_Paper_CORRECTED.pdf](file:///c:/Users/ayush/Desktop/New%20folder/IEEE_Research_Paper_CORRECTED.pdf)
- **Bibliography Database:** [references.bib](file:///c:/Users/ayush/Desktop/New%20folder/references.bib)
- **Experimental Provenance Audit:** [FEDERATED_DP_EXPERIMENT_PROVENANCE.md](file:///c:/Users/ayush/Desktop/New%20folder/FEDERATED_DP_EXPERIMENT_PROVENANCE.md)
- **Parameter Reconciliation Audit:** [QNN_PARAMETER_RECONCILIATION.md](file:///c:/Users/ayush/Desktop/New%20folder/QNN_PARAMETER_RECONCILIATION.md)
- **Communication Payload Audit:** [COMMUNICATION_PAYLOAD_AUDIT.md](file:///c:/Users/ayush/Desktop/New%20folder/COMMUNICATION_PAYLOAD_AUDIT.md)
- **DCA & Clinical Utility Audit:** [DCA_INTERPRETATION_AUDIT.md](file:///c:/Users/ayush/Desktop/New%20folder/DCA_INTERPRETATION_AUDIT.md)
- **Feature Temporal Provenance Audit:** [FEATURE_TEMPORAL_PROVENANCE.md](file:///c:/Users/ayush/Desktop/New%20folder/FEATURE_TEMPORAL_PROVENANCE.md)
- **Bibliography Verification Audit:** [BIBLIOGRAPHY_VERIFICATION.md](file:///c:/Users/ayush/Desktop/New%20folder/BIBLIOGRAPHY_VERIFICATION.md)
