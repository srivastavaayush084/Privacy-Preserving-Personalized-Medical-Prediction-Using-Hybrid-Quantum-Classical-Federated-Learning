# Phase 14 — Manuscript Final Scientific and Ethical Audit

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Manuscript Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Target Venue:** IEEE Conference on Artificial Intelligence in Medicine and Healthcare  
**Author:** Ayush Srivastava  
**Lead Auditor:** Senior Medical AI Researcher, Biostatistician & Differential Privacy Auditor  
**Date:** November 20, 2026  
**Final Audit Verdict:** **UNCONDITIONALLY CERTIFIED FOR SUBMISSION**  

---

## 1. Executive Summary

This final audit serves as the formal scientific sign-off for Phase 14 of the research project. The manuscript `IEEE_Research_Paper.tex` (compiled as `IEEE_Research_Paper.pdf`) integrates the complete corpus of experimental evidence, rigorous mathematical formulations, privacy guarantees, and biostatistical validations established across Phases 1 through 13.3.6.

Every scientific constraint, ethical requirement, and numerical specification has been verified against the frozen codebase and experimental logs.

---

## 2. Scientific Constraint Compliance Matrix

| # | Scientific Constraint / Boundary | Manuscript Enforcement Location | Verification Detail | Compliance Status |
| :---: | :--- | :--- | :--- | :---: |
| 1 | **No Quantum Advantage Claim** | Abstract, Section III, Section VIII.C, Section X.A | Formally declares statistical parity ($\Delta\text{AUROC}=+0.0491, p=0.290$; $\Delta\text{AUPRC}=+0.0235, p=0.624$). Explicitly states that classical GBDTs (XGBoost AUROC $0.8445$) remain the empirical standard for tabular EHR. | **CERTIFIED** |
| 2 | **Record-Level DP Only** | Abstract, Section VI.A, Section XI.2 | Explicitly defines neighborhood relation as $|\mathcal{D} \Delta \mathcal{D}'| = 1$ at the clinical encounter level. Clarifies in Section XI.2 that patient-level DP is not asserted. | **CERTIFIED** |
| 3 | **Simulated SecAgg Only** | Section VI.B, Section XI.3 | Specifies 4-round Shamir-based secret sharing evaluated in an asynchronous simulated shared-memory runtime. Disclaims production WAN/TLS deployment. | **CERTIFIED** |
| 4 | **Simulated Clients Only** | Section I, Section VII.A | Explicitly denotes $K=5$ simulated consortium nodes with Dirichlet non-IID partition ($\alpha=0.5$). Does not claim physical hospital deployment. | **CERTIFIED** |
| 5 | **Single-Cohort Retrospective** | Abstract, Section V.A, Section XI.1 | Discloses dataset origin as eICU Collaborative Research Database Demo v2.0.1. States external validation on MIMIC-IV as future work under formal DUA. | **CERTIFIED** |
| 6 | **Pre-Clinical Feasibility Only** | Abstract, Section X.B, Section XI.5, Section XII | Stated strictly as a computational and cryptographic feasibility benchmark. Disclaims clinical deployment readiness or demonstrated patient survival benefit. | **CERTIFIED** |
| 7 | **Distinguish Variational vs. Total QNN Parameters** | Abstract, Section IV.B, Section IV.C, Section VIII.D | Explicitly specifies: $24$ variational angles in 4-qubit depth-2 PQC, $760$ parameters in classical encoder, $173$ in head $\implies \mathbf{957}$ total trainable parameters. | **CERTIFIED** |
| 8 | **Total Model Parameter Counts** | Abstract, Table II, Section VIII.D | QNN: $957$ parameters; Classical MLP: $5,537$ parameters ($82.7\%$ reduction). Communication: $0.80$ MB vs $2.88$ MB ($72.2\%$ reduction). | **CERTIFIED** |
| 9 | **Non-Private Labeling as $\varepsilon = \infty$** | Table II, Section VII.A, Section VIII.A, Fig. 4 | Non-private baseline models are uniformly labeled as $\varepsilon = \infty$ across tables, figures, and text. | **CERTIFIED** |
| 10 | **Validation-Frozen Platt Scaling** | Abstract, Section IX.A, Table II, Fig. 5 | Parameters ($a_{\text{val}}=0.4415, b_{\text{val}}=-0.0924$) fitted strictly on validation split ($N=234$) and applied out-of-sample to test logits without test retraining. | **CERTIFIED** |
| 11 | **Exploratory Status of DCA** | Section IX.B, Table IV, Fig. 6, Section XI.5 | Explicitly labeled as exploratory Decision Curve Analysis. Net Benefit bounds ($p_t \in [0.09, 0.25]$) stated conservatively. | **CERTIFIED** |

---

## 3. Data Governance and Research Ethics Disclosure

1. **Database Source and Access:**
   - Multicenter critical care data utilized in this study were derived from the open-access **eICU Collaborative Research Database Demo v2.0.1**, developed by the MIT Laboratory for Computational Physiology and Philips Healthcare.
2. **De-Identification and HIPAA Compliance:**
   - All clinical records in the database have been de-identified in accordance with the Health Insurance Portability and Accountability Act (HIPAA) Safe Harbor provision (45 CFR § 164.514(b)(2)). Protected Health Information (PHI)—including direct patient identifiers, contact details, precise calendar dates, and institutional names—was removed or perturbed prior to public deposit.
3. **Institutional Review Board (IRB) Exemption:**
   - Because this investigation involves solely secondary analysis of fully anonymized, publicly available open-access research records with no re-identification risk and no intervention on human subjects, it satisfies the criteria for Institutional Review Board exemption under 45 CFR § 46.104(d)(4).
4. **Data Dissemination and Reproducibility:**
   - In compliance with the PhysioNet Data Use Agreement, only derived aggregated statistics, feature transforms, model weights, and containerized simulation scripts are disseminated. Raw EHR tables are referenced by canonical database accession.

---

## 4. Traceability Matrix: Manuscript to Phase 13.3.6 Artifacts

| Manuscript Content | LaTeX Source Location | Authoritative Codebase / Data Artifact | SHA-256 Checksum / Reference |
| :--- | :--- | :--- | :--- |
| **Cohort Counts** | `IEEE_Research_Paper.tex` lines 162–179 | `data/cohort_24h_train.csv`<br>`data/cohort_24h_val.csv`<br>`data/cohort_24h_test.csv` | Train: `1,132` stays, `921` pts<br>Val: `234` stays, `198` pts<br>Test: `241` stays, `198` pts |
| **QNN Model Architecture** | `IEEE_Research_Paper.tex` lines 92–148 | `src/models/quantum_classifier.py`<br>`src/models/pqc_layers.py` | 4-qubit Strongly Entangling PQC, 2 layers, 24 variational angles, 957 total params |
| **DP-SGD Engine** | `IEEE_Research_Paper.tex` lines 191–212 | `src/privacy/formal_dp_engine.py` | Poisson subsampling ($q=1/36$ for QNN, $q=1/18$ for MLP), micro-batched backprop |
| **Secure Aggregation** | `IEEE_Research_Paper.tex` lines 213–231 | `src/federated/secure_aggregation.py` | 4-round Shamir $(t=3, n=5)$ secret sharing, RMSE $\le 4.46 \times 10^{-7}$ |
| **Privacy-Utility Metrics** | `IEEE_Research_Paper.tex` Table II | `results/phase13_3_6/METRIC_RECOMPUTATION.csv` | Exact concordance across all 10 checkpoints ($\varepsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$) |
| **Bootstrap Analysis** | `IEEE_Research_Paper.tex` Table III | `results/phase13_3_4/bootstrap_paired_metrics.csv` | $B=2,000$ patient-clustered resamples, seed 42 |
| **Platt Calibration** | `IEEE_Research_Paper.tex` Section IX.A | `results/phase13_3_4/platt_scaling_stability.csv` | $a_{\text{val}}=0.441545, b_{\text{val}}=-0.092357$ |
| **Decision Curve Analysis** | `IEEE_Research_Paper.tex` Table IV | `results/phase13_3_4/decision_curve_analysis_corrected.csv` | Net Benefit evaluated across $p_t \in [0.01, 0.35]$ |

---

## 5. Submission Readiness Sign-Off

The manuscript `IEEE_Research_Paper.pdf` has completed all technical, mathematical, typographic, and peer-review audits. It conforms strictly to IEEE formatting guidelines (exactly 8 pages, two-column format), contains authentic references, and reports genuine experimental data without overclaiming quantum utility.

**Final Approval:** APPROVED FOR ARCHIVING, GITHUB REPOSITORY SYNCHRONIZATION, AND IEEE CONFERENCE SUBMISSION.
