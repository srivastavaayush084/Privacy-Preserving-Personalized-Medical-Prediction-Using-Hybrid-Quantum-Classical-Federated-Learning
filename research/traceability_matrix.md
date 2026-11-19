# Research Traceability Matrix

## Project Title
**Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning**

---

## 1. Traceability Mapping: Research Contributions to Modules & Experiments

| Contribution ID | Architecture Document Section | Paper Draft Section | Primary Focus | Implementation Module | Core Experiment ID | Primary Validation Metric |
|---|---|---|---|---|---|---|
| **C1: Tabular EHR Feature Pipeline & Leakage Safeguards** | Sec 4, Stage 1-3 | Sec 3.2, 5.1, 5.2 | Patient-disjoint & time-bounded (first 24h) clinical feature extraction | `src/data/` | E1 (Data Pipeline Validation) | Zero patient overlap across splits; target leakage audit |
| **C2: Matched Classical Baselines** | Sec 5, Sec 12 | Sec 5.3, 5.4 | Establish rigorous non-quantum benchmarks (LR, RF, SVM, XGBoost, MLP) | `src/models/classical/` | E1 (Centralized), E2 (Fed Classical) | AUROC, AUPRC, Brier Score, ECE, F1-Score |
| **C3: Hybrid Quantum-Classical Architecture** | Sec 5, Sec 10 | Sec 3.3, 4, 5.4 | Classical bottleneck encoder -> Angle encoding -> PQC -> Pauli-Z measurements -> Classical prediction head | `src/models/quantum/` | E3 (Fed QML Global), E5 (Fed QML Personalized) | AUROC/AUPRC vs Classical MLP; Circuit depth, Qubit count, Gate count, Simulation runtime |
| **C4: Federated Learning Orchestration** | Sec 8, Sec 12 | Sec 3.6, 4, 5.4 | Multi-hospital decentralized coordination (FedAvg, FedProx) without centralizing raw EHR | `src/federated/` | E2 (Fed Classical), E3 (Fed QML) | Global vs Local-only AUROC; Communication rounds, Round convergence |
| **C5: Hospital-Level Personalization** | Sec 6, Sec 10 | Sec 3.4, 4, 5.6 | Shared global representation + hospital-specific adapters + local probability calibration | `src/personalization/` | E4 (Pers Classical), E5 (Pers QML) | Per-hospital AUROC gain ($\Delta AUROC_k$); Worst-client performance; Calibration curve slope |
| **C6: Differential Privacy Accounting** | Sec 7, Sec 12 | Sec 3.5, 4, 5.7 | Formal DP via gradient clipping ($C$) and Gaussian noise ($\sigma$) with RDP accounting | `src/privacy/` | E6 (Pers QML + DP) | $(\epsilon, \delta)$ privacy budget; Privacy-utility trade-off curve |
| **C7: Secure Aggregation Protocol** | Sec 7, Sec 8 | Sec 3.5, 4, 5.7 | Masked client updates preventing honest-but-curious coordinator from observing individual vectors | `src/privacy/` | E6-Ablation (SecAgg on/off) | Coordinator reconstruction success rate (0%); Communication overhead |
| **C8: Privacy Threat & Empirical Attack Evaluation** | Sec 7, Sec 10 | Sec 5.7 | Empirical red-teaming: Membership Inference Attacks (MIA) & Gradient Inversion attacks | `src/privacy/` | Attack Evaluation Suite | MIA AUC/Accuracy; Mean Squared Error of gradient reconstruction |
| **C9: Robustness to Non-IID & Client Drift** | Sec 10, Sec 12 | Sec 5.5 | Evaluate performance under label imbalance, volume skew, and missing clinical variables | `experiments/robustness/` | Robustness Experiments R1-R4 | Degradation under Dirichlet $\alpha \in \{0.1, 0.5, 1.0\}$; Missingness failure rate |
| **C10: Clinical Reliability & Explainability** | Sec 9, Sec 10 | Sec 3.7, 5.8 | Temperature/Platt scaling calibration, prediction uncertainty intervals, SHAP feature attribution | `src/evaluation/`, `src/inference/` | Reliability Evaluation Suite | Expected Calibration Error (ECE), Brier Score, SHAP attribution alignment |

---

## 2. Research Gap to Methodology Alignment

| Identified Literature Gap (from Synthesis) | Evidence from Uploaded Literature | How This Framework Systematically Solves It |
|---|---|---|
| **G1: Limited Personalized Medical Prediction** | Most prior studies (FedQCNN, Li et al.) focus on global classification, disregarding hospital-level clinical distribution shifts. | Explicit parameter partition: $\Theta_s$ (shared feature encoder and PQC) and $\Theta_k$ (local adapter and calibration). Evaluated by measuring worst-client and per-hospital gain. |
| **G2: Incomplete QML + Formal Privacy Integration** | Many QML works lack DP (or use heuristic perturbations); classical FL works use DP without QML. | Full stack integration of Opacus/Rényi DP accounting on local hybrid gradient updates alongside secure aggregation. |
| **G3: Limited Hospital Generalization / Synthetic Sites** | Prior works artificially partition single datasets without validating multi-center validity. | Real-hospital multi-site identification using eICU's 186 site IDs, with documented grouping and non-IID stress tests. |
| **G4: Unproven Quantum Advantage Claims** | Gupta et al. (2025) systematic review found no empirical proof of QML superiority under realistic noisy regimes. | Treats quantum advantage strictly as a hypothesis; matches train/test splits, tuning budgets, and features identically to classical baselines. |
| **G5: Insufficient Privacy Attack Validation** | Theoretical privacy claims without empirical adversary simulation. | Implements concrete Black-box Shadow Model MIA and White-box Gradient Inversion attack evaluations. |
| **G6: Clinical Data Heterogeneity** | Variable availability differs substantially between academic teaching hospitals and small community hospitals. | Incorporates robust imputation, missingness indicator features, and hospital adapter layers. |
| **G7: Quantum Hardware & Simulation Constraints** | Classical simulation scales exponentially with qubit count; NISQ noise degrades fidelity. | Low-dimensional classical bottleneck (4-8 features) feeding compact PQC (4-8 qubits); tracks depth, gate counts, and simulated depolarizing noise. |
| **G8: Limited Clinical Interpretability** | Black-box quantum states lack clinical auditability. | Classical prediction head post-measurement allows SHAP feature attributions and local Platt calibration. |
| **G9: Reproducibility & Provenance** | Lack of open code, random seeds, and leakage audits in published literature. | Deterministic seeds, patient-disjoint splits, automated test suite, and MLflow/structured JSON tracking. |

---

## 3. Implementation Status Ledger

- [x] Workspace file discovery and literature synthesis
- [x] Initial dataset directory and schema audit (`data_audit/`)
- [x] Traceability matrix definition (`research/traceability_matrix.md`)
- [ ] Clinical endpoint lock and data dictionary approval (Awaiting user confirmation on In-Hospital Mortality)
- [ ] Feature engineering and leakage-free patient-disjoint splitting (`src/data/`)
- [ ] Classical baselines training & evaluation (`src/models/classical/`)
- [ ] Hybrid Quantum-Classical model (`src/models/quantum/`)
- [ ] Federated learning pipeline with FedAvg/FedProx (`src/federated/`)
- [ ] Hospital-specific personalization adapters (`src/personalization/`)
- [ ] Differential privacy and secure aggregation (`src/privacy/`)
- [ ] Privacy red-teaming & attack evaluation (`src/privacy/`)
- [ ] Comprehensive experiment matrix execution (E1 - E6 + Ablations)
- [ ] Interactive Streamlit Dashboard (`dashboard/`)
- [ ] Final research reporting and paper updates

