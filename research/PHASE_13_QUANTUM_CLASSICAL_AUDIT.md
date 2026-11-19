# PHASE 13 — INDEPENDENT QUANTUM VS. CLASSICAL MODEL COMPLEXITY & COMMUNICATION AUDIT REPORT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13 Independent Reproducibility & Claim Verification  
**Auditor Roles:** Quantum Machine Learning Engineer, Systems & Networking Auditor  
**Date:** November 19, 2026  
**Artifact References:** [`results/phase13/model_complexity_comparison.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/model_complexity_comparison.csv), [`results/phase13/communication_recalculation.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/communication_recalculation.csv)  

---

## 1. Executive Summary & Complexity Audit Verdict

An independent architectural decomposition and byte-level communication profiling was executed comparing the Classical Deep Neural Network (`ClinicalDPMLPNet`) against the Hybrid Quantum-Classical Neural Network (`DPHybridQNNNet`).

### High-Level Audit Verdict: **COMMUNICATION EFFICIENCY VERIFIED; "QUANTUM ADVANTAGE" DEBUNKED**
1. **Communication Advantage Confirmed:** Hybrid Quantum Federated Learning achieves a **72.38% reduction in total transmitted network bandwidth** (0.80 MB vs 2.88 MB across 10 rounds; **$3.62\times$ smaller payload footprint**).
2. **"24 Parameters" Myth Dismantled:** The widely cited claim of "only 24 parameters" applies **exclusively to the Parameterized Quantum Circuit (PQC) variational layer**. The full hybrid architecture contains **957 trainable parameters** (772 encoder parameters + 24 PQC parameters + 161 head parameters). The quantum variational layer accounts for only **2.51%** of the hybrid model's parameter budget.
3. **Absence of Predictive "Quantum Advantage":** On tabular electronic health record features, classical tree-based ensembles (**Random Forest: AUROC 0.8411**, **XGBoost: AUROC 0.8334 / 0.8445**) systematically outperform both the Hybrid QNN (**0.7556 / 0.8346**) and the Classical MLP (**0.7381 / 0.7944**). The genuine merit of quantum machine learning in this framework is **compact variational parameterization**, not superior clinical discrimination.

---

## 2. Exhaustive Model Complexity & Parameter Decomposition

### 2.1 Layer-by-Layer Architectural Breakdown

```
========================================================================================
CLASSICAL DP-MLP ARCHITECTURE (Total: 5,537 Trainable Parameters | 26.31 KB State Dict)
========================================================================================
  [Layer 1] Linear(in_features=41, out_features=64)     --> W: 2,624 | b: 64
  [Norm 1]  GroupNorm(num_groups=2, num_channels=64)    --> W: 64    | b: 64
  [Layer 2] Linear(in_features=64, out_features=32)     --> W: 2,048 | b: 32
  [Norm 2]  GroupNorm(num_groups=2, num_channels=32)    --> W: 32    | b: 32
  [Layer 3] Linear(in_features=32, out_features=16)     --> W: 512   | b: 16
  [Norm 3]  GroupNorm(num_groups=2, num_channels=16)    --> W: 16    | b: 16
  [Output]  Linear(in_features=16, out_features=1)      --> W: 16    | b: 1
========================================================================================
HYBRID DP-QNN ARCHITECTURE (Total: 957 Trainable Parameters | 7.69 KB State Dict)
========================================================================================
  CLASSICAL ENCODER (772 parameters / 80.67% of total):
    - Linear(in_features=41, out_features=16)           --> W: 656   | b: 16
    - GroupNorm(num_groups=2, num_channels=16)          --> W: 16    | b: 16
    - Linear(in_features=16, out_features=4)            --> W: 64    | b: 4
  VARIATIONAL QUANTUM CIRCUIT (24 parameters / 2.51% of total):
    - StronglyEntanglingLayers(shape=(2, 4, 3))         --> Theta: 24
  CLASSICAL HEAD (161 parameters / 16.82% of total):
    - Linear(in_features=8, out_features=16)            --> W: 128   | b: 16
    - Linear(in_features=16, out_features=1)            --> W: 16    | b: 1
========================================================================================
```

### 2.2 Complexity Comparison Summary

| Model Representation | Classical Encoder Params | Variational PQC Params | Classical Head Params | Total Trainable Params | Quantum Parameter Share (%) | Serialized State Dict Size | In-Memory Disk Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | 0 | 0 | 0 | 5,537 | 0.00% | 26,937 bytes | **26.31 KB** |
| **Classical MLP Baseline** | 0 | 0 | 0 | 5,537 | 0.00% | 26,937 bytes | **26.31 KB** |
| **Hybrid DP-QNN (4-qubit)** | 772 | 24 | 161 | **957** | **2.51%** | 7,874 bytes | **7.69 KB** |
| **PQC Sublayer Only** | 0 | 24 | 0 | **24** | 100.00% | 96 bytes | **0.09 KB** |

*Audit Finding:* The hybrid architecture reduces total model parameters by **$5.79\times$** compared to the classical MLP. However, claiming the model has only 24 parameters is false; 24 parameters describes only the variational rotation angles within the 4-qubit Hilbert space.

---

## 3. Federated Communication Cost & Bandwidth Profiling

The communication protocol was benchmarked across a standard 5-hospital consortium executing 10 synchronous rounds. In each round, the central server broadcasts global model weights to 5 clients (5 downlinks), and each client transmits updated local weights to the server (5 uplinks), yielding **100 total network transmissions**.

### Communication Benchmark Table

| Network Metric | Classical FL Baseline (MLP) | Hybrid Quantum FL (QNN) | Compression & Bandwidth Reduction |
| :--- | :---: | :---: | :--- |
| **Participating Hospital Clients ($K$)** | 5 | 5 | Identical consortium size |
| **Federated Communication Rounds ($R$)** | 10 | 10 | Identical communication rounds |
| **Total Message Transmissions ($2 \cdot K \cdot R$)** | 100 | 100 | Identical message count |
| **Single Payload Size (Compressed Binary)** | **29.50 KB** | **8.15 KB** | **$3.62\times$ smaller payload footprint** |
| **Bandwidth Per Round (Consortium-Wide)** | **0.2881 MB** | **0.0796 MB** | **$3.62\times$ bandwidth reduction per round** |
| **Total Consortium Bandwidth (10 Rounds)** | **2.8808 MB** | **0.7956 MB** | **72.38% reduction ($3.62\times$ reduction factor)** |
| **Variational Circuit Parameter Overhead** | 0 params | 24 params | 96 bytes uncompressed |

*Audit Verification:* Hybrid QNN achieves genuine, reproducible communication savings. Transmitting 957 parameters rather than 5,537 parameters saves over 2 MB of bandwidth across 10 rounds, which scales linearly to tens of gigabytes in high-dimensional or hundred-hospital federations.

---

## 4. The Predictive Reality: Classical Trees vs. Quantum Neural Networks

### Scientific Comparison on Identical Holdout Cohort ($N_{\text{test}} = 393$)

| Rank | Model Architecture | Paradigm | Test AUROC | Test AUPRC | Brier Loss | 95% Bootstrap CI | Parameter Count |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **XGBoost (Depth 4)** | Centralized GBDT | **0.8445** | **0.3744** | 0.1204 | [0.7879, 0.8939] | Non-parametric trees |
| **2** | **Random Forest (150 trees)** | Centralized Bagging | **0.8411** | **0.3403** | 0.1216 | [0.7837, 0.8945] | Non-parametric trees |
| **3** | **Hybrid QNN (Depth 2)** | Centralized Quantum-Classical | **0.8346** | 0.2819 | 0.1412 | [0.7754, 0.8873] | 957 |
| **4** | **Hybrid DP-QNN (Non-private)**| Centralized GroupNorm QNN | 0.7556 | 0.2444 | 0.1997 | [0.6812, 0.8245] | 957 |
| **5** | **Classical MLP Baseline** | Centralized 3-Layer Neural Net | **0.7944** | 0.2474 | 0.1546 | [0.7182, 0.8639] | 5,537 |
| **6** | **Classical DP-MLP (Non-private)**| Centralized GroupNorm MLP | 0.7381 | 0.2474 | 0.1640 | [0.6651, 0.8083] | 5,537 |

### Critical Publication Guidelines
1. **Never assert predictive "Quantum Advantage":** XGBoost (0.8445) and Random Forest (0.8411) outperform the best Hybrid QNN (0.8346). The paired difference between Hybrid QNN and Classical MLP is not statistically significant ($p = 0.0840$).
2. **Legitimate Scientific Contribution:** The valid, defensible claim is:
   > *"Hybrid Quantum Neural Networks provide a parameter-efficient representation (957 vs 5,537 parameters) that reduces federated communication payload by 72.4% while maintaining competitive discrimination against classical multi-layer perceptrons."*
