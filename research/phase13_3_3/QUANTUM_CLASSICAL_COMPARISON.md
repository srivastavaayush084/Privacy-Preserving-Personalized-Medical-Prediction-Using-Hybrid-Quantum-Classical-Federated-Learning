# Quantum-versus-Classical Statistical Comparison & Rigorous Advantage Audit

**Phase:** 13.3.3  
**Task:** Evidence-Based Quantum vs. Classical Benchmark and Statistical Significance Audit  
**Date:** November 19, 2026  
**Auditor:** Senior Research Auditor, Quantum Computing Specialist & Biostatistician  

---

## 1. Executive Summary

A central objective of this research project is to evaluate whether a **Hybrid Quantum-Classical Neural Network (HQNN)** provides demonstrable performance advantages over a standard **Classical Multi-Layer Perceptron (MLP)** for clinical mortality risk prediction under differential privacy.

This audit independently synthesizes the empirical evidence, patient-clustered paired bootstrap testing ($B=2,000$), and architectural constraints to establish the following conclusions:
1. **No Statistically Significant Difference:** After accounting for intra-patient clustering across 198 unique patients and 241 test encounters, the performance difference between the non-private Hybrid QNN and Classical MLP is **statistically non-significant across all primary evaluation metrics**:
   - **$\Delta$ AUROC:** $+0.0494$ ($0.7658$ vs. $0.7166$), $95\%$ Clustered CI: **$[-0.0409, 0.1393]$**, $p = \mathbf{0.290}$.
   - **$\Delta$ AUPRC:** $+0.0244$ ($0.2957$ vs. $0.2722$), $95\%$ Clustered CI: **$[-0.0843, 0.1256]$**, $p = \mathbf{0.624}$.
   - **$\Delta$ Calibrated Brier:** $+0.0015$ ($0.0980$ vs. $0.0964$), $95\%$ Clustered CI: **$[-0.0042, 0.0082]$**, $p = \mathbf{0.629}$.
   - **$\Delta$ Sensitivity (Youden):** $-0.0011$, $95\%$ Clustered CI: **$[-0.2000, 0.1905]$**, $p = \mathbf{1.000}$.
   - **$\Delta$ Specificity (Youden):** $+0.0333$, $95\%$ Clustered CI: **$[-0.0193, 0.0876]$**, $p = \mathbf{0.239}$.
   - **$\Delta$ F1 (Youden):** $+0.0174$, $95\%$ Clustered CI: **$[-0.0596, 0.0940]$**, $p = \mathbf{0.667}$.
2. **The "Brier Advantage" Was a Calibration Artifact:** The raw Brier difference ($-0.0519$, $p=0.020$) was driven entirely by uncalibrated overconfidence in the classical MLP ($p \in [0.07, 0.94]$, mean $0.234$). Once standard validation-fit Platt calibration is applied, this difference vanishes completely ($+0.0015$, $p=0.629$).
3. **No Quantum Advantage Claim is Defensible:** The experimental architecture consists of a 4-qubit Parameterized Quantum Circuit (PQC) sandwiched between classical dense layers, simulated on a classical CPU using PennyLane's `default.qubit` state-vector simulator. Neither asymptotic complexity advantage nor empirical statistical superiority exists.
4. **Valid Publication Framing:** The manuscript must describe the work as demonstrating **algorithmic feasibility and competitive performance parity** of hybrid quantum circuits in privacy-preserving clinical machine learning, explicitly refuting any claim of "quantum advantage" or "quantum supremacy."

---

## 2. Paired Statistical Evidence Matrix

Table 1 summarizes the paired differences ($\Delta = \text{QNN} - \text{MLP}$) evaluated on identical patient-clustered bootstrap samples ($B = 2,000$ replicates):

| Evaluation Dimension | Metric | Point Estimate (QNN) | Point Estimate (MLP) | Paired Difference ($\Delta$) | 95% Clustered CI | Two-Sided $p$-Value | Statistically Significant ($\alpha=0.05$)? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Discrimination** | AUROC | $0.7658$ | $0.7166$ | $+0.0494$ | **$[-0.0409, 0.1393]$** | **$0.290$** | **No** |
| **Precision-Recall** | AUPRC | $0.2957$ | $0.2722$ | $+0.0244$ | **$[-0.0843, 0.1256]$** | **$0.624$** | **No** |
| **Raw Calibration** | Brier (Raw) | $0.1106$ | $0.1621$ | $-0.0519$ | **$[-0.0933, -0.0087]$** | **$0.020$** | **Yes (MLP Overconfidence)** |
| **Calibrated Brier** | Brier (Platt) | $0.0980$ | $0.0964$ | $+0.0015$ | **$[-0.0042, 0.0082]$** | **$0.629$** | **No (Parity)** |
| **Clinical Decision** | Sensitivity ($\tau^*$) | $0.7500$ | $0.7500$ | $-0.0011$ | **$[-0.2000, 0.1905]$** | **$1.000$** | **No** |
| **Clinical Decision** | Specificity ($\tau^*$) | $0.6479$ | $0.6150$ | $+0.0333$ | **$[-0.0193, 0.0876]$** | **$0.239$** | **No** |
| **Clinical Decision** | F1 ($\tau^*$) | $0.3387$ | $0.3206$ | $+0.0174$ | **$[-0.0596, 0.0940]$** | **$0.667$** | **No** |

---

## 3. Analysis of Privacy-Utility Trade-Offs

Across differential privacy budgets ($\epsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$), both architectures experience similar privacy-induced utility degradation:

| Privacy Level ($\epsilon$) | Model Architecture | AUROC Point Estimate | AUROC 95% Clustered CI | Achieved F1 ($\tau^*$) | MIA AUC (Empirical Privacy Leakage) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **$\epsilon = 1.0$** | Classical DP-MLP | $0.5857$ | $[0.4566, 0.7176]$ | $0.2887$ | $0.5289$ |
| **$\epsilon = 1.0$** | Hybrid DP-QNN | $0.5904$ | $[0.4919, 0.6884]$ | $0.1920$ | $0.5312$ |
| **$\epsilon = 2.0$** | Classical DP-MLP | $0.5778$ | $[0.4566, 0.7061]$ | $0.2632$ | $0.5424$ |
| **$\epsilon = 2.0$** | Hybrid DP-QNN | $0.6184$ | $[0.5186, 0.7136]$ | $0.2830$ | $0.5435$ |
| **$\epsilon = 4.0$** | Classical DP-MLP | $0.6016$ | $[0.4795, 0.7247]$ | $0.3043$ | $0.5455$ |
| **$\epsilon = 4.0$** | Hybrid DP-QNN | $0.6410$ | $[0.5369, 0.7381]$ | $0.2759$ | $0.5493$ |
| **$\epsilon = 8.0$** | Classical DP-MLP | $0.6281$ | $[0.5058, 0.7483]$ | $0.3011$ | $0.5515$ |
| **$\epsilon = 8.0$** | Hybrid DP-QNN | $0.6648$ | $[0.5630, 0.7588]$ | $0.2667$ | $0.5520$ |
| **$\epsilon = \infty$** | Classical DP-MLP | $0.7166$ | $[0.6033, 0.8200]$ | $0.3206$ | $0.6145$ |
| **$\epsilon = \infty$** | Hybrid DP-QNN | $0.7658$ | $[0.6807, 0.8449]$ | $0.3387$ | $0.5507$ |

### Key Observations:
1. **Parallel Degradation Trajectories:** As $\epsilon$ is tightened from $\infty$ to $1.0$, both QNN and MLP degrade gracefully toward near-random discrimination ($\text{AUROC} \approx 0.58 - 0.59$). The rate of utility loss per unit of privacy parameter $\epsilon$ is statistically comparable.
2. **Empirical Privacy Leakage Parity:** Membership Inference Attack (MIA) risk remains tightly bounded between $0.5289$ and $0.5520$ for all private models, confirming effective defense against training-set reconstruction without significant architectural differentiation.

---

## 4. Architectural & Theoretical Limitations of the Quantum Layer

To understand why a quantum advantage cannot exist in this experimental setup, we audit the exact computational pipeline:

```
Input Features (D = 41)
       │
       ▼
Classical Encoder [Linear(41, 16) ➔ GroupNorm(2, 16) ➔ ReLU ➔ Linear(16, 4) ➔ Tanh]
       │
       ├─► Classical Bottleneck Features (4 continuous values in [-1, 1])
       │
       ▼ Angle Embedding (x_i * pi)
4-Qubit Parameterized Quantum Circuit (PQC)
  - 4 Qubits
  - 1 Strongly Entangling Layer (Rotations R_x, R_y, R_z + CNOT rings)
  - Pauli-Z Expectation Measurement: <Z_i> in [-1, 1] for i in {0, 1, 2, 3}
       │
       ▼ Concatenation [Classical Bottleneck (4) + Quantum Expectations (4) = 8 dims]
Classical Classification Head [Linear(8, 16) ➔ ReLU ➔ Dropout(0.20) ➔ Linear(16, 1)]
       │
       ▼
Binary Logit Output
```

### 4.1 Four Fatal Theoretical Barriers to Quantum Advantage:
1. **Classical Information Bottleneck:** The 41 clinical predictors are compressed by classical linear layers down to 4 continuous angles before reaching the quantum circuit. Any non-linear interactions captured by the PQC operate on already lossy classical abstractions.
2. **Trivial Hilbert Space Dimension:** A 4-qubit quantum state lives in a $2^4 = 16$-dimensional complex Hilbert space $\mathbb{C}^{16}$. A state vector of 16 double-precision complex numbers occupies only **$256$ bytes** of memory. Classical linear algebra performs exact unitary operations on $\mathbb{C}^{16}$ in fractions of a microsecond. There is no quantum complexity barrier or exponential Hilbert space advantage.
3. **Classical Hardware Simulation:** All quantum circuits in this project were simulated deterministically on a classical x86-64 Intel/AMD CPU using PennyLane's `default.qubit` state-vector backend. The simulation executed entirely on classical floating-point hardware without physical quantum coherence, superposition, or entanglement hardware.
4. **Tabular Clinical Data Topology:** Clinical ICU variables (blood pressure, creatinine, Glasgow Coma Scale, lactate) do not possess intrinsic quantum mechanical correlations, topological order, or discrete group symmetries. Classical gradient-boosted trees and deep tabular networks routinely match or exceed hybrid circuits on such datasets.

---

## 5. Formal Manuscript Guidance

The manuscript must adopt the following scientific stances:
1. **Refute "Quantum Advantage":** Never use terms such as "quantum advantage", "quantum supremacy", or "quantum-enhanced clinical efficacy" in the title, abstract, or text.
2. **Affirm "Algorithmic Feasibility":** Frame the contribution accurately:
   > *"We demonstrate that hybrid quantum-classical neural network architectures can be successfully integrated into federated learning frameworks under rigorous differential privacy guarantees, achieving discriminative performance ($AUROC = 0.7658$) comparable to classical deep tabular architectures ($AUROC = 0.7166, p = 0.290$) without introducing architectural training instabilities or excessive empirical privacy leakage."*
3. **Transparent Reporting of Uncertainty:** Always present AUROC, AUPRC, and Brier metrics alongside their patient-clustered bootstrap confidence intervals to reflect clinical sampling uncertainty.
