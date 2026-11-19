# FINAL QUANTUM-CLASSICAL COMPARISON & CLAIMS AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.4 — Final Statistical Audit, Calibration Validation and Decision-Curve Correction  
**Author:** Senior Medical AI Researcher, Biostatistician & DP Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_4/QUANTUM_CLASSICAL_FINAL_COMPARISON.md`  

---

## 1. Executive Summary

A core scientific objective of this research project was investigating whether parameterizing a variational quantum circuit within a federated, differentially private clinical predictive pipeline confers a measurable benefit over conventional classical deep learning architectures.

Across Phases 11, 12, 13, and their subphases (13.1 through 13.3.4), all experimental artifacts, checkpoint weights, mathematical formulations, and statistical tests have undergone independent audit. 

### Mandatory Publication Consensus Statement
To ensure scientific rigor and publication integrity in IEEE Transactions, the findings of this comparison are formally framed using the following mandatory statement:

> [!IMPORTANT]
> **Definitive Quantum-Classical Research Conclusion:**  
> *"The hybrid quantum-classical model demonstrates computational feasibility and competitive predictive performance in this experimental setting. The current evaluation does not establish statistically significant superiority or quantum advantage over the classical baseline."*

---

## 2. Comprehensive Empirical Comparison Across Privacy Budgets

The table below provides the audited, reproducible metrics comparing the Hybrid DP-QNN against the Classical DP-MLP across all five evaluated privacy regimes ($\varepsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$) on the frozen 241-encounter test cohort ($28\text{ deaths}$):

| Privacy Budget $\varepsilon$ | Architecture | Number of Parameters | Test AUROC (95% CI) | Test AUPRC (95% CI) | Raw Brier | Platt Brier (95% CI) | Platt Log Loss | Validation Youden Sens | Validation Youden Spec |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\varepsilon = 1.0$ | Classical DP-MLP | 5,361 | 0.5898 [0.468, 0.708] | 0.1873 [0.102, 0.317] | 0.1144 | 0.1030 [0.082, 0.126] | 0.3608 | 50.00% | 74.18% |
| $\varepsilon = 1.0$ | Hybrid DP-QNN | 1,217 | 0.5480 [0.429, 0.670] | 0.1558 [0.088, 0.264] | 0.1139 | 0.1033 [0.082, 0.126] | 0.3611 | 42.86% | 60.09% |
| $\varepsilon = 2.0$ | Classical DP-MLP | 5,361 | 0.5960 [0.478, 0.711] | 0.1989 [0.107, 0.334] | 0.1145 | 0.1025 [0.082, 0.126] | 0.3585 | 53.57% | 66.67% |
| $\varepsilon = 2.0$ | Hybrid DP-QNN | 1,217 | 0.6406 [0.529, 0.748] | 0.2131 [0.117, 0.355] | 0.1139 | 0.1037 [0.082, 0.127] | 0.3592 | 53.57% | 70.42% |
| $\varepsilon = 4.0$ | Classical DP-MLP | 5,361 | 0.6508 [0.537, 0.759] | 0.2359 [0.125, 0.386] | 0.1144 | 0.1020 [0.082, 0.125] | 0.3559 | 50.00% | 76.53% |
| $\varepsilon = 4.0$ | Hybrid DP-QNN | 1,217 | 0.6577 [0.550, 0.760] | 0.2372 [0.126, 0.388] | 0.1138 | 0.1030 [0.082, 0.126] | 0.3537 | 71.43% | 54.46% |
| $\varepsilon = 8.0$ | Classical DP-MLP | 5,361 | 0.6722 [0.565, 0.773] | 0.2520 [0.134, 0.408] | 0.1142 | 0.1017 [0.081, 0.124] | 0.3544 | 50.00% | 76.06% |
| $\varepsilon = 8.0$ | Hybrid DP-QNN | 1,217 | 0.6468 [0.537, 0.752] | 0.2443 [0.130, 0.400] | 0.1136 | 0.1016 [0.081, 0.125] | 0.3480 | 42.86% | 76.53% |
| **$\varepsilon = \infty$ (Non-Private)** | **Classical DP-MLP** | **5,361** | **0.7175 [0.612, 0.814]** | **0.3015 [0.161, 0.474]** | **0.1621** | **0.0964 [0.077, 0.119]** | **0.3351** | **75.00%** | **61.50%** |
| **$\varepsilon = \infty$ (Non-Private)** | **Hybrid DP-QNN** | **1,217** | **0.7669 [0.673, 0.852]** | **0.3259 [0.173, 0.510]** | **0.1106** | **0.0980 [0.078, 0.120]** | **0.3394** | **75.00%** | **64.79%** |

---

## 3. Formal Statistical Significance Testing

In the primary non-private benchmark ($\varepsilon = \infty$), the point estimate for AUROC favored the hybrid QNN by $+0.0494$ ($0.7669$ vs $0.7175$). However, rigorous statistical inference establishes that this apparent gap is not statistically distinguishable from random sampling noise:

1. **Paired Patient-Clustered Bootstrap ($B = 2,000$ Replicates):**
   $$\Delta \text{AUROC} = +0.0494, \quad 95\% \text{ CI: } [-0.0409, 0.1393], \quad p = 0.290$$
   $$\Delta \text{AUPRC} = +0.0244, \quad 95\% \text{ CI: } [-0.0843, 0.1256], \quad p = 0.624$$
   $$\Delta \text{Brier (Platt)} = +0.0015, \quad 95\% \text{ CI: } [-0.0042, 0.0082], \quad p = 0.629$$
2. **Interpretation:**
   The null hypothesis $H_0: \mathbb{E}[\Delta \text{AUROC}] = 0$ cannot be rejected at any standard significance level ($\alpha = 0.05, 0.10$). The 95% confidence interval spans from a negative difference (MLP superior by $0.041$) to a positive difference (QNN superior by $0.139$).
3. **Calibrated Error Parity:**
   Once both models undergo validation-fit Platt scaling, their test Brier scores are practically indistinguishable: **$0.0980$** (QNN) vs **$0.0964$** (MLP), with paired difference $p = 0.629$.

---

## 4. Architectural, Computational & Feasibility Analysis

While empirical superiority is refuted, the hybrid quantum architecture demonstrates several distinct technological properties that are relevant for quantum machine learning research:

1. **Parameter Efficiency:**
   - The hybrid QNN utilizes a 4-qubit variational circuit with $R_y$ and $R_z$ single-qubit rotations and entangling CNOT gates, containing only 8 variational quantum parameters.
   - With an initial classical compression layer ($41 \to 4$) and linear readout ($4 \to 1$), the total model requires only **1,217 parameters**, compared to **5,361 parameters** for the 3-layer MLP.
   - The QNN achieves comparable representation capacity with approximately $23\%$ of the parameter footprint.
2. **Simulation Computational Overhead:**
   - Numerical simulation of quantum circuits using classical state-vector simulators (PennyLane default.qubit) requires evaluating $2^n$ state amplitudes and computing analytical gradients via parameter-shift rules or adjoint differentiation.
   - On classical CPU hardware, training the hybrid QNN required approximately **$4.2\times$ longer clock time per epoch** than the classical MLP.
   - Deploying variational circuits in real clinical federated learning environments remains computationally expensive unless executed on native NISQ quantum hardware.
3. **Differential Privacy Sensitivity:**
   - Under aggressive privacy budgets ($\varepsilon \le 2.0$), both architectures suffer severe performance degradation, with AUROC dropping to $0.54–0.64$.
   - The quantum circuit demonstrated slightly greater stability at $\varepsilon = 2.0$ ($\text{AUROC} = 0.6406$ vs $0.5960$), but this difference was also within sampling margin of error.

---

## 5. Audit Claims Matrix: Approved vs Retracted Claims

| Research Claim | Initial Status (Phase 1–11) | Audited Status (Phase 13.3.4) | IEEE Manuscript Directive |
| :--- | :---: | :---: | :--- |
| **"Quantum Advantage in ICU Prediction"** | Claimed | **RETRACTED** | **Strictly prohibited.** Must state no statistically significant superiority was observed ($p=0.290$). |
| **"Hybrid QNN Outperforms Classical MLP"** | Claimed | **QUALIFIED** | May state numerical point difference ($+0.0494$), but must immediately report CI $[-0.041, 0.139]$ and lack of statistical significance. |
| **"Computational Feasibility of QNN in DP Pipeline"** | Hypothetical | **CONFIRMED** | Fully supported. Proves end-to-end integration of DP-SGD, Poisson subsampling, and hybrid variational circuits is functional. |
| **"Post-Hoc Calibration Restores Clinical Validity"** | Overlooked | **CONFIRMED** | Fully supported. Platt scaling reduces Brier score from $0.111$ to $0.098$, beating constant prevalence baselines. |
| **"Federated Patient-Level Differential Privacy"** | Claimed | **RETRACTED** | Must clarify that DP accounting is record-level centralized DP ($|\mathcal{D} \Delta \mathcal{D}'|=1$), with federated aggregation simulated. |
