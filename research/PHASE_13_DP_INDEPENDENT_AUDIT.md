# PHASE 13 — INDEPENDENT DIFFERENTIAL PRIVACY AUDIT REPORT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13 Independent Reproducibility & Claim Verification  
**Auditor Roles:** Senior Cryptography & Privacy Auditor, Statistical Reviewer  
**Date:** November 19, 2026  
**Implementation Modules Audited:** `src/privacy/formal_dp_engine.py`, `experiments/privacy/run_formal_dp.py`  

---

## 1. Executive Summary & Privacy Audit Verdict

An independent, mathematical and code-level audit was conducted on the Differential Privacy implementations across both the Classical Deep Neural Network (`ClinicalDPMLPNet`) and the Hybrid Quantum Neural Network (`DPHybridQNNNet`).

### Audit Verdict: **VERIFIED MATHEMATICALLY & EXPERIMENTALLY**
1. **Per-Sample Gradient Clipping:** Verified. Both architectures compute true per-sample gradients and enforce strict $L_2$ norm bounds ($\|g_i\|_2 \le C = 1.0$) prior to batch accumulation.
2. **Noise Perturbation:** Verified. Calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ is added directly to the unweighted sum of clipped sample gradients before dividing by batch size $B$, strictly implementing the canonical subsampled Gaussian mechanism (Abadi et al., 2016).
3. **Rényi DP Accounting:** Verified. Privacy budgets $(\epsilon, \delta)$ were independently recomputed using Opacus `RDPAccountant` at $\delta = 10^{-5}$. Achieved epsilons match target budgets ($\epsilon \in \{1.0, 2.0, 4.0, 8.0\}$) within a maximum discrepancy of $< 0.010$.
4. **Architectural Isolation:** Verified. Batch Normalization was completely removed and replaced with `nn.GroupNorm(num_groups=2, ...)`. GroupNorm operates strictly within each sample's channel dimensions, ensuring zero statistical leakage between samples in the same mini-batch.

---

## 2. Mathematical Pipeline Verification

### 2.1 Classical Model (`ClinicalDPMLPNet`)
- **Integration:** Utilizes the official **Opacus v1.6.0** `PrivacyEngine`.
- **Module Wrapping:** Automatically wraps layers into `GradSampleModule`, capturing per-sample gradients via PyTorch backward autograd hooks.
- **Batch Normalization Replacement:** Standard `BatchNorm1d` violates per-sample DP because mini-batch mean and variance couple samples across the batch. The architecture strictly deploys `nn.GroupNorm(num_groups=2, num_channels=h_dim)`, which satisfies per-sample independence.
- **Sampling Mechanism:** Uniform random subsampling without replacement via PyTorch `DataLoader` with batch size $B = 64$ over $N = 1,712$ training stays ($q = \frac{64}{1712} \approx 0.03738$).

### 2.2 Hybrid Quantum Model (`DPHybridQNNNet`)
- **Architectural Bottleneck:** PennyLane Parameterized Quantum Circuit (PQC) QNodes encapsulated in `qml.qnn.TorchLayer` generate dynamic autograd computation graphs that do not expose standard PyTorch backward hooks to Opacus.
- **Remediation Implementation:** Engineered an exact **micro-batched per-sample autograd engine**:
  $$\text{For each sample } i \in \{1, \dots, B\}: \quad g_i = \nabla_\theta \ell(f(x_i; \theta), y_i)$$
  $$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2 + 10^{-8}}\right)$$
  $$\tilde{g} = \frac{1}{B} \left( \sum_{i=1}^B \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) \right)$$
- **Mathematical Equivalence:** Because $\sum_{i=1}^B \bar{g}_i$ has global $L_2$ sensitivity $\Delta_2 = C = 1.0$ under the addition or removal of any single patient record, the subsampled Gaussian mechanism on $\tilde{g}$ satisfies identical differential privacy bounds as native Opacus DP-SGD.

---

## 3. Independent Privacy Accounting & Budget Recalculation

All privacy budgets were independently recomputed using the Opacus `RDPAccountant` across the exact training hyperparameters.

### Independent Verification Audit Table ($\delta = 10^{-5}$)

| Model Architecture | Implementation Type | Sample Size ($N$) | Batch Size ($B$) | Sample Rate ($q$) | Epochs | Total Steps | Target $\epsilon$ | Calibrated $\sigma$ | Verified Achieved $\epsilon$ | Discrepancy | Accounting Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | Native Opacus Engine | 1,712 | 64 | 0.03738 | 20 | 540 | 1.0 | 3.6719 | **0.9979** | -0.0021 | **VERIFIED** |
| **Classical DP-MLP** | Native Opacus Engine | 1,712 | 64 | 0.03738 | 20 | 540 | 2.0 | 2.0703 | **1.9996** | -0.0004 | **VERIFIED** |
| **Classical DP-MLP** | Native Opacus Engine | 1,712 | 64 | 0.03738 | 20 | 540 | 4.0 | 1.2817 | **3.9984** | -0.0016 | **VERIFIED** |
| **Classical DP-MLP** | Native Opacus Engine | 1,712 | 64 | 0.03738 | 20 | 540 | 8.0 | 0.8923 | **7.9906** | -0.0094 | **VERIFIED** |
| **Classical DP-MLP** | Standard Non-Private | 1,712 | 64 | 0.03738 | 20 | 540 | $\infty$ | 0.0000 | **$\infty$** | 0.0000 | **VERIFIED** |
| **Hybrid DP-QNN** | Micro-Batched Autograd | 1,712 | 32 | 0.01869 | 15 | 810 | 1.0 | 2.3438 | **0.9927** | -0.0073 | **VERIFIED** |
| **Hybrid DP-QNN** | Micro-Batched Autograd | 1,712 | 32 | 0.01869 | 15 | 810 | 2.0 | 1.4014 | **1.9901** | -0.0099 | **VERIFIED** |
| **Hybrid DP-QNN** | Micro-Batched Autograd | 1,712 | 32 | 0.01869 | 15 | 810 | 4.0 | 0.9570 | **3.9985** | -0.0015 | **VERIFIED** |
| **Hybrid DP-QNN** | Micro-Batched Autograd | 1,712 | 32 | 0.01869 | 15 | 810 | 8.0 | 0.7233 | **7.9976** | -0.0024 | **VERIFIED** |
| **Hybrid DP-QNN** | Standard Non-Private | 1,712 | 32 | 0.01869 | 15 | 810 | $\infty$ | 0.0000 | **$\infty$** | 0.0000 | **VERIFIED** |

*Audit Artifact Reference:* Full accounting records are saved in [`results/phase13/privacy_accounting_audit.csv`](file:///c:/Users/ayush/Desktop/New%20folder/results/phase13/privacy_accounting_audit.csv).

---

## 4. Critical Scientific Caveats & Required Disclosures

### 4.1 Granularity: Record-Level DP vs. Patient-Level DP
- **Mandatory Disclosure:** The privacy guarantees implemented in this framework are strictly **Record-Level Differential Privacy** (where each individual ICU stay record $x_i$ is protected).
- In the eICU training cohort ($N = 1,712$ stays), there are $1,274$ unique patients (`uniquepid`). Certain patients experienced multiple ICU admissions within the training set.
- True **Patient-Level Differential Privacy** requires grouping all stays of a given patient and clipping the aggregate sensitivity across all admissions of that patient ($\Delta_{\text{patient}} = k \cdot C$, where $k$ is the maximum number of stays per patient).
- *Publication Requirement:* All manuscripts must state explicitly that $(\epsilon, \delta)$ bounds apply per ICU admission record, not per patient lifetime.

### 4.2 Empirical MIA Advantage vs. Mathematical DP Guarantees
- In Phase 12 results, Membership Inference Attack (MIA) evaluation showed an empirical attacker AUC of $\sim 0.50$ to $0.52$ (near-zero attacker advantage).
- **Audit Clarification:** An empirical MIA advantage near zero **does NOT constitute mathematical proof of privacy**. It demonstrates empirical resistance against the specific shadow-model attack family tested.
- The true privacy guarantee is provided solely by the mathematical bounding of the Gaussian mechanism under RDP composition.

### 4.3 Delta Parameter Justification
- $\delta = 10^{-5}$ is strictly compliant with privacy theory guidelines requiring $\delta \ll \frac{1}{N}$. For $N = 1,712$, $\frac{1}{N} \approx 5.84 \times 10^{-4}$, so $\delta = 10^{-5}$ provides a safety margin $> 58\times$ below $\frac{1}{N}$.
- For the non-private baseline, $\delta = 0.0, \epsilon = \infty$ is mathematically sound because a deterministic algorithm without perturbation provides no privacy ($\epsilon = \infty$) with zero catastrophic failure probability ($\delta = 0$). In Phase 12 logging, Hybrid QNN non-private had a typographical default of $\delta = 10^{-5}$ in `formal_dp_metrics.csv`, which is reconciled here to $\delta = 0.0$.
