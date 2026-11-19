# Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch: 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane: QML](https://img.shields.io/badge/PennyLane-QML-00d2be.svg)](https://pennylane.ai/)
[![Opacus: DP--SGD](https://img.shields.io/badge/Opacus-DP--SGD-blueviolet.svg)](https://opacus.ai/)
[![Tests: 70 Passing](https://img.shields.io/badge/Tests-70%20Passing-success.svg)](tests/)

---

## 1. Overview

This repository hosts the official research codebase, clinical datasets, experiment pipelines, statistical audits, and reproducibility artifacts for:

> **"Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning"**

This project investigates the intersection of **Variational Quantum Machine Learning (QML)**, **Federated Learning (FL)**, **Formal Differential Privacy (DP-SGD)**, and **Multi-Center Clinical Prediction** for intensive care medicine.

### Primary Clinical Task
- **Task Definition:** Prediction of subsequent in-hospital mortality using clinical data available strictly at the **24-hour ICU landmark** ($1,440$ minutes post-admission).
- **Clinical Cohort:** $1,607$ ICU stays across $1,317$ unique patients from the eICU Collaborative Research Database, partitioned under a strict patient-disjoint split:
  - **Training Cohort:** $1,132$ encounters ($89$ deaths, $7.86\%$ prevalence, $921$ unique patients)
  - **Validation Cohort:** $234$ encounters ($24$ deaths, $10.26\%$ prevalence, $198$ unique patients)
  - **Test Cohort:** $241$ encounters ($28$ deaths, $11.62\%$ prevalence, $198$ unique patients)
- **Feature Space:** $41$ preprocessed clinical features spanning admission demographics, APACHE physiology scores, 24-hour vital sign trajectories, and laboratory measurements.

---

## 2. Key Methodological Innovations

```
                                  [ CLINICAL DATA SOURCE: eICU ]
                                                │
                                    (24h Landmark Filtering)
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                             LOCAL CLIENT (ICU HOSPITAL NODE)                            │
│                                                                                          │
│   Clinical Input (41-dim) ──> Classical Compression (41 -> 4)                           │
│                                      │                                                   │
│                                      ▼                                                   │
│                        Angle Encoding (4 Qubits)                                         │
│                                      │                                                   │
│                                      ▼                                                   │
│                  Parameterized Variational Quantum Circuit (Ansatz)                      │
│                           [Ry, Rz Rotations + CNOT Entanglement]                         │
│                                      │                                                   │
│                                      ▼                                                   │
│                     Pauli-Z Expectation Measurements: <Z_i>                              │
│                                      │                                                   │
│                                      ▼                                                   │
│                      Linear Readout Layer (4 -> 1 Logit)                                 │
│                                      │                                                   │
│                                      ▼                                                   │
│                       Formal Per-Sample DP-SGD Engine                                    │
│             [Per-Sample Gradient Clipping C=1.0 + Calibrated Gaussian Noise]             │
└──────────────────────────────────────────────┬───────────────────────────────────────────┘
                                               │
                                (SecAgg Masking: Shamir Secret Sharing)
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            FEDERATED AGGREGATION SERVER                                  │
│                                                                                          │
│                 Dropout-Resilient Secure Aggregation (Bonawitz et al.)                   │
│                                              │                                           │
│                                              ▼                                           │
│                       Dirichlet Non-IID Multi-Hospital Federated Rounds                  │
│                                              │                                           │
│                                              ▼                                           │
│                      Personalized Fine-Tuning & Local Post-Hoc Platt Scaling             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Formal Hybrid DP-SGD Engine (`src/privacy/formal_dp_engine.py`):**
   - Exact micro-batched per-sample autograd clipping for variational quantum circuits (`DPHybridQNNNet`).
   - Opacus v1.6.0 DP-SGD integration for classical baselines (`ClinicalDPMLPNet`).
   - Renyi Differential Privacy (RDP) accounting with exact float Poisson sampling rate normalization ($q = \mathbb{E}[B]/N$).
2. **Dropout-Resilient Secure Aggregation (`src/privacy/secure_aggregation_v2.py`):**
   - 4-round cryptographic protocol with Shamir $(t, n)$ threshold secret sharing, Diffie-Hellman pair-wise masking, and verifiable reconstruction ($\text{error} \le 1.73 \times 10^{-6}$) under up to $40\%$ client dropout.
3. **Dirichlet Non-IID Federated Learning (`experiments/federated/`):**
   - Simulates multi-center heterogeneity across hospital partitions with parameter tracking: quantum communication payload is $72.2\%$ smaller ($0.80$ MB vs $2.88$ MB) than classical models.
4. **Validation-Fit Platt Scaling & Decision-Analytic Triage:**
   - Out-of-sample Platt calibration resolves probability compression, restoring clinical calibration and yielding positive Net Benefit across decision thresholds $p_t \in [0.09, 0.25]$.

---

## 3. Audited Empirical Results (Phase 13.3.4)

### 3.1 Model Calibration vs Constant Prevalence Benchmarks (Test Set: $N=241, 28\text{ Deaths}$)

| Model Architecture | $\varepsilon$ | Calibration Mode | Brier Score | Log Loss | $\text{ECE}_{10}$ | Calib Slope | $\Delta \text{Brier vs Train Base}$ | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training Prevalence Baseline** | **N/A** | **Primary Deployable ($p=0.0786$)** | **0.104095** | **0.367836** | **0.037561** | **1.0000** | **0.000000** | Reference |
| Validation Prevalence Baseline | N/A | Validation Reference ($p=0.1026$) | 0.102870 | 0.360220 | 0.013618 | 1.0000 | -0.001225 | Reference |
| Oracle Test Prevalence Reference | N/A | Theoretical Ceiling ($p=0.1162$) | 0.102684 | 0.359249 | 0.000000 | 1.0000 | -0.001411 | Reference |
| Classical DP-MLP | $\infty$ | Raw Uncalibrated | 0.162129 | 0.518477 | 0.123177 | 0.3377 | +0.058034 | Degraded |
| **Classical DP-MLP** | $\infty$ | **Validation-Calibrated (Platt)** | **0.096438** | **0.335121** | **0.016929** | **1.0494** | **-0.007657** | **Optimal** |
| Hybrid DP-QNN | $\infty$ | Raw Uncalibrated | 0.110555 | 0.502657 | 0.102692 | 0.7526 | +0.006460 | Degraded |
| **Hybrid DP-QNN** | $\infty$ | **Validation-Calibrated (Platt)** | **0.097960** | **0.339444** | **0.044539** | **1.7096** | **-0.006135** | **Optimal** |

### 3.2 Paired Quantum vs Classical Statistical Tests ($B=2,000$ Clustered Bootstrap Replicates)

- **$\Delta$ AUROC ($\text{QNN} - \text{MLP}$):** $+0.0494$, $95\%$ CI: $[-0.0409, 0.1393]$, **$p = 0.290$** (Not Statistically Significant)
- **$\Delta$ AUPRC ($\text{QNN} - \text{MLP}$):** $+0.0244$, $95\%$ CI: $[-0.0843, 0.1256]$, **$p = 0.624$** (Not Statistically Significant)
- **$\Delta$ Brier Platt ($\text{QNN} - \text{MLP}$):** $+0.0015$, $95\%$ CI: $[-0.0042, 0.0082]$, **$p = 0.629$** (Not Statistically Significant)

> **Consensus Scientific Conclusion:** The hybrid quantum-classical model demonstrates computational feasibility and competitive predictive performance in this experimental setting. The current evaluation does not establish statistically significant superiority or quantum advantage over the classical baseline.

---

## 4. Repository Structure

```
├── configs/                     # Experiment configurations (DP, FL, architectures)
├── data/                        # Processed clinical landmark cohort (1,607 stays)
│   ├── processed/               # Legacy 2,160 cohort (preserved for historical audits)
│   └── processed_landmark/      # Primary landmark cohort (train, val, test splits)
├── dashboard/                   # 9-page interactive Streamlit clinical dashboard
├── experiments/                 # Modular reproducible experiment scripts
│   ├── landmark/                # Landmark cohort generation
│   ├── federated/               # Dirichlet non-IID federated training
│   ├── privacy/                 # DP-SGD sweeps across epsilon in {1, 2, 4, 8, inf}
│   ├── phase13_3_4/             # Final statistical audit & validation runner
│   └── quantum/                 # Quantum circuit profiling and noise analysis
├── paper/                       # LaTeX manuscript files and figures
├── research/                    # Detailed peer-review audit reports (Phases 11-13.3.4)
│   ├── phase13_3_4/             # 7 authoritative Phase 13.3.4 audit reports
│   └── ...
├── results/                     # Checkpoints, metric CSVs, plots, and manifests
│   ├── phase13_3_1/checkpoints/ # 10 frozen PyTorch checkpoints
│   └── phase13_3_4/             # Publication plots, tables, and sha256 checksums
├── scripts/                     # Utility scripts and dataset verification
├── src/                         # Production source package
│   ├── data/                    # Clinical preprocessing, landmarking, schemas
│   ├── federated/               # Secure aggregation and federated coordinator
│   ├── inference/               # FastAPI real-time REST endpoint
│   ├── models/                  # PyTorch classical MLP and PennyLane QNN
│   └── privacy/                 # Formal DP engines, RDP accounting, SecAgg v2
└── tests/                       # 70 automated regression tests (pytest)
```

---

## 5. Installation & Reproduction

### Prerequisites
- Python 3.11+
- Git & Git LFS

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/srivastavaayush084/Privacy-Preserving-Personalized-Medical-Prediction-Using-Hybrid-Quantum-Classical-Federated-Learning.git
cd Privacy-Preserving-Personalized-Medical-Prediction-Using-Hybrid-Quantum-Classical-Federated-Learning

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Test Suite
```bash
python -m pytest
# Expected: 70 passed
```

### Executing the Phase 13.3.4 Audit
```bash
python experiments/phase13_3_4/run_phase13_3_4_audit.py
```

### Launching the Clinical Dashboard & API
```bash
# Interactive Research Dashboard
streamlit run dashboard/app.py

# Real-Time Clinical Inference API
uvicorn src.inference.predict_api:app --host 0.0.0.0 --port 8000
```

---


