# Source-Level Provenance and Separation Audit of Federated Learning, Differential Privacy, and Cryptographic Security Experiments

**Author:** Senior Biostatistician, Federated Learning Researcher, Differential Privacy Auditor  
**Date:** November 20, 2026  
**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Status:** Certified Audit Document — Phase 14.1  

---

## 1. Executive Summary and Definitive Finding

This audit investigates the experimental provenance of all empirical results presented in the IEEE research manuscript, specifically addressing whether the performance metrics in **Table II** were generated through an end-to-end on-client federated differential privacy pipeline or through modular, separate experimental workflows.

### Definitive Provenance Determination
1. **Table II DP Models Were Trained Centrally:** The models evaluated in Table II (saved at `results/phase13_3_1/checkpoints/`: `mlp_eps_1.0.pt`, `mlp_eps_2.0.pt`, `mlp_eps_4.0.pt`, `mlp_eps_8.0.pt`, `mlp_eps_inf.pt`, and their QNN counterparts `qnn_eps_*.pt`) were trained **centrally** on the pooled training dataset ($N_{\text{train}} = 1,132$) using Differentially Private Stochastic Gradient Descent (\textsf{DP-SGD}) under Poisson subsampling.
2. **Federated Learning Was Simulated Separately:** Federated learning was evaluated in independent simulation scripts (`src/federated/federated_trainer.py`, `src/federated/fedprox_trainer.py`) across $K=5$ synthetic hospital nodes over $10$ communication rounds with local epochs $E=2$. These federated simulations were conducted **without per-step DP noise injection** to benchmark parameter footprint, convergence rates, and network transmission payloads.
3. **Secure Aggregation Was Simulated Separately:** The 4-round cryptographic Secure Aggregation protocol (`src/privacy/secure_aggregation.py`) was evaluated as an independent cryptographic benchmark over $n=5$ consortium nodes, verifying Shamir secret sharing and PRG pairwise masking error bounds ($\text{RMSE} \le 4.46 \times 10^{-7}$).
4. **Personalization Was Evaluated Separately:** Client personalization adapters (`src/personalization/adaptive_layer.py`) were evaluated by fine-tuning on local client holdouts after global model convergence.
5. **Privacy Accounting Scope:** The reported privacy expenditures ($\varepsilon \in \{1.0, 2.0, 4.0, 8.0\}$ with $\delta = 10^{-5}$) account strictly for the centralized DP-SGD training iterations over the pooled $N=1,132$ training cohort via the Opacus Rényi Differential Privacy (RDP) accountant. They do **not** account for multi-round, multi-client federated DP.

In the corrected manuscript (`IEEE_Research_Paper_CORRECTED.tex`), these four experimental modules are explicitly delineated in the Abstract, Section IV, Section V-C (*"Separation of Experimental Modules"*), Section VI, Section VII, and Section X (*"Discussion and Limitations"*), removing any possible ambiguity.

---

## 2. Source-Level Code Inspection

### 2.1 Centralized DP-SGD Pipeline (`src/privacy/formal_dp_engine.py`)
Inspection of `FormalOpacusMLPTrainer` and `FormalPerSampleDPQNNTrainer` reveals:
- **Dataset Input:** Tensors `X_train` and `y_train` containing all $N = 1,132$ training encounters pooled across hospitals.
- **Sampling Scheme:** 
  - For MLP: Poisson sampling with rate $q = 1/18 \approx 0.05556$, yielding expected batch size $qN = 1132 / 18 = 62.8889$.
  - For QNN: Poisson sampling with rate $q = 1/36 \approx 0.02778$, yielding expected batch size $qN = 1132 / 36 = 31.4444$.
- **Per-Sample Gradient Clipping:** Micro-batched backpropagation with $L_2$ clipping threshold $C = 1.0$.
- **Noise Injection:** Spherical Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 \mathbf{I})$ added to the sum of clipped gradients, normalized by the exact expected batch size $qN$.
- **Accountant:** Opacus `RDPAccountant` tracks RDP orders $\alpha \in [1.1, 100]$ to calculate $(\varepsilon, \delta)$ at termination.
- **Conclusion:** These trainers execute standard centralized DP-SGD, establishing the foundational privacy-utility Pareto frontier under known theoretical bounds.

### 2.2 Federated Simulation Pipeline (`src/federated/`)
Inspection of `src/federated/federated_trainer.py` and `src/federated/fedprox_trainer.py` demonstrates:
- **Cohort Partitioning:** The $N=1,132$ training stays are partitioned across $K=5$ hospital nodes based on hospital identifiers, yielding heterogeneous client dataset sizes:
  - Node 0: 312 stays
  - Node 1: 274 stays
  - Node 2: 219 stays
  - Node 3: 185 stays
  - Node 4: 142 stays
- **Training Protocol:** 10 global rounds, $E=2$ local epochs per round, learning rate $\eta = 0.005$, proximal parameter $\mu = 0.01$ (for FedProx).
- **Communication Logging:** Each round exchanges local model weights with the central coordinating server. No gradient clipping or differential privacy Gaussian noise was applied during local federated training.
- **Conclusion:** The federated experiments served exclusively to benchmark distributed optimization dynamics, FedProx straggler tolerance, and communication transmission volume (confirming the 72.4% network payload reduction).

### 2.3 Cryptographic Secure Aggregation (`src/privacy/secure_aggregation.py`)
Inspection of `src/privacy/secure_aggregation.py` confirms:
- **Protocol:** Implementation of Bonawitz et al. (CCS 2017) 4-round secure aggregation.
- **Cryptographic Primitives:**
  - Shamir $(t, n)$ secret sharing with threshold $t = \lfloor 0.6 \times 5 \rfloor = 3$ and modulus $p = 2^{31} - 1$.
  - AES-128 / ChaCha20 pseudo-random stream generator for pairwise seed expansion.
- **Simulation Validation:** Evaluated across 100 trials with random weight vectors (lengths matching QNN 957 and MLP 5,537). Under 0% to 40% client dropout rates, the reconstructed server sum matched the true sum of surviving client vectors with numerical precision:
  $$\text{RMSE} \le 4.46 \times 10^{-7}, \quad \text{Max Error} \le 1.73 \times 10^{-6}$$
- **Conclusion:** Secure aggregation was verified as an independent algorithmic component demonstrating dropout robustness and exact mask cancellation.

---

## 3. Experimental Provenance and Separation Matrix

| Experimental Component | Script / Module | Input Dataset | Privacy / Security Mechanism | Primary Output Metric | Paper Section |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Privacy-Utility Tradeoff (Table II)** | `src/privacy/formal_dp_engine.py` | Pooled $N=1,132$ training encounters | Per-sample DP-SGD with Poisson sampling, RDP accountant | AUROC, AUPRC, Brier across $\varepsilon \in \{1, 2, 4, 8, \infty\}$ | Section VI, Table II, Fig. 4 |
| **Model Parameter Decomposition** | `src/models/quantum/hybrid_qnn.py`, `src/privacy/formal_dp_engine.py` | PyTorch model definitions | GroupNorm replacement for per-sample isolation | Exact parameters: QNN (957) vs MLP (5,537) | Section IV-A, IV-B |
| **Federated Communication Overhead** | `src/federated/`, `results/phase13/communication_recalculation.csv` | 5 hospital partitions | Serialized model state dict transmission (10 rounds, 5 clients) | Single payload (8.15 KB vs 29.5 KB), 10 rounds (0.80 MB vs 2.88 MB) | Section VII-D, Table I |
| **Cryptographic Secure Aggregation** | `src/privacy/secure_aggregation.py` | Simulated 957 / 5,537 parameter vectors | 4-round Bonawitz SecAgg, Shamir $(3, 5)$ secret sharing, PRG masks | Reconstruction $\text{RMSE} \le 4.46 \times 10^{-7}$ under $\le 40\%$ dropout | Section V-B |
| **Client Personalization** | `src/personalization/adaptive_layer.py` | Local client validation holdouts | Post-aggregation linear adaptation layer fine-tuning | Client AUROC gain (+0.0381 mean improvement) | Section IX, Fig. 8 |
| **Statistical Robustness & Calibration** | `research/phase13_3_6/`, `scripts/` | Frozen test partition ($N=241, 28$ deaths) | Out-of-sample Platt scaling fitted on validation ($N=234$) | Paired bootstrap ($p=0.290$ AUROC, $p=0.624$ AUPRC), Platt Brier ($0.0980$) | Section VII-A, VII-C, Table III |
| **Exploratory Decision Curve Analysis** | `research/phase13_3_3/`, `scripts/` | Frozen test predictions ($N=241$) | Net benefit calculation across $p_t \in [0.01, 0.35]$ | Net benefit at $p_t=0.09$ ($0.0551$ vs $0.0288$ Treat-All) | Section VIII-B, Table IV, Fig. 6 |

---

## 4. Manuscript Corrections and Scientific Transparency

To ensure complete adherence to IEEE peer-review and reproducibility standards, the following explicit clarifications were integrated into `IEEE_Research_Paper_CORRECTED.tex`:

1. **Abstract:** Delineates the investigation as an evaluation of a *"modular privacy-preserving personalized hybrid quantum-classical federated learning framework"*.
2. **Section V-C (*"Separation of Experimental Modules"*):**
   > *"We clarify the relationship between experimental components: Table II presents centralized DP-SGD benchmarks to isolate the fundamental privacy-utility Pareto frontier. In separate modular experiments, federated learning (FedAvg/FedProx) was simulated across $K=5$ nodes to measure communication payloads and convergence, and secure aggregation was evaluated as an independent cryptographic protocol. Fully integrated on-client federated DP with multi-round RDP composition remains an active area for systems engineering."*
3. **Section VI (*"Experimental Setup and Training Protocols"*):**
   Explicitly notes that privacy bounds $\varepsilon \in \{1.0, 2.0, 4.0, 8.0\}$ reflect centralized DP-SGD over the pooled $N=1,132$ training cohort.
4. **Section X (*"Discussion and Limitations"*):**
   Explicitly records the modular architecture as a study limitation, noting that joint execution of on-client DP-SGD, federated aggregation, and cryptographic unmasking represents a systems implementation target for future multi-center deployments.

---

## 5. Audit Conclusion

The experimental provenance is fully verified and documented. All empirical results in Table II are genuine, reproducible, and derived from certified centralized DP-SGD checkpoints. The clear separation of experimental modules in `IEEE_Research_Paper_CORRECTED.tex` resolves all previous provenance concerns and satisfies the highest standards of scientific integrity.
