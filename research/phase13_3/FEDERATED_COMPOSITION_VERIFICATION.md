# PHASE 13.3 AUDIT: FEDERATED PRIVACY COMPOSITION & PROTOCOL BOUNDARIES

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3 — Final Source-Code Verification, DP Accounting Reproducibility and Experiment Consistency Audit  
**Author:** Independent Cybersecurity & Privacy Audit Specialist  
**Date:** November 2026  
**Status:** Protocol Scope Formally Delineated  

---

## 1. Executive Summary & Critical Protocol Finding

A central objective of the Phase 13.3 audit is to examine whether the reported differential privacy budgets ($\epsilon \in \{1.0, 2.0, 4.0, 8.0\}$) cover the **complete multi-round federated training protocol** or **only a centralized DP training mechanism**.

**Definitive Audit Finding:** The reported $(\epsilon, \delta)$ differential privacy guarantees in `results/phase13_2/privacy_utility_metrics.csv` and `dp_accounting_trace.csv` strictly and exclusively cover the **centralized DP-SGD benchmark** on the pooled landmark training cohort ($N = 1,132$). In the federated learning experiments (`FedAvg`, `FedProx`, and personalization fine-tuning across 5 Dirichlet clients for 10 rounds), models were trained using standard local SGD with simulated cryptographic Secure Aggregation (SecAgg). **The federated experiments did not execute local DP-SGD within each round.**

Therefore, the manuscript must **not** claim that the federated training runs or the final federated global model achieve $(\epsilon=1.0)$-differential privacy. The manuscript must restrict formal DP claims strictly to the centralized benchmark where it has been verified, while presenting the federated learning and secure aggregation components as complementary architectural evaluations.

---

## 2. Formal Distinctions Across Federated Privacy Dimensions

To prevent scientific misrepresentation in IEEE peer review, the following protocol dimensions must be clearly delineated:

### 2.1 Centralized DP-SGD vs. Federated Local DP
- **What Was Evaluated Under DP:** `Classical_DP_MLP` and `Hybrid_DP_QNN` trained with per-sample gradient clipping ($C = 1.0$) and calibrated Gaussian noise on the unified training dataset of 1,132 encounters.
- **What Was Evaluated Under Federated Learning:** 5 Dirichlet non-IID clients ($\alpha = 0.5$) communicating with a central server for $R = 10$ rounds, where each client trained locally for $E = 2$ epochs per round without gradient clipping or noise injection.
- **Implication:** The federated rounds did not spend differential privacy budget because DP-SGD was not enabled in the client local training loop.

### 2.2 Multi-Round Federated DP Composition (Theoretical Framework)
If a federated learning protocol were to execute local DP-SGD within each round:
1. In each round $r \in \{1, \dots, R\}$, client $k$ would perform $E_{\text{local}} \times \lceil N_k / B \rceil$ steps of DP-SGD with local subsampling rate $q_k$.
2. Because the server observes client model updates in every round (or observes the aggregate), privacy degrades with each round.
3. Under Rényi Differential Privacy, total privacy loss over $R$ rounds would compose linearly:
   $$\epsilon_{\text{total}}(\alpha) = \sum_{r=1}^R \epsilon_r(\alpha)$$
4. To maintain a final bound of $\epsilon = 1.0$ after 10 rounds, each round would need to operate at a much tighter per-round budget ($\epsilon_r \approx 0.1$), requiring significantly higher noise multipliers ($\sigma > 10.0$) which would degrade local utility.
5. Because this multi-round local DP protocol was not executed, **no claim of federated differential privacy can be made**.

### 2.3 Record-Level Adjacency vs. Patient Multiplicity
- **Verified Unit:** Add/remove-one-encounter ($|\mathcal{D} \Delta \mathcal{D}'| = 1$ ICU stay).
- **Patient Reality:** In the training split, 182 patients have multiple ICU stays (up to 5 stays).
- **Group Privacy Degradation:** Under standard group privacy, if patient $j$ has $k_j$ stays in the database, the privacy loss bounds expand to $(k_j \epsilon, k_j e^{(k_j-1)\epsilon}\delta)$.
- **Mandatory Manuscript Disclosure:** The paper must explicitly disclose that the privacy guarantee is at the encounter level, and patient-level DP is not claimed.

### 2.4 Secure Aggregation is NOT Differential Privacy
- **Role of SecAgg:** Cryptographic Secure Aggregation (via Shamir secret sharing and Diffie-Hellman masking) ensures that the server learns *only* the aggregate sum $\sum_{k=1}^K w_k$, keeping individual client updates $w_k$ confidential from an honest-but-curious server during transit.
- **Limitation of SecAgg:** SecAgg provides **zero** statistical indistinguishability. If the aggregate model $\sum w_k$ is released, it can still memorize sensitive patient features and remain vulnerable to membership inference or reconstruction attacks.
- **Rule:** SecAgg and DP must be discussed as orthogonal, complementary defenses: SecAgg protects data in transit to the server; DP protects data in the released model against downstream adversaries.

---

## 3. Explicit Privacy Accounting & Scope Matrix

The table below provides a comprehensive accounting of every experimental mechanism in the repository, specifying what is formally private and what is not:

| Protocol Component | Target Artifact / Experiment | Mechanism | Subsampling Rate $q$ | Noise Scale $\sigma$ | Steps $T$ | Compositions | Target $\delta$ | Resulting $\epsilon$ | Adjacency Unit | Formal DP Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Centralized DP-MLP** | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Opacus) | 0.055556 | 3.8867 | 270 | 270 | $10^{-5}$ | **0.9953** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-MLP | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Opacus) | 0.055556 | 2.1924 | 270 | 270 | $10^{-5}$ | **1.9954** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-MLP | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Opacus) | 0.055556 | 1.3501 | 270 | 270 | $10^{-5}$ | **3.9985** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-MLP | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Opacus) | 0.055556 | 0.9277 | 270 | 270 | $10^{-5}$ | **7.9977** | 1 ICU encounter | **FORMALLY VERIFIED** |
| **Centralized DP-QNN** | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Exact Microbatch) | 0.027778 | 1.8066 | 180 | 180 | $10^{-5}$ | **0.9940** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-QNN | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Exact Microbatch) | 0.027778 | 1.1890 | 180 | 180 | $10^{-5}$ | **1.9959** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-QNN | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Exact Microbatch) | 0.027778 | 0.8728 | 180 | 180 | $10^{-5}$ | **3.9949** | 1 ICU encounter | **FORMALLY VERIFIED** |
| Centralized DP-QNN | `results/phase13_2/privacy_utility_metrics.csv` | Poisson DP-SGD (Exact Microbatch) | 0.027778 | 0.6656 | 180 | 180 | $10^{-5}$ | **7.9944** | 1 ICU encounter | **FORMALLY VERIFIED** |
| **Federated FedAvg** | `results/phase13_1/metrics/federated_metrics.csv` | Standard FedAvg | None | 0.0 | 10 rounds $\times$ 2 epochs | 10 rounds | N/A | $\infty$ (Non-private) | N/A | **NOT DP (Architectural Only)** |
| **Federated FedProx** | `results/phase13_1/metrics/federated_metrics.csv` | FedProx ($\mu = 0.01$) | None | 0.0 | 10 rounds $\times$ 2 epochs | 10 rounds | N/A | $\infty$ (Non-private) | N/A | **NOT DP (Architectural Only)** |
| **Secure Aggregation** | `results/phase12/metrics/secure_aggregation_benchmark.csv` | Shamir Secret Sharing + DH Masking | None | 0.0 | 1 algebraic round | 1 round | N/A | N/A (Cryptographic) | N/A | **Confidentiality Only (Zero DP)** |
| **Personalization** | `results/phase13_2/federated_client_identity.csv` | Local Fine-Tuning | None | 0.0 | 3 epochs | 3 epochs | N/A | $\infty$ (Non-private) | N/A | **NOT DP (Utility Only)** |

---

## 4. Manuscript Directives

To maintain scientific integrity in the Phase 14 manuscript:
1. **Scope Restriction:** All differential privacy trade-off curves, numerical $(\epsilon, \delta)$ claims, and Membership Inference Attack evaluations must be explicitly described as evaluating **centralized DP-SGD training**.
2. **Federated Distinction:** The federated learning architecture must be presented as demonstrating feasibility of collaborative training on non-IID clinical data, while noting that integrating local DP across multiple communication rounds is a recognized open challenge and future research direction.
3. **Multi-Admission Patient Disclosure:** Explicitly include the group privacy formula showing how multi-admission patients experience expanded bounds.
