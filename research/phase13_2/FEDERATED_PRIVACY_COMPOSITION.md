# PHASE 13.2 AUDIT: FORMAL PRIVACY COMPOSITION IN FEDERATED LEARNING

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.2 — Final DP Mechanism Verification & Statistical Audit  
**Author:** Independent Cybersecurity & Privacy Audit Specialist  
**Date:** November 2026  
**Status:** Formally Specified & Audited  

---

## 1. Executive Summary & Audit Mandate

A recurring point of ambiguity in privacy-preserving federated healthcare systems is the precise scope of the differential privacy guarantee. Scientific reviewers frequently challenge claims where:
1. Record-level differential privacy is conflated with patient-level differential privacy.
2. Local training privacy is conflated with multi-round federated communication privacy.
3. Cryptographic secure aggregation (SecAgg) is conflated with differential privacy (DP).

The Phase 13.2 audit mandate requires a formal mathematical treatment distinguishing local DP training, repeated training, multiple federated rounds, client participation frequency, and multi-admission patient records. 

**Core Audit Finding & Commitment:** This project strictly implements and guarantees **Record-Level Differential Privacy** under add/remove-one-encounter adjacency ($|\mathcal{D} \Delta \mathcal{D}'| = 1$ ICU stay). It does **not** claim Patient-Level DP, and all manuscript representations must explicitly adhere to this boundary.

---

## 2. Record-Level DP vs. Patient-Level DP

### 2.1 Adjacency Definition
In the clinical landmark cohort, data instances correspond to individual ICU stays (encounters):
- Locked landmark cohort: $N = 1,607$ ICU stays across $M = 1,317$ unique patients.
- Training split: $N_{\text{train}} = 1,132$ ICU stays across $M_{\text{train}} = 921$ unique patients.
- Multiplicity: 182 patients have $\ge 2$ ICU stays (maximum 5 stays for a single patient).

Under **Record-Level Adjacency**, two datasets $\mathcal{D}, \mathcal{D}'$ are adjacent if they differ by at most one ICU stay record:

$$\text{dist}_{\text{record}}(\mathcal{D}, \mathcal{D}') = |\mathcal{D} \Delta \mathcal{D}'| = 1$$

Under this guarantee, the $L_2$ gradient sensitivity of the clipped sum is strictly:

$$\Delta_2 = C = 1.0$$

### 2.2 Mathematical Degradation to Patient-Level Privacy (Group Privacy)
If a patient $j$ has $k_j$ distinct ICU admissions in $\mathcal{D}$, changing the presence of patient $j$ alters $k_j$ records simultaneously:

$$\text{dist}_{\text{patient}}(\mathcal{D}, \mathcal{D}') = k_j \le k_{\text{max}} = 5$$

By the standard Group Privacy Theorem (Dwork & Roth, 2014):
If a mechanism $\mathcal{M}$ is $(\epsilon, \delta)$-record-level DP, then for a group of $k$ records (i.e., a patient with $k$ admissions), $\mathcal{M}$ satisfies $(k\epsilon, k e^{(k-1)\epsilon}\delta)$-DP.

For example, at target record-level $\epsilon = 1.0, \delta = 10^{-5}$:
- A patient with $k=1$ admission has $(\epsilon=1.0, \delta=10^{-5})$.
- A patient with $k=3$ admissions has effective bound $(\epsilon \approx 3.0, \delta \approx 2.2 \times 10^{-4})$.
- A patient with $k=5$ admissions has effective bound $(\epsilon \approx 5.0, \delta \approx 2.7 \times 10^{-3})$.

**Formal Requirement:** The manuscript must state:
> *"The differential privacy mechanisms evaluated in this work provide record-level differential privacy (where one record corresponds to one ICU encounter). For patients with multiple ICU admissions, privacy protection degrades according to standard group privacy bounds. Patient-level DP would require clipping the aggregate gradient across all encounters per patient and is an area for future work."*

---

## 3. Privacy Composition Across Federated Dimensions

### 3.1 Local Training vs. Central Release
We distinguish three distinct architectures:
1. **Centralized DP-SGD:** A single trusted server trains on pooled data with per-sample DP-SGD. The output model achieves $(\epsilon, \delta)$-DP.
2. **Local DP (LDP) in Federated Learning:** Each client applies DP-SGD locally before transmitting model weights $\theta_k^{(r)}$ to the aggregator. The transmitted update is itself $(\epsilon_{\text{local}}, \delta_{\text{local}})$-DP.
3. **Central DP with Secure Aggregation:** Clients send unperturbed clipped updates via cryptographic secure aggregation; noise is added to the sum at the server (or via distributed noise addition).

### 3.2 Composition Across Federated Communication Rounds
When clients participate in $R$ communication rounds of federated training, where in each round $r \in \{1, \dots, R\}$ a local DP mechanism $\mathcal{M}_r$ with budget $(\epsilon_r, \delta_r)$ is executed:
- **Linear RDP Composition:** Under Rényi Differential Privacy, the Rényi divergences $\mathbb{D}_\alpha$ compose linearly across rounds:

$$\mathbb{D}_\alpha(\mathcal{M}_{1:R}(\mathcal{D}) \parallel \mathcal{M}_{1:R}(\mathcal{D}')) = \sum_{r=1}^R \mathbb{D}_\alpha(\mathcal{M}_r(\mathcal{D}) \parallel \mathcal{M}_r(\mathcal{D}'))$$

- **Total Budget:** If client $k$ participates in all $R$ rounds with noise multiplier $\sigma$ and local subsampling rate $q_{\text{local}}$, the effective step count in the privacy accountant is $T_{\text{total}} = R \times E_{\text{local}} \times \lceil 1/q_{\text{local}} \rceil$.
- If a client trains locally for multiple rounds, the total privacy budget must be tracked over the full sequence of rounds.

### 3.3 Repeated Training and Hyperparameter Tuning
Every execution of DP-SGD on the private dataset consumes privacy budget. Specifically:
- Running grid search across learning rates $\eta \in \{0.001, 0.005, 0.01\}$ without privacy accounting would consume $3 \times$ the stated budget.
- In this study, hyperparameter selection was conducted on non-private validation holdouts, and the reported $(\epsilon, \delta)$ pairs represent the single execution of the finalized training runs.

---

## 4. Disentangling Differential Privacy and Secure Aggregation

Differential Privacy (DP) and Secure Aggregation (SecAgg) address complementary threat vectors and must never be conflated:

```
+-----------------------------------------------------------------------------------+
| THREAT VECTOR 1: Honest-but-Curious Server / Eavesdropper on Network Traffic      |
| DEFENSE: Secure Aggregation (Diffie-Hellman / Shamir Secret Sharing / Masking)     |
| GUARANTEE: Server learns ONLY the aggregate sum \sum_k w_k, never individual w_k. |
| LIMITATION: Does NOT prevent memorization of training samples in the sum!         |
+-----------------------------------------------------------------------------------+
                                         +
+-----------------------------------------------------------------------------------+
| THREAT VECTOR 2: Reconstruction / Membership Inference Attacks on Final Model     |
| DEFENSE: Differential Privacy (Per-Sample Gradient Clipping + Calibrated Gaussian) |
| GUARANTEE: Bounded statistical indistinguishability (eps, delta) of any record.   |
| LIMITATION: Injects noise that trades off with predictive utility.                |
+-----------------------------------------------------------------------------------+
```

In Phase 12 and 13.1, cryptographic SecAgg was verified to achieve exact algebraic reconstruction ($\|W_{\text{server}} - \sum w_k\|_2 < 10^{-6}$) with simulated Byzantine dropouts. However, SecAgg alone provides **zero** formal $(\epsilon, \delta)$ differential privacy. Formal differential privacy is provided exclusively by the per-sample DP-SGD engine (`FormalOpacusMLPTrainer` and `FormalPerSampleDPQNNTrainer`).

---

## 5. Formal Claims Matrix for IEEE Manuscript

| Concept | Audited & Supported Claim | Prohibited / Disallowed Claim |
|---|---|---|
| **Privacy Unit** | Record-level DP (1 ICU encounter) | Patient-level DP |
| **Adjacency** | Add/Remove-one-encounter ($|\mathcal{D} \Delta \mathcal{D}'| = 1$) | Zero-leakage for multi-admission patients |
| **Composition** | Analytical RDP accounting over total steps $T$ | Ad-hoc or informal composition claims |
| **SecAgg Role** | Confidential aggregation against honest-but-curious server | SecAgg provides differential privacy |
| **Quantum DP** | Exact micro-batched per-sample clipping with fixed $B_{\text{exp}}$ | Quantum supremacy or inherent quantum privacy |

---

## 6. Conclusion

By formally establishing the boundaries of record-level differential privacy, quantifying multi-admission group privacy degradation, and separating cryptographic secure aggregation from differential privacy accounting, the research is positioned on solid theoretical ground suitable for IEEE peer review.
