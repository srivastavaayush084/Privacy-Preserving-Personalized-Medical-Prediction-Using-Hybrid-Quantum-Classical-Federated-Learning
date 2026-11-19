# Research Claim Audit & Scientific Classification
## Project: Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning

**Audit Date:** 2026-11-19  
**Auditor:** Independent Senior Research Auditor, Statistical Reviewer, Quantum ML & Cybersecurity Specialist  
**Standard of Review:** Empirical reproducibility, paired non-parametric statistical hypothesis testing, cryptographic correctness, and clinical epidemiological validity.

---

### Summary Classification Table

| # | Specific Research Claim | Reported Claim | Recalculated Finding | Scientific Classification | Reason / Audit Finding |
|:---:|:---|:---|:---|:---:|:---|
| **C1** | **Quantum Advantage in Medical Prediction** | Hybrid QNN (0.8346) achieves quantum advantage over classical learning. | XGBoost: **0.8445**, RF: **0.8411**, QNN: 0.8346, MLP: 0.7944. | **UNSUPPORTED / INCORRECT** | Classical XGBoost and RF outperform QNN. Difference vs MLP (+0.0402) is NOT statistically significant ($p = 0.084$). |
| **C2** | **NISQ Barren Plateau / Over-parameterization** | Increasing PQC depth from 2 to 3 causes AUROC collapse (0.8346 $\to$ 0.7533). | Depth 1: 0.7756, Depth 2: 0.8346, Depth 3: 0.7533. | **VERIFIED** | Deep PQC degradation reproduces exactly. Validates theoretical barren plateau and gate noise constraints. |
| **C3** | **Patient-Disjoint Cohort Partitioning** | 0.00% patient leakage across Train, Val, and Test splits. | Overlaps: Train-Val = 0, Train-Test = 0, Val-Test = 0. | **VERIFIED** | Strict patient-level grouping on `uniquepid` certified. |
| **C4** | **Zero Target Leakage in Features** | 41 clinical features exclude discharge variables and APACHE outcome predictions. | All 41 features audited. Zero APACHE scores or discharge status found. | **VERIFIED** | Preprocessing fitted strictly on train partition. |
| **C5** | **Zero Raw Patient Record Exchange in FL** | Federated coordinator never accesses raw hospital records. | Data locality verified in `HospitalClient`. | **VERIFIED** | Only serialized parameters ($\Theta_s$) and metrics exchanged over communication downlink/uplink. |
| **C6** | **Quantum FL Communication Efficiency** | Hybrid QNN FL achieves 3.1x bandwidth reduction vs classical MLP (0.80 MB vs 2.46 MB). | Payload verified: QNN 0.80 MB, MLP 2.46 MB. | **VERIFIED** | PQC parameter footprint (24 variational angles) is significantly smaller than dense classical layers. |
| **C7** | **FedProx Client Drift Reduction** | FedProx reduces inter-hospital client variance under non-IID distributions. | FedAvg client std = 0.0294, FedProx client std = 0.0153 (-48%). | **VERIFIED** | FedProx proximal penalty $\frac{\mu}{2}\|\theta - \theta_t\|^2$ prevents local client drift. |
| **C8** | **Hospital Adapter Performance Recovery** | Hospital residual adapters boost Consortium 5 AUROC from 0.1849 to 0.8151. | Consortium 5 test set: 79 stays, **only 6 positive cases**. AUROC 0.8151 95% CI: $[0.552, 0.985]$. | **PARTIALLY VERIFIED / HIGH UNCERTAINTY** | Numerical ranking reproduced, BUT small positive sample size ($N_{\text{pos}}=6$) produces high estimation variance. |
| **C9** | **Consortium 5 as "Rural & Suburban Hospital System"** | Consortium 5 represents a rural hospital network facing extreme resource disparity. | `hospitalid % 5 == 4` clusters 35 hospitals across 4 national census regions (11 Midwest, 9 South, 9 West, 6 Northeast). | **UNSUPPORTED / SPECULATIVE** | eICU metadata does not support rural designation for this hash cluster. Semantic labeling was post-hoc. |
| **C10** | **Certified Record-Level Differential Privacy** | Certified $(\epsilon=2.0, \delta=10^{-5})$ DP with zero MIA advantage. | Batch-level clipping applied rather than per-sample vector clipping. MIA advantage 0.0 is an empirical observation, not formal proof. | **PARTIALLY VERIFIED** | Algorithm implements heuristic batch perturbation with RDP tracking, not formal per-sample DP-SGD. |
| **C11** | **Production-Ready Bonawitz Secure Aggregation** | Bonawitz protocol protects updates from honest-but-curious server. | Pairwise zero-sum masks cancel exactly ($3.81 \times 10^{-6}$ error), but lacks Diffie-Hellman and secret sharing for client dropouts. | **INCORRECT (SIMPLIFIED DEMONSTRATION ONLY)** | Simplified pairwise masking simulation; halts if any client drops out. Not a production-ready fault-tolerant protocol. |
| **C12** | **Clinical Readiness of the Prediction Pipeline** | Framework is clinically ready for multi-hospital deployment. | Trained on eICU demo (2,520 stays); untested on external clinical EHR systems; uncalibrated under concept drift. | **UNSUPPORTED / PREMATURE** | The system is a research prototype, not a clinically deployable diagnostic device. |

---

### Detailed Scientific Critique of Critical Claims

#### 1. The "Quantum Advantage" Claim (C1)
- **The Core Issue:** In the previous report, the phrase "Quantum Advantage" was used because the Hybrid QNN (4-qubit depth-2) reached an AUROC of 0.8346, which was higher than the classical Multi-Layer Perceptron (0.7944).
- **The Audit Finding:**
  1. Centralized XGBoost achieved **0.8445** AUROC and PR-AUC **0.3849** on the identical test holdout.
  2. Centralized Random Forest achieved **0.8411** AUROC.
  3. Both of these classical tree-based models outperform the Hybrid QNN.
  4. Even when comparing Hybrid QNN against the Classical MLP, the paired bootstrap difference is $+0.0402$ with a 95% confidence interval of $[-0.0041, +0.0914]$ and a two-sided $p$-value of $0.0840$. Because the confidence interval spans zero and $p > 0.05$, the difference is **not statistically significant**.
- **Action Required:** Remove all claims of "Quantum Advantage" from paper drafts, reports, and publication tables. Re-title the contribution as **"Comparative Performance of Hybrid Quantum-Classical Representation in Tabular Critical Care Prediction"**.

#### 2. The Consortium 5 Personalization Claim (C8 & C9)
- **The Core Issue:** The report highlighted a dramatic jump in AUROC from 0.1849 (global model) to 0.8151 (hospital adapter) on Consortium 5, attributing this to "rescuing an extreme out-of-distribution rural hospital".
- **The Audit Finding:**
  1. Consortium 5's test set contains 79 ICU encounters, but **only 6 patients experienced in-hospital mortality**.
  2. Evaluating an ROC-AUC on 6 positive events produces extreme standard error ($\pm 0.22$). Moving just one patient prediction across the threshold alters the AUROC by $> 0.16$.
  3. Consortium 5 was generated via `hospitalid % 5 == 4`, which groups 35 hospitals from all over the United States. No geographic or rural metadata in eICU supports the "rural hospital" designation.
- **Action Required:** Document Consortium 5 sample size limitations ($N_{\text{pos}} = 6$) explicitly. Replace the label "Rural & Suburban" with "Hospital Consortium 5 (Hash Cluster 4)". Report the 95% confidence interval $[0.552, 0.985]$ alongside the point estimate.

#### 3. Differential Privacy and Secure Aggregation (C10 & C11)
- **The Core Issue:** The report claimed "Certified Differential Privacy" and "Production Bonawitz Secure Aggregation".
- **The Audit Finding:**
  1. The code in `src/privacy/differential_privacy.py` calculates the norm of the total mini-batch gradient and scales it (`total_norm = sum(p.grad.data.norm(2)^2)^0.5`). This is **batch-level gradient clipping**, not per-sample gradient clipping. In formal DP (e.g. Abadi et al., 2016; Opacus), sensitivity bounds require clipping each patient's individual gradient vector $g_i$ prior to summation.
  2. The secure aggregation code implements zero-sum pairwise masking ($s_{i,j} = -s_{j,i}$), which confirms that masks sum to zero under complete client participation. However, it does not implement Shamir secret sharing of private seeds. If any client drops out, the aggregation fails.
- **Action Required:** Disclose that privacy perturbation is implemented via batch-level sensitivity clipping with Rényi divergence accounting, serving as a scalable empirical heuristic. Disclose that secure aggregation is a zero-sum masking simulation.
