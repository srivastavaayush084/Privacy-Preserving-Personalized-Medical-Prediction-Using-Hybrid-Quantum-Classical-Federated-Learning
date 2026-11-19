# Final Research Limitations & Scientific Disclosure
## Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning

**Document Version:** 1.0 (Post-Audit Edition)  
**Target Submission:** IEEE Journal of Biomedical and Health Informatics / Nature Scientific Reports  
**Ethical & Scientific Integrity Standard:** Full transparent disclosure of methodology boundaries, statistical uncertainties, and algorithmic simplifications.

---

### 1. Dataset Scale and Statistical Power Limitations

1. **Subsample Constraints of eICU Demo:**
   - The research experiments were conducted on the publicly accessible `eICU Collaborative Research Database Demo v2.0.1`, comprising **2,520 ICU admissions** across **186 hospital facilities**.
   - After strict patient-disjoint filtering (`uniquepid` isolation), the total cohort consists of 1,712 training stays, 387 validation stays, and **393 test holdout stays**.
2. **Severe Class Imbalance and Small Positive Cell Counts:**
   - In-hospital mortality is rare ($8.41\%$ overall prevalence in eICU Demo).
   - In the centralized test holdout ($N = 393$), there are **only 40 positive (Expired) cases**.
   - When partitioned into 5 federated hospital consortiums:
     - Consortium 1 test set: 77 stays, **4 positive cases** ($5.2\%$)
     - Consortium 2 test set: 63 stays, **3 positive cases** ($4.8\%$)
     - Consortium 3 test set: 76 stays, **16 positive cases** ($21.1\%$)
     - Consortium 4 test set: 98 stays, **11 positive cases** ($11.2\%$)
     - Consortium 5 test set: 79 stays, **6 positive cases** ($7.6\%$)
3. **Statistical Uncertainty on Per-Client Personalization Metrics:**
   - Estimating receiver operating characteristic curves (ROC-AUC) on sub-cohorts with 3 to 6 positive events produces massive standard errors ($\pm 0.15$ to $\pm 0.25$).
   - The reported AUROC recovery on Consortium 5 (from 0.1849 to 0.8151) has a **95% bootstrap confidence interval of $[0.5520, 0.9850]$**.
   - While the directional benefit of local residual adaptation is computationally verified, claims of definitive clinical efficacy must be moderated until validated on the full eICU dataset ($> 200,000$ stays).

---

### 2. Quantum Machine Learning Boundaries

1. **Absence of Empirical "Quantum Advantage":**
   - The hybrid quantum neural network (4-qubit, depth-2 strongly entangling circuit) achieved an AUROC of **0.8346** on the test holdout.
   - However, standard centralized **XGBoost (AUROC 0.8445)** and **Random Forest (AUROC 0.8411)** both outperformed the hybrid quantum model on the exact same patient-disjoint split.
   - Paired bootstrap hypothesis testing demonstrated that the AUROC difference between Hybrid QNN and Classical MLP ($+0.0402$) is **not statistically significant ($p = 0.0840$)**.
   - Therefore, the findings must be presented as an **exploratory feasibility study of compact quantum variational representations in healthcare**, rather than proof of computational or predictive quantum advantage.
2. **Barren Plateaus in Deeper Circuits:**
   - Experiments confirmed that increasing PQC circuit depth from 2 to 3 caused AUROC degradation (from 0.8346 down to **0.7533**), reflecting barren plateau gradient concentration and over-parameterization on tabular clinical data.
3. **Classical Simulation vs. Physical Quantum Hardware:**
   - Circuits were evaluated using PennyLane's `default.qubit` statevector simulator on classical CPU hardware.
   - Physical NISQ processors suffer from thermal relaxation ($T_1$), dephasing ($T_2$), two-qubit CNOT gate infidelities ($0.5\% - 2.0\%$), and state preparation/measurement (SPAM) errors.
   - While our density matrix depolarizing simulation (`default.mixed`) confirmed noise tolerance at $p_{\text{depol}} \le 0.02$, physical hardware deployment requires active quantum error mitigation (e.g. Zero-Noise Extrapolation).

---

### 3. Federated Learning & Systems Constraints

1. **Simulated vs. Production Distributed Infrastructure:**
   - Federated learning rounds were executed on a single host machine via Python orchestration, rather than across genuine physical network boundaries connecting hospital EHR firewalls.
   - Network latency, TLS handshake overhead, asynchronous client arrivals, socket dropouts, and hardware heterogeneity (e.g., CPU vs GPU hospital nodes) were modeled programmatically.
2. **Hash-Based Consortium Clustering:**
   - To achieve sufficient sample size per client, 186 hospitals were clustered into 5 federated consortiums using deterministic modulo partitioning (`hospitalid % 5`).
   - Although this preserved real-world inter-hospital non-IID variance and kept every patient in their true institutional cluster, semantic designations such as "Rural & Suburban" or "Academic Medical Center" were artificial post-hoc labels not derived from official eICU hospital metadata.

---

### 4. Differential Privacy & Cryptographic Scope

1. **Batch-Level vs. Per-Sample Gradient Clipping:**
   - Formal mathematical differential privacy (e.g., Abadi et al., 2016; Opacus DP-SGD) requires computing and clipping the $L_2$ norm of **each individual patient's gradient vector** prior to accumulation ($g_i \leftarrow g_i / \max(1, \|g_i\|_2 / C)$).
   - In our current implementation, the $L_2$ norm was computed and scaled over the **aggregated mini-batch gradient tensor**.
   - While this provides empirical perturbation and bounded updates tracked via Rényi Differential Privacy, it represents an engineering approximation of strict per-sample DP-SGD.
2. **Empirical Attack Limitations:**
   - The Membership Inference Attack (MIA) evaluated a black-box shadow model. An empirical attacker advantage of zero ($0.0000$) on the finite validation holdout demonstrates empirical resistance against this specific shadow attack family, but **does not constitute formal cryptographic proof** of zero membership leakage.
3. **Simplified Pairwise Additive Masking:**
   - The secure aggregation implementation demonstrated the core zero-sum additive masking concept ($s_{i,j} = -s_{j,i}$), achieving exact mathematical cancellation ($3.81 \times 10^{-6}$ error) under 100% client participation.
   - However, the protocol lacks Shamir's $t$-out-of-$n$ threshold secret sharing and double-masking. Consequently, if a participating hospital client drops out during the upload round, the masks cannot be unmasked, and the central aggregation round fails.

---

### 5. Clinical Deployment and Temporal Validity

1. **Static 24-Hour Admission Snapshot:**
   - The prediction target evaluates in-hospital mortality using features derived from the first 24 hours of ICU admission.
   - The model does not accommodate dynamic, continuously arriving physiological telemetry streams or sequential trajectory updates over an extended ICU length of stay.
2. **External Validation Deficit:**
   - The models have been evaluated exclusively on eICU Demo v2.0.1.
   - Generalizability to other healthcare systems (e.g., MIMIC-IV, AmsterdamUMCdb, or contemporary COVID-19/sepsis ICU cohorts) remains untested.
   - The framework must be classified as a **research proof-of-concept**, not an FDA-cleared or clinically deployable medical device.
