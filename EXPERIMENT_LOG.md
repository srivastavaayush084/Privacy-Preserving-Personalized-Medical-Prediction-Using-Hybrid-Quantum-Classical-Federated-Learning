# Formal Scientific Experiment Log
## Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning

**Execution Timestamp:** 2026-11-19  
**Platform:** Windows 11 x64, Python 3.11.9, PyTorch, PennyLane 0.45.1, Opacus 1.6.0, Scikit-Learn 1.8.dev0  
**Dataset:** eICU Collaborative Research Database Demo v2.0.1  
**Clinical Prediction Target:** In-Hospital Mortality (`hospitaldischargestatus == 'Expired'`) within 24h ICU admission  
**Partitioning Strategy:** Strict Patient-Disjoint Partitions (`uniquepid` isolation): Train (1,712 stays, 1,274 patients), Val (387 stays, 273 patients), Test (393 stays, 273 patients). 0.00% Patient Overlap.  

---

### Experiment E1: Centralized Classical Baselines
- **Objective:** Establish reproducible empirical upper bounds on centralized patient-disjoint holdouts.
- **Hyperparameters:** Cost-sensitive positive class weight = 10.89. L2 penalty, RBF SVM kernel, 150 RF estimators, XGBoost max_depth=4.
- **Empirical Results:**
  - **XGBoost:** AUROC **0.8445**, PR-AUC **0.3849**, Sensitivity 0.6500, Specificity 0.8640, Brier 0.1286, ECE 0.1900.
  - **Random Forest:** AUROC 0.8411, PR-AUC 0.3403, Sensitivity 0.6000, Specificity 0.8754, Brier 0.1216, ECE 0.2013.
  - **Logistic Regression:** AUROC 0.8170, PR-AUC 0.3569, Sensitivity 0.7750, Specificity 0.7224, Brier 0.1948, ECE 0.2879.
  - **Support Vector Machine (RBF):** AUROC 0.7948, PR-AUC 0.2508, Sensitivity 0.3500, Specificity 0.9405, Brier 0.0842, ECE 0.0272.
  - **Clinical Deep Tabular MLP:** AUROC 0.7944, PR-AUC 0.3295, Sensitivity 0.6500, Specificity 0.8017, Brier 0.1804, ECE 0.2203.
- **Artifacts:** `checkpoints/classical/`, `results/metrics/classical_baseline_metrics.csv`, `results/figures/baseline_roc_curves.png`.

---

### Experiment E2: Classical Federated Learning (FedAvg vs. FedProx)
- **Objective:** Evaluate federated optimization under natural clinical non-IID data across 5 hospital consortiums.
- **Protocol:** 10 communication rounds, 5 participating hospital consortium clients, 2 local epochs per round, learning rate 0.005.
- **Empirical Results:**
  - **Classical FL (FedAvg):** Global AUROC 0.7590, PR-AUC 0.2835, Worst-Client AUROC 0.7095, Client Std 0.0294, Bandwidth 2.46 MB.
  - **Classical FL (FedProx, $\mu=0.02$):** Global AUROC **0.7746**, PR-AUC 0.2730, Worst-Client AUROC **0.7482**, Client Std **0.0153**, Bandwidth 2.46 MB.
- **Key Finding:** FedProx reduces client variance by 48% and raises worst-client AUROC by +0.0387 under natural clinical heterogeneity.
- **Artifacts:** `checkpoints/federated/fed_fedavg_classical_best.pt`, `checkpoints/federated/fed_fedprox_classical_best.pt`, `results/metrics/federated_experiments_metrics.csv`.

---

### Experiment E3: Hybrid Quantum Federated Learning
- **Objective:** Evaluate federated training of Parameterized Quantum Circuits (PQCs) with classical bottleneck compression.
- **Protocol:** 4-qubit, depth-2 Strongly Entangling PQC (24 variational parameters, 36 quantum gates). 10 communication rounds, 5 consortium sites.
- **Empirical Results:**
  - **Hybrid Quantum FL (FedAvg):** Global AUROC **0.7868**, PR-AUC 0.2937, Sensitivity 0.6500, Worst-Client AUROC 0.6944, Bandwidth **0.80 MB**.
  - **Hybrid Quantum FL (FedProx, $\mu=0.02$):** Global AUROC 0.7638, PR-AUC **0.3318**, Sensitivity **0.7250**, Best-Client AUROC 0.9292, Bandwidth **0.80 MB**.
- **Key Finding:** Hybrid Quantum FL outperformed Classical FedAvg (0.7868 vs 0.7590) while requiring **3.1x less communication bandwidth** (0.80 MB vs 2.46 MB).
- **Artifacts:** `checkpoints/federated/fed_fedavg_quantum_best.pt`, `checkpoints/federated/fed_fedprox_quantum_best.pt`, `results/figures/federated_convergence.png`.

---

### Experiment E4 & E5: Hospital-Level Personalization
- **Objective:** Compare Global-Only, Local Fine-Tuning, Parameter-Isolated Hospital Adapters, and Local Temperature Calibration.
- **Empirical Results (Mean Client AUROC / Worst-Client AUROC):**
  - **Classical MLP (E4):**
    - Global Only: Mean 0.5249 | Worst 0.4281 | Std 0.1032
    - Local Fine-Tuning: Mean 0.6261 | Worst 0.4087 | Std 0.1521
    - **Hospital Adapter:** Mean **0.6289** | Worst **0.4995** | Std 0.1376 | Brier **0.2333**
  - **Hybrid Quantum (E5):**
    - Global Only: Mean 0.5374 | Worst 0.1849 | Std 0.2352 (Severe out-of-distribution failure on rural site)
    - Local Fine-Tuning: Mean 0.4509 | Worst 0.2055 | Std 0.1665 (Catastrophic forgetting)
    - **Hospital Adapter:** Mean **0.7214** | Worst **0.6886** | Std **0.0479** | Brier **0.2236** (Recovers rural site from 0.1849 to 0.8151!)
- **Artifacts:** `results/metrics/personalization_experiments_metrics.csv`, `results/metrics/per_hospital_personalization_gains.csv`, `results/figures/personalization_per_hospital_gains.png`.

---

### Experiment E6: Certified Differential Privacy & Adversarial Attacks
- **Objective:** Evaluate privacy-utility Pareto frontier and empirical defense against Membership Inference (MIA) and Gradient Inversion (DLG).
- **Protocol:** Rényi Differential Privacy (RDP) Subsampled Gaussian Mechanism ($\delta=10^{-5}$) across $\epsilon \in \{0.5, 1.0, 2.0, 5.0, 10.0, \infty\}$ and Bonawitz Pairwise Masking.
- **Empirical Results:**
  - $\epsilon = \infty$ (No DP): Test AUROC 0.8227 | MIA AUROC 0.5593 | MIA Advantage 0.0992 | DLG MSE 0.0919 (Vulnerable)
  - $\mathbf{\epsilon = 2.0}$ **(Pareto Optimal):** Test AUROC **0.6659** | MIA AUROC **0.4900** | MIA Advantage **0.0000** | DLG MSE 0.1842 (Certified Protected)
  - $\epsilon = 1.0$ (Strict): Test AUROC 0.4750 | MIA AUROC 0.5102 | MIA Advantage 0.0000
  - **Secure Aggregation:** Exact pairwise mask zero cancellation error = $3.81 \times 10^{-6}$ (information-theoretic hiding from coordinator).
- **Artifacts:** `results/metrics/privacy_utility_tradeoff.csv`, `results/metrics/privacy_attacks_benchmark.csv`, `results/figures/privacy_utility_pareto_curve.png`.


---

### Phase 12: Scientific Remediation, Formal Privacy Implementation & Advanced Validation
- **Objective:** Address all limitations identified in Phase 11 independent audit.
- **Key Deliverables & Implementations:**
  1. **Formal Per-Sample DP-SGD (`src/privacy/formal_dp_engine.py`, `experiments/privacy/run_formal_dp.py`):**
     - Classical MLP: Replaced BatchNorm with GroupNorm (`ClinicalDPMLPNet`) and trained via official Opacus 1.6 `PrivacyEngine`.
     - Hybrid QNN: Engineered micro-batched per-sample autograd engine with individual $L_2$ clipping ($C=1.0$) and calibrated Gaussian noise tracked via Opacus `RDPAccountant` ($\delta = 10^{-5}$).
     - Evaluated $\epsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$. Certified MIA defense (MIA AUC $\approx 0.50$). Checkpoints saved to `results/phase12/checkpoints/`.
  2. **Dropout-Resilient Secure Aggregation (`src/privacy/secure_aggregation_v2.py`, `scripts/validate_secure_aggregation_v2.py`):**
     - Full 4-round cryptographic protocol incorporating Shamir $(t, n)$ threshold secret sharing.
     - Verified exact algebraic reconstruction across 0%, 20%, 40% client dropouts (Max Error $\le 1.73 \times 10^{-6}$, RMSE $\le 4.46 \times 10^{-7}$). Verified clean abort when drops exceed threshold.
  3. **Advanced Federated Learning (`experiments/federated/run_phase12_federated.py`):**
     - Dirichlet non-IID partitioning ($\alpha=0.5$).
     - Dual-metric reporting: Round 10 asymptotic performance vs Peak validation round. Checkpoints preserved separately.
     - Communication payload tracked: Quantum FL requires only 0.80 MB vs 2.88 MB for Classical FL (72.2% reduction).
  4. **Advanced Hospital Personalization (`experiments/personalization/run_phase12_personalization.py`):**
     - Transparent reporting of sample sizes ($N_{\text{test}}, N_{\text{pos}}$) and 95% bootstrap confidence intervals for all nodes.
  5. **Quantum Resource and Noise Profile (`experiments/quantum/run_phase12_quantum_profile.py`):**
     - Clarified that Tree Ensembles (Random Forest 0.8411, XGBoost 0.8334) outperform neural and quantum models.
     - Validated quantum parameter efficiency (24 variational angles in core vs 5,313 weights in MLP) and resilience under depolarizing noise ($p \le 0.05$) and finite shots ($S \ge 250$).
  6. **External Validation Readiness (`src/data/external_dataset_adapter.py`, `research/EXTERNAL_VALIDATION_READINESS.md`):**
     - Provided schema mapping for MIMIC-IV v2.2 and Full eICU v2.0 with PhysioNet credentialing and DUA protocol.
  7. **Automated Unit Testing:**
     - Expanded test suite: 42/42 tests passing (100% pass rate).
- **Artifacts:**
  - Reports: `research/PHASE_12_REMEDIATION_REPORT.md`, `research/FORMAL_DP_VALIDATION.md`, `research/SECURE_AGGREGATION_VALIDATION.md`, `research/ADVANCED_FEDERATED_VALIDATION.md`, `research/EXTERNAL_VALIDATION_READINESS.md`.
  - Metrics: `results/phase12/metrics/formal_dp_metrics.csv`, `results/phase12/metrics/secure_aggregation_benchmark.csv`, `results/phase12/metrics/advanced_federated_metrics.csv`, `results/phase12/metrics/advanced_personalization_metrics.csv`, `results/phase12/metrics/quantum_detailed_profile.csv`.
  - Figures: `results/phase12/figures/formal_dp_privacy_utility_tradeoff.png`, `results/phase12/figures/advanced_federated_validation.png`, `results/phase12/figures/advanced_personalization_gains.png`, `results/phase12/figures/quantum_noise_decay_profile.png`.

---

## [Phase 13.3.4] - Final Statistical Audit, Calibration Validation & Decision-Curve Correction (2026-11-19)
- **Objective:** Remediate Phase 13.3.3 feedback regarding prevalence baselines, DCA crossover, Platt scaling stability, operating point uncertainty, and quantum advantage framing.
- **Commands Executed:**
  - `python experiments/phase13_3_4/run_phase13_3_4_audit.py`
  - `python -m pytest` (70/70 passing)
- **Key Results:**
  - Training Prevalence Baseline: $p = 0.078622$, Test Brier $= 0.104095$, LogLoss $= 0.367836$, ECE $= 0.037561$.
  - Validation Prevalence Baseline: $p = 0.102564$, Test Brier $= 0.102870$, LogLoss $= 0.360220$, ECE $= 0.013618$.
  - Oracle Test Reference: $p = 0.116183$, Test Brier $= 0.102684$, LogLoss $= 0.359249$, ECE $= 0.000000$.
  - Calibrated Models: QNN Platt Brier $= 0.097960$, MLP Platt Brier $= 0.096438$ (both strictly outperform all 3 baselines).
  - Validation Resampling Stability ($B=1,000$): QNN slope 95% CI: $[-0.1176, 0.8572]$; mean test Brier $= 0.099371$ [0.096666, 0.105395].
  - Paired Differences ($B=2,000$ test replicates): $\Delta \text{AUROC} = +0.0494$ ($p = 0.290$, NS); $\Delta \text{AUPRC} = +0.0244$ ($p = 0.624$, NS); $\Delta \text{Brier (Platt)} = +0.0015$ ($p = 0.629$, NS).
  - Corrected DCA: Crossover identified at $p_t = 0.09$; at $p_t = 0.08$ Treat-All ties QNN ($0.039329$) and beats MLP ($0.038968$). Clinical utility window: $p_t \in [0.09, 0.25]$.
- **Artifacts Generated:**
  - Reports: `research/phase13_3_4/` (7 markdown files).
  - Data & Plots: `results/phase13_3_4/` (6 CSVs, 5 high-res PNG plots, manifest, sha256 checksums).
