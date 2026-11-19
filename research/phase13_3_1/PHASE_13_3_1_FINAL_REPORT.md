# Phase 13.3.1 — Master Verification Report: DP Expected-Batch-Size & Sampling-Rate Consistency

**Project Title:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Audit Phase:** Phase 13.3.1 — Critical DP Expected-Batch-Size and Sampling-Rate Consistency Verification  
**Evaluation Role:** Senior Independent Scientific Auditor, Quantum Machine Learning Specialist, Healthcare AI Researcher & Cybersecurity Specialist  
**Clinical Prediction Task:** Subsequent in-hospital mortality prediction using observations available at the 24-hour ICU landmark ($T = 1,440$ min)  
**Landmark Clinical Cohort:** 1,607 stays across 1,317 unique patients ($N_{\text{train}} = 1,132$, $N_{\text{val}} = 234$, $N_{\text{test}} = 241$, Prevalence = 11.20%)  
**Verification Date:** November 19, 2026  
**Status:** Remediated, Fully Re-Executed, and Scientifically Verified  

---

## 1. Executive Summary

Phase 13.3 reported verified differential privacy accounting for the Phase 13.2 experimental schedule. However, a rigorous audit identified a mathematical inconsistency between the stated Poisson sampling probabilities ($q$) and the expected batch size normalization constants ($B_{\text{exp}}$) reported in documentation:

1. **The Inconsistency:**
   - Training cohort size: $N = 1,132$ ICU stays.
   - Hybrid DP-QNN: Documented as $q = 1/36$, nominal batch size 32, and reported normalization denominator $B_{\text{exp}} = 32.0$. However:
     $$qN = \frac{1,132}{36} = 31.444444... \ne 32.0$$
   - Classical DP-MLP: Documented as $q = 1/18$, nominal batch size 64, and reported normalization denominator $B_{\text{exp}} = 62.0$. However:
     $$qN = \frac{1,132}{18} = 62.888888... \ne 62.0$$
2. **Root Cause in Implementation:**
   - In `FormalPerSampleDPQNNTrainer`, `expected_batch_size` was hardcoded to `float(self.batch_size) = 32.0`, scaling gradients by $32.0$ instead of true mean batch size $31.4444$.
   - In `FormalOpacusMLPTrainer`, Opacus's `PrivacyEngine.make_private` internally truncated `int(len(dataset) * sample_rate) = int(1132 * (1/18)) = 62`, scaling gradients by $62.0$ instead of true mean batch size $62.8889$.
3. **Scientific Impact & Remediation:**
   - Because $32.0$ and $62.0$ are fixed, data-independent constants, dividing by them preserved the exact $(\epsilon, \delta)$ guarantee by the Post-Processing Theorem of Differential Privacy.
   - However, it introduced a minor gradient scaling distortion ($1.74\%$ shrinkage for QNN, $1.43\%$ inflation for MLP) and created a terminological falsehood in reporting that $B_{\text{exp}} = qN$.
   - Phase 13.3.1 formally evaluated **Option A** (preserve $q = 1/K$, normalize by exact float $qN$) versus **Option B** (preserve integer $B_{\text{exp}}$, derive $q = B_{\text{exp}} / N$).
   - **Option A was formally selected and implemented**, eliminating the gradient bias and harmonizing code, math, and documentation.
   - All affected DP-MLP and DP-QNN models were re-trained across all budgets ($\epsilon \in \{1.0, 2.0, 4.0, 8.0, \infty\}$), new checkpoints were preserved in `results/phase13_3_1/checkpoints/`, metrics were regenerated, and paired patient-clustered bootstrap tests were recomputed.
   - The entire unit test suite (70/70 tests) passed with 100% compliance.

---

## 2. Exhaustive Executable Parameter Inventory

The table below records the 10 critical parameters read directly from executable code (`src/privacy/formal_dp_engine.py`) and runtime execution traces:

| Parameter # | Description | Hybrid DP-QNN (Phase 13.3.1 Corrected) | Classical DP-MLP (Phase 13.3.1 Corrected) | Verification Source |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Actual training dataset size ($N$) | **1,132** encounters | **1,132** encounters | `data/processed_landmark/X_train.npy` |
| **2** | Actual Poisson sampling probability ($q$) | **$1/36 \approx 0.0277778$** | **$1/18 \approx 0.0555556$** | `formal_dp_engine.py:333`, Opacus `DPDataLoader` |
| **3** | Expected batch size passed to optimizer | **$31.444444...$** (`float(N * q)`) | **$62.888888...$** (`float(N * q)`) | `formal_dp_engine.py:338`, `optimizer.expected_batch_size` |
| **4** | Actual denominator used in normalization | **$31.444444...$** (`(acc_g + noise) / B_exp`) | **$62.888888...$** (`p.grad /= B_exp`) | `formal_dp_engine.py:406`, Opacus `DPOptimizer.scale_grad` |
| **5** | Actual clipping norm ($C$) | **1.0** | **1.0** | `self.max_grad_norm = 1.0` |
| **6** | Actual Gaussian noise std added to sum | **$\sigma \cdot C = \sigma$** | **$\sigma \cdot C = \sigma$** | `noise = randn * (sigma * C)`, Opacus `add_noise` |
| **7** | Actual noise multiplier ($\sigma$) | $\epsilon=1.0: 1.8066$<br>$\epsilon=2.0: 1.1890$<br>$\epsilon=4.0: 0.8728$<br>$\epsilon=8.0: 0.6656$ | $\epsilon=1.0: 3.8867$<br>$\epsilon=2.0: 2.1924$<br>$\epsilon=4.0: 1.3501$<br>$\epsilon=8.0: 0.9277$ | `results/phase13_3_1/dp_accounting_trace.csv` |
| **8** | Actual number of sampling attempts ($T$) | **180** ($5 \text{ epochs} \times 36 \text{ steps}$) | **270** ($15 \text{ epochs} \times 18 \text{ steps}$) | Outer epoch/batch loop count |
| **9** | Actual accountant sampling probability ($q_{\text{acc}}$) | **$1/36 \approx 0.0277778$** | **$1/18 \approx 0.0555556$** | Opacus `RDPAccountant` step hook |
| **10** | Actual accountant step count ($T_{\text{acc}}$) | **180** (private runs); 0 (inf baseline) | **270** (private runs); 0 (inf baseline) | `results/phase13_3_1/dp_accounting_trace.csv` |

---

## 3. Formal Comparative Evaluation: Option A vs. Option B

The user prompt mandated an explicit evaluation of Option A and Option B without silent assumption:

```
Option A: Preserve q and normalize by exact qN.
  QNN: q = 1/36, B_exp = 31.4444.
  MLP: q = 1/18, B_exp = 62.8889.

Option B: Preserve nominal expected batch sizes and derive q from them.
  QNN: B_exp = 32.0, q = 32 / 1132 = 0.02826855.
  MLP: B_exp = 62.0, q = 62 / 1132 = 0.05477032 (or 64 / 1132 = 0.05653710).
```

### 3.1 Detailed Technical Evaluation

| Aspect | Option A (Preserve $q = 1/K$, $B_{\text{exp}} = qN$) | Option B (Preserve $B_{\text{nominal}}$, derive $q = B / N$) | Evaluation Verdict |
| :--- | :--- | :--- | :--- |
| **Epoch Definition & Sampler Schedule** | $K = \lceil N / B_{\text{nominal}} \rceil$ is integer ($36$ for QNN, $18$ for MLP). In Opacus `UniformWithReplacementSampler`, `steps = int(1/q) = K` with zero fractional loss. Every record is sampled on average $K \cdot q = 1.0$ times per epoch. | `int(1/q) = int(1132/32) = 35` for QNN. Drops $0.375$ batches per epoch. Total steps $T = 175$ (not 180). Expected draws per record per epoch = $35 \times (32/1132) = 0.9894 < 1.0$. Cohort systematically undersampled by 12 patients per epoch. | **Option A is mathematically superior.** Option B creates schedule truncation defects. |
| **Gradient Expectation Unbiasedness** | $\mathbb{E}[\tilde{g}] = \frac{q \sum \bar{g}_i}{qN} = \frac{1}{N} \sum \bar{g}_i$. Exactly unbiased estimator of the population mean clipped gradient. | $\mathbb{E}[\tilde{g}] = \frac{q \sum \bar{g}_i}{B} = \frac{(B/N) \sum \bar{g}_i}{B} = \frac{1}{N} \sum \bar{g}_i$. Unbiased, but under an irregular 35-step epoch schedule. | **Both unbiased**, but Option A maintains standard epoch geometry. |
| **Privacy Accounting Consistency** | $(q = 1/36, T = 180)$ and $(q = 1/18, T = 270)$. Calibrated noise multipliers $\sigma$ and achieved $\epsilon$ values match validated historical composition curves identically. | $q$ changes to $0.028269$. If $T=175$, $\sigma=1.8164$ (at $\epsilon=1.0$). If $T=180$, $\sigma=1.8359$. Invalidates all historical privacy checkpoints. | **Option A preserves validated privacy calibration.** |
| **Opacus Compatibility** | Fully compatible. Double-precision float divisor $B_{\text{exp}} = 62.888888888888886$ executes natively in `DPOptimizer.scale_grad()`. | Requires custom sampling wrappers to avoid integer truncation in sampler length. | **Option A is cleanly executable.** |

### 3.2 Formal Selection
**Option A is selected as the scientifically defensible and mathematically rigorous configuration.**  
It enforces $B_{\text{exp}} = qN = 31.444444...$ for QNN and $B_{\text{exp}} = qN = 62.888888...$ for MLP.

---

## 4. Verification of the Gaussian Mechanism & Empty Batch Handling

### 4.1 Gaussian Mechanism Verification
For every DP execution:
1. **Per-Sample Clipping:** Every sample gradient $g_i$ satisfies $\|\bar{g}_i\|_2 \le C = 1.0$.
2. **Noise Calibration:** Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2 C^2 I)$ is added directly to the un-normalized sum of clipped gradients $\sum_{i \in B_t} \bar{g}_i$.
3. **Data-Independent Normalization:** The noisy sum is divided by fixed constant $B_{\text{exp}} = qN$. By the Post-Processing Theorem, this operation has zero privacy leakage.
4. **No Secondary Scaling:** Gradients flow directly into the Adam optimizer without hidden scaling factors.

### 4.2 Empty Batch Handling & Collate Safeguard
Under Poisson subsampling, drawing an empty batch ($|B_t| = 0$) is mathematically possible.  
- **Discovered PyTorch Bug:** Standard PyTorch `default_collate` raises `IndexError: list index out of range` when passed an empty list `[]`.
- **Engine Remediation:** In `FormalPerSampleDPQNNTrainer`, a `safe_collate` wrapper was introduced:
  ```python
  def safe_collate(batch):
      if len(batch) == 0:
          return torch.empty((0, self.input_dim), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)
      return torch.utils.data.dataloader.default_collate(batch)
  ```
- **Mechanism Behavior on Empty Batch:**
  - `len(bx) == 0` skips gradient computation.
  - `accum_grads` remains all zeros.
  - Pure Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2 C^2 I)$ is divided by $B_{\text{exp}}$ and assigned to `p.grad`.
  - The accountant steps: `acc.step(noise_multiplier=sigma, sample_rate=sample_rate)`.
  - The optimizer steps: `optimizer.step()`.
  - Full trace equality holds: $T_{\text{sampler}} = T_{\text{optimizer}} = T_{\text{accountant}}$.

---

## 5. Phase 13.3.1 Experimental Re-Run Results

All 10 models (5 Classical DP-MLP, 5 Hybrid DP-QNN) were re-trained on the 24-hour landmark clinical cohort ($N_{\text{train}} = 1,132$, $N_{\text{test}} = 241$, 27 positive deaths) using the corrected Option A mechanism.

### 5.1 Privacy-Utility Benchmark Results

| Model Architecture | Target $\epsilon$ | Achieved $\epsilon$ ($\delta=10^{-5}$) | Noise $\sigma$ | Expected Batch Size $B_{\text{exp}}$ | Test AUROC | Test PRAUC | Test Brier | Test Accuracy | Test F1 | MIA AUC | MIA Adv | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.9953$ | $3.8867$ | **$62.8889$** | $0.5857$ | $0.1772$ | $0.1144$ | $0.8880$ | $0.0000$ | $0.5098$ | $0.0197$ | $0.38$ |
| **Classical DP-MLP** | $2.0$ | $1.9954$ | $2.1924$ | **$62.8889$** | $0.5778$ | $0.1898$ | $0.1145$ | $0.8880$ | $0.0000$ | $0.5134$ | $0.0268$ | $0.33$ |
| **Classical DP-MLP** | $4.0$ | $3.9985$ | $1.3501$ | **$62.8889$** | $0.6016$ | $0.1945$ | $0.1143$ | $0.8880$ | $0.0000$ | $0.5186$ | $0.0371$ | $0.33$ |
| **Classical DP-MLP** | $8.0$ | $7.9977$ | $0.9277$ | **$62.8889$** | $0.6281$ | $0.2006$ | $0.1142$ | $0.8880$ | $0.0000$ | $0.5244$ | $0.0488$ | $0.33$ |
| **Classical DP-MLP** | $\infty$ | $\infty$ | $0.0000$ | **$62.8889$** | $0.7166$ | $0.2722$ | $0.1621$ | $0.8714$ | $0.3099$ | $0.5312$ | $0.0624$ | $0.20$ |
| **Hybrid DP-QNN** | $1.0$ | $0.9940$ | $1.8066$ | **$31.4444$** | $0.5904$ | $0.1382$ | $0.1139$ | $0.8880$ | $0.0000$ | $0.5052$ | $0.0104$ | $43.91$ |
| **Hybrid DP-QNN** | $2.0$ | $1.9959$ | $1.1890$ | **$31.4444$** | $0.6184$ | $0.1522$ | $0.1139$ | $0.8880$ | $0.0000$ | $0.5088$ | $0.0176$ | $44.12$ |
| **Hybrid DP-QNN** | $4.0$ | $3.9949$ | $0.8728$ | **$31.4444$** | $0.6410$ | $0.1675$ | $0.1138$ | $0.8880$ | $0.0000$ | $0.5122$ | $0.0244$ | $44.05$ |
| **Hybrid DP-QNN** | $8.0$ | $7.9944$ | $0.6656$ | **$31.4444$** | $0.6648$ | $0.1817$ | $0.1136$ | $0.8880$ | $0.0000$ | $0.5165$ | $0.0330$ | $44.20$ |
| **Hybrid DP-QNN** | $\infty$ | $\infty$ | $0.0000$ | **$31.4444$** | $0.7658$ | $0.2957$ | $0.1106$ | $0.8880$ | $0.0000$ | $0.5210$ | $0.0420$ | $43.85$ |

*Reconciliation Note:* Due to the adaptive scaling invariance of the Adam optimizer ($\frac{m_t}{\sqrt{v_t} + \epsilon_{\text{Adam}}}$), the $1.74\%$ gradient normalization correction produced model parameter updates identical to Phase 13.2 within four decimal places. However, the physical divisor is now mathematically exact and verified.

---

## 6. Patient-Clustered Paired Bootstrap Statistical Comparison

To evaluate whether Hybrid DP-QNN outperforms Classical DP-MLP, we executed a patient-clustered paired bootstrap ($B = 1,000$ resamples with replacement, seed=42) on the test cohort ($N = 241$), clustering by `uniquepid`:

$$\Delta \text{Metric} = \text{Metric}_{\text{QNN}} - \text{Metric}_{\text{MLP}}$$

### 6.1 Paired Statistical Comparison Results

| Privacy Budget ($\epsilon$) | Metric | MLP Score | QNN Score | Point Difference ($\Delta$) | Bootstrap Mean Diff | Bootstrap SE | 95% Confidence Interval | Two-Sided $p$-value | Statistically Significant? | Scientific Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\epsilon = 1.0$** | **AUROC** | $0.5857$ | $0.5904$ | $+0.0047$ | $+0.0051$ | $0.0552$ | **$[-0.1028, +0.1135]$** | $0.920$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 1.0$** | **AUPRC** | $0.1772$ | $0.1382$ | $-0.0390$ | $-0.0398$ | $0.0438$ | **$[-0.1522, +0.0194]$** | $0.210$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 1.0$** | **Brier** | $0.1144$ | $0.1139$ | $-0.0005$ | $-0.0005$ | $0.0001$ | **$[-0.0008, -0.0002]$** | $< 0.001$ | Yes (Calibration) | Slight calibration difference |
| **$\epsilon = 2.0$** | **AUROC** | $0.5778$ | $0.6184$ | $+0.0406$ | $+0.0412$ | $0.0526$ | **$[-0.0570, +0.1489]$** | $0.416$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 2.0$** | **AUPRC** | $0.1898$ | $0.1522$ | $-0.0377$ | $-0.0384$ | $0.0517$ | **$[-0.1693, +0.0332]$** | $0.312$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 2.0$** | **Brier** | $0.1145$ | $0.1139$ | $-0.0006$ | $-0.0006$ | $0.0002$ | **$[-0.0009, -0.0003]$** | $< 0.001$ | Yes (Calibration) | Slight calibration difference |
| **$\epsilon = 4.0$** | **AUROC** | $0.6016$ | $0.6410$ | $+0.0394$ | $+0.0398$ | $0.0507$ | **$[-0.0583, +0.1404]$** | $0.396$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 4.0$** | **AUPRC** | $0.1945$ | $0.1675$ | $-0.0270$ | $-0.0275$ | $0.0511$ | **$[-0.1574, +0.0427]$** | $0.476$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 4.0$** | **Brier** | $0.1143$ | $0.1138$ | $-0.0005$ | $-0.0005$ | $0.0002$ | **$[-0.0010, -0.0002]$** | $< 0.001$ | Yes (Calibration) | Slight calibration difference |
| **$\epsilon = 8.0$** | **AUROC** | $0.6281$ | $0.6648$ | $+0.0367$ | $+0.0371$ | $0.0467$ | **$[-0.0553, +0.1279]$** | $0.424$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 8.0$** | **AUPRC** | $0.2006$ | $0.1817$ | $-0.0189$ | $-0.0194$ | $0.0511$ | **$[-0.1399, +0.0604]$** | $0.630$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = 8.0$** | **Brier** | $0.1142$ | $0.1136$ | $-0.0006$ | $-0.0006$ | $0.0003$ | **$[-0.0012, -0.0001]$** | $0.018$ | Yes (Calibration) | Slight calibration difference |
| **$\epsilon = \infty$** | **AUROC** | $0.7166$ | $0.7658$ | $+0.0491$ | $+0.0498$ | $0.0465$ | **$[-0.0424, +0.1397]$** | $0.310$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = \infty$** | **AUPRC** | $0.2722$ | $0.2957$ | $+0.0235$ | $+0.0241$ | $0.0537$ | **$[-0.0868, +0.1237]$** | $0.608$ | **No** | 95% CI spans zero; no difference |
| **$\epsilon = \infty$** | **Brier** | $0.1621$ | $0.1106$ | $-0.0516$ | $-0.0520$ | $0.0219$ | **$[-0.0955, -0.0094]$** | $0.018$ | Yes (Calibration) | Overfitting in non-DP MLP |

### 6.2 Definitive Statistical Verdict
Across all formal privacy budgets $\epsilon \in \{1.0, 2.0, 4.0, 8.0\}$ and the non-private baseline $\epsilon = \infty$:
- Every 95% bootstrap confidence interval for $\Delta \text{AUROC}$ and $\Delta \text{AUPRC}$ spans zero ($p > 0.05$).
- **No statistically significant discrimination advantage is demonstrated by Hybrid DP-QNN over Classical DP-MLP.**
- **No quantum advantage exists in this benchmark.**

---

## 7. Artifact Inventory & Reproducibility Checksums

All Phase 13.3.1 artifacts have been generated and independently hashed (SHA-256):

| Artifact Relative Path | Size | SHA-256 Checksum | Verification Role |
| :--- | :---: | :--- | :--- |
| `results/phase13_3_1/privacy_utility_metrics.csv` | 1,446 B | `5a286699ca2f7ef4687a2506ac6f5f315314604e52a97a7c923058dccfaddf0f` | Primary benchmark metrics |
| `results/phase13_3_1/dp_accounting_trace.csv` | 1,514 B | `4374309eac70961b661e37b5b15e879ce5f10e6fb06f27b130709be3d39e1826` | Execution parity trace |
| `results/phase13_3_1/paired_statistical_tests.csv` | 3,112 B | `458bdba88ccdca0870ff7c7c1897c12323c7897728271a251d59c05f1dca4d14` | Clustered paired bootstrap |
| `results/phase13_3_1/execution_manifest.json` | 2,860 B | `06069c95da7614dfab47134b54519f07e3639fa27e122c11b746fa14fca4428b` | Runtime environment manifest |
| `results/phase13_3_1/checkpoints/mlp_eps_1.0.pt` | 23,245 B | `01458caef1b50c6a7f56bd5cffe770fbccd6fb0bf76dba87aead20a54fc00258` | Trained model weights |
| `results/phase13_3_1/checkpoints/mlp_eps_2.0.pt` | 23,245 B | `596508e5947efbb7d93c46457ddb00cf5e27d84ee4659e7e28212e12297bc982` | Trained model weights |
| `results/phase13_3_1/checkpoints/mlp_eps_4.0.pt` | 23,245 B | `570c9a1d8c1b0d33e597ad33cbae509a00537531632664b06f3b310dcda4af04` | Trained model weights |
| `results/phase13_3_1/checkpoints/mlp_eps_8.0.pt` | 23,245 B | `be88226f025d3c86c9c60c17cefacfff48f88ac051dc2370c5e8e1f8007133bc` | Trained model weights |
| `results/phase13_3_1/checkpoints/mlp_eps_inf.pt` | 23,245 B | `4a80af6d6f454968dd2fe5ae7c2c95d7cf6431a67e0aa14a9c265b926f2ff812` | Trained model weights |
| `results/phase13_3_1/checkpoints/qnn_eps_1.0.pt` | 22,045 B | `461b9deb9e1d150b2320b942baef1345b5302a4e362ab8200e1781db4dceef23` | Trained model weights |
| `results/phase13_3_1/checkpoints/qnn_eps_2.0.pt` | 22,045 B | `3cfa308b06f92004754074c1995daf202221ec0543225c1cb5b05711b76b3d31` | Trained model weights |
| `results/phase13_3_1/checkpoints/qnn_eps_4.0.pt` | 22,045 B | `47ce9485fa690e4572174349a45773ea809d5a028fe7feaa8189dcea2aa007ad` | Trained model weights |
| `results/phase13_3_1/checkpoints/qnn_eps_8.0.pt` | 22,045 B | `2e8b502df86b1e73e47f818d8882cddee1cb56afdd2bda1ab3862ab28a9cec83` | Trained model weights |
| `results/phase13_3_1/checkpoints/qnn_eps_inf.pt` | 22,045 B | `2993f7c1f03122a504e990b64ea0762f0f94ba33b80c671c9ccbfe1f5ff8782b` | Trained model weights |

---

## 8. Scientific Restrictions Compliance

1. **Adjacency Model:** Strictly record-level add/remove-one-record adjacency ($|\mathcal{D} \Delta \mathcal{D}'| = 1$). No patient-level differential privacy claims are made.
2. **Federated Setting:** Federated learning experiments in this repository used FedAvg without DP noise; no federated differential privacy guarantees are claimed.
3. **Quantum Claims:** In every formal statistical test ($B=1,000$), Hybrid DP-QNN failed to achieve statistically significant superior AUROC or AUPRC compared to Classical DP-MLP ($p > 0.05$). No quantum advantage is claimed.
4. **Historical Artifact Preservation:** All Phase 13.1 and Phase 13.2 files remain untouched in their respective directories (`results/phase13_1/`, `results/phase13_2/`, `research/phase13_1/`, `research/phase13_2/`, `research/phase13_3/`).

---

## 9. Final Decision

Based on the source code audit, mathematical sensitivity derivations, engine remediation, and 100% passing test execution (70/70 tests):

### **DECISION: B. DP MECHANISM CORRECTED; NEW EXPERIMENTS REQUIRED.**

**Formal Technical Rationale:**
1. **Defect Identified:** The Phase 13.2 executable code contained an actual mathematical discrepancy where gradient normalization divided by nominal constants ($32.0$ for QNN and $62.0$ for MLP) rather than the true mathematical expected batch size ($B_{\text{exp}} = qN = 31.4444...$ and $62.8889...$).
2. **Mechanism Remediated:** The mechanism in `src/privacy/formal_dp_engine.py` was corrected to enforce exact float $qN$ normalization and equipped with a `safe_collate` handler for empty Poisson draws.
3. **Requirement Met:** New experiments were required to replace the biased historical runs with mathematically pure executions. All 10 affected models were re-trained, verified, and archived in `results/phase13_3_1/`.
4. **Verification Established:** The corrected DP mechanism, analytical accounting, execution traces, checkpoints, and statistical tests are now fully consistent, mathematically sound, and independently reproducible.

Phase 14 manuscript drafting must not begin automatically. Awaiting explicit user authorization.
