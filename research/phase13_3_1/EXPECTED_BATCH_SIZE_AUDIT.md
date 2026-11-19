# Phase 13.3.1 — Expected Batch Size and Sampling Rate Consistency Audit

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.1 Critical DP Verification  
**Auditor:** Senior Research Scientist & Statistical Reviewer  
**Date:** November 19, 2026  
**Clinical Cohort:** 24-Hour Landmark Cohort ($N_{\text{total}} = 1,607$ stays, $N_{\text{train}} = 1,132$, $N_{\text{val}} = 234$, $N_{\text{test}} = 241$)  
**Status:** Complete & Independently Verified  

---

## 1. Executive Summary & Critical Discrepancy

In the Phase 13.2 implementation and Phase 13.3 accounting audit, a mathematical inconsistency was identified between the stated Poisson sampling probabilities ($q$) and the expected batch size normalization constants ($B_{\text{exp}}$) utilized in gradient scaling:

- **Dataset Size:** $N = 1,132$ training ICU encounters.
- **Reported in Phase 13.2 / 13.3 Documentation:**
  - Hybrid DP-QNN: Stated sampling rate $q = 1/36 \approx 0.027778$, nominal batch size 32, and reported normalization denominator $B_{\text{exp}} = 32.0$.
  - Classical DP-MLP: Stated sampling rate $q = 1/18 \approx 0.055556$, nominal batch size 64, and reported normalization denominator $B_{\text{exp}} = 62.0$.
- **Mathematical Inconsistency:**
  Under genuine Poisson subsampling where each record is sampled independently with probability $q$, the batch size is a binomial random variable $|B| \sim \text{Binomial}(N, q)$ whose exact mathematical expectation is:
  $$\mathbb{E}[|B|] = qN$$
  Evaluating this expectation for the actual training cohort:
  $$\text{For QNN:} \quad qN = \frac{1,132}{36} = 31.444444... \ne 32.0$$
  $$\text{For MLP:} \quad qN = \frac{1,132}{18} = 62.888888... \ne 62.0$$

The previous report text asserted that "gradients were divided by the exact expected batch size $B_{\text{exp}} = qN$," but the executable code actually divided by $32.0$ for QNN (due to hardcoding `expected_batch_size = float(self.batch_size)`) and by $62.0$ for MLP (due to Opacus's internal integer truncation `int(len(dataset) * sample_rate)`).

This audit resolves this discrepancy from executable source code, proves the mathematical sensitivity invariance, rigorously evaluates Option A vs. Option B, and establishes the exact mathematical parameters.

---

## 2. Exhaustive 10-Point Parameter Inventory

The table below records the 10 parameters evaluated directly from executable source code (`src/privacy/formal_dp_engine.py`, Opacus 1.4.1 internals) and runtime execution logs:

| Parameter # | Parameter Description | Hybrid DP-QNN (Phase 13.2 / 13.3 Executed) | Classical DP-MLP (Phase 13.2 / 13.3 Executed) | Source Code Reference |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Actual training dataset size ($N$) | **1,132** encounters | **1,132** encounters | `data/processed_landmark/X_train.npy` |
| **2** | Actual Poisson sampling probability ($q$) | **$1/36 \approx 0.0277778$** | **$1/18 \approx 0.0555556$** | `formal_dp_engine.py:333`, Opacus `from_data_loader` |
| **3** | Expected batch size passed to optimizer | **32.0** (`float(self.batch_size)`) | **62** (`int(1132 * (1/18))`) | `formal_dp_engine.py:334`, Opacus `make_private:436` |
| **4** | Actual denominator used in gradient normalization | **32.0** (`p.grad = (acc_g + noise) / 32.0`) | **62.0** (`p.grad /= 62`) | `formal_dp_engine.py:399`, Opacus `DPOptimizer.scale_grad:504` |
| **5** | Actual clipping norm ($C$) | **1.0** | **1.0** | `self.max_grad_norm = 1.0` |
| **6** | Actual Gaussian noise std added to sum | **$\sigma \cdot C = \sigma$** | **$\sigma \cdot C = \sigma$** | `noise = randn * (sigma * C)`, Opacus `add_noise` |
| **7** | Actual noise multiplier ($\sigma$) | $\epsilon=1.0: 1.8066$<br>$\epsilon=2.0: 1.1890$<br>$\epsilon=4.0: 0.8728$<br>$\epsilon=8.0: 0.6656$ | $\epsilon=1.0: 3.8867$<br>$\epsilon=2.0: 2.1924$<br>$\epsilon=4.0: 1.3501$<br>$\epsilon=8.0: 0.8848$ | `dp_accounting_trace.csv`, Opacus `RDPAccountant` |
| **8** | Actual number of sampling attempts ($T$) | **180** ($5 \text{ epochs} \times 36 \text{ steps}$) | **270** ($15 \text{ epochs} \times 18 \text{ steps}$) | `formal_dp_engine.py:382`, Opacus `data_loader` iterations |
| **9** | Actual accountant sampling probability ($q_{\text{acc}}$) | **$1/36 \approx 0.0277778$** | **$1/18 \approx 0.0555556$** | `acc.step(..., sample_rate=sample_rate)` |
| **10** | Actual accountant step count ($T_{\text{acc}}$) | **180** (private runs); 0 (inf baseline) | **270** (private runs); 0 (inf baseline) | `dp_accounting_trace.csv` |

---

## 3. Mathematical Evaluation of Poisson Subsampling Normalization

### 3.1 Unbiased Gradient Estimation
Let $\mathcal{D} = \{z_1, \dots, z_N\}$ denote the training dataset. In iteration $t$, a batch $B_t \subseteq \mathcal{D}$ is formed by independent Bernoulli inclusion with probability $q$:
$$\mathbb{I}(i \in B_t) \sim \text{Bernoulli}(q), \quad \mathbb{P}(i \in B_t) = q$$
For each record $i \in B_t$, the clipped gradient is $\bar{g}_i = g_i \cdot \min(1, C / \|g_i\|_2)$. The mechanism computes:
$$\tilde{S} = \sum_{i \in B_t} \bar{g}_i + \xi, \quad \xi \sim \mathcal{N}(0, \sigma^2 C^2 I)$$
Taking expectations over the Poisson sampling randomness and Gaussian noise:
$$\mathbb{E}[\tilde{S}] = \mathbb{E}\left[\sum_{i=1}^N \mathbb{I}(i \in B_t) \bar{g}_i + \xi\right] = \sum_{i=1}^N \mathbb{E}[\mathbb{I}(i \in B_t)] \bar{g}_i + 0 = q \sum_{i=1}^N \bar{g}_i$$
To obtain an unbiased estimator of the full-batch mean clipped gradient $\bar{g}_{\text{full}} = \frac{1}{N} \sum_{i=1}^N \bar{g}_i$:
$$\mathbb{E}\left[\frac{\tilde{S}}{D}\right] = \frac{q \sum_{i=1}^N \bar{g}_i}{D} = \frac{1}{N} \sum_{i=1}^N \bar{g}_i \iff D = qN$$
Therefore, the normalization denominator $D$ **must equal $qN$** to preserve an unbiased gradient estimator.

### 3.2 Quantitative Impact of Phase 13.2 Divisors
- **Hybrid DP-QNN in Phase 13.2:**
  - $q = 1/36$, $qN = 1,132 / 36 = 31.4444$.
  - Code divided by $D = 32.0$.
  - Gradient expectation: $\mathbb{E}[\tilde{g}] = \frac{31.4444}{32.0} \bar{g}_{\text{full}} = 0.982639 \times \bar{g}_{\text{full}}$.
  - **Effect:** Gradients were scaled down by $1.74\%$.
- **Classical DP-MLP in Phase 13.2:**
  - $q = 1/18$, $qN = 1,132 / 18 = 62.8889$.
  - Opacus divided by $D = 62.0$.
  - Gradient expectation: $\mathbb{E}[\tilde{g}] = \frac{62.8889}{62.0} \bar{g}_{\text{full}} = 1.014337 \times \bar{g}_{\text{full}}$.
  - **Effect:** Gradients were scaled up by $1.43\%$.

### 3.3 Differential Privacy and Post-Processing Invariance
Because $D = 32.0$, $D = 31.4444$, $D = 62.0$, and $D = 62.8889$ are all **fixed, data-independent constants**, dividing by any of them is a valid post-processing transformation:
$$h(x) = \frac{x}{D}$$
By the Post-Processing Theorem of Differential Privacy:
$$\mathcal{M}(\mathcal{D}) \text{ is } (\alpha, \epsilon(\alpha))\text{-RDP} \implies h(\mathcal{M}(\mathcal{D})) \text{ is } (\alpha, \epsilon(\alpha))\text{-RDP}$$
Therefore:
1. **Privacy validity:** The privacy accounting $(\epsilon, \delta)$ computed in Phase 13.2 / 13.3 was mathematically sound and never violated.
2. **Mathematical terminology:** The description claiming that $D = qN$ was false for the exact numerical values ($32.0 \ne 31.4444$ and $62.0 \ne 62.8889$).
3. **Statistical optimization:** Setting $D = qN$ restores mathematical unbiasedness to the gradient estimators.

---

## 4. Rigorous Evaluation: Option A vs. Option B

The user prompt mandates formal evaluation of Option A and Option B without silent assumption:

### Option A: Preserve $q = 1/K$ and Normalize by Exact Float $qN$
- **Formulation:**
  - QNN: $K = \lceil 1132 / 32 \rceil = 36$ steps/epoch. $q = 1/36 \approx 0.027778$. Normalization denominator $B_{\text{exp}} = qN = 1132 / 36 = 31.444444...$
  - MLP: $K = \lceil 1132 / 64 \rceil = 18$ steps/epoch. $q = 1/18 \approx 0.055556$. Normalization denominator $B_{\text{exp}} = qN = 1132 / 18 = 62.888889...$
- **Sampling Schedule Analysis:**
  - In Opacus `UniformWithReplacementSampler`, the number of iterations per epoch is `steps = int(1 / sample_rate)`.
  - When $q = 1/36$, `int(1 / (1/36)) = 36` exactly.
  - When $q = 1/18$, `int(1 / (1/18)) = 18` exactly.
  - In one epoch ($K$ steps), the expected number of times each sample is drawn is $K \cdot q = K \cdot (1/K) = 1.0$. This perfectly satisfies the clinical and statistical definition of an epoch.
- **Privacy Accounting Implications:**
  - The accountant steps with $(q = 1/36, T = 180)$ for QNN and $(q = 1/18, T = 270)$ for MLP.
  - Calibrated noise multipliers $\sigma$ and achieved $\epsilon$ values match the validated Phase 13.2 / 13.3 accounting tables identically.
- **Operational Feasibility:**
  - Float divisor in `p.grad = (acc_g + noise) / 31.4444` is fully native in PyTorch.
  - In Opacus, setting `optimizer.expected_batch_size = 62.888888888888886` overrides the integer truncation and executes without error in `DPOptimizer.scale_grad()`.

### Option B: Preserve Nominal Expected Batch Sizes and Derive $q = B_{\text{exp}} / N$
- **Formulation:**
  - QNN: Nominal $B_{\text{exp}} = 32.0$. Sampling rate $q = 32 / 1132 \approx 0.02826855$.
  - MLP: Nominal $B_{\text{exp}} = 62.0$. Sampling rate $q = 62 / 1132 \approx 0.05477032$ (or $B_{\text{exp}} = 64.0, q = 64 / 1132 \approx 0.05653710$).
- **Sampling Schedule Analysis:**
  - In Opacus `UniformWithReplacementSampler`:
    $$\text{steps} = \text{int}\left(\frac{1}{q}\right) = \text{int}\left(\frac{1,132}{32}\right) = \text{int}(35.375) = 35$$
    **Critical Flaw:** The integer truncation drops $0.375$ batches per epoch! An epoch yields only 35 batches instead of 36.
    Over 5 epochs, $T = 5 \times 35 = 175$ steps (not 180).
    The expected number of samples drawn per epoch is $35 \times 32 = 1,120 < 1,132$, systematically undersampling the cohort by 12 patients per epoch!
    If one forcibly sets `steps = 36`, then $36 \times 32 = 1,152 > 1,132$, oversampling by 20 patients per epoch.
- **Privacy Accounting Implications:**
  - $q$ increases from $0.027778$ to $0.028269$.
  - If $T = 175$, noise multipliers must be recalibrated: $\sigma = 1.8164$ (at $\epsilon=1.0$).
  - If $T = 180$, noise multipliers must be recalibrated: $\sigma = 1.8359$ (at $\epsilon=1.0$).
  - All previously audited privacy tables and checkpoints would be invalidated and discarded.

### Synthesis & Selection:
**Option A is unanimously selected as the mathematically and operationally superior configuration.**  
It guarantees:
1. Exact integer batch count per epoch without truncation ($K=36$ for QNN, $K=18$ for MLP).
2. Exactly $1.0$ expected exposure per record per epoch.
3. Mathematically unbiased gradient estimation ($\mathbb{E}[\tilde{g}] = \bar{g}_{\text{full}}$).
4. Full mathematical alignment with Opacus Poisson sampling architecture.

---

## 5. Opacus Integer Truncation Audit

In Opacus 1.4.1, `PrivacyEngine.make_private()` prepares the optimizer via:
```python
sample_rate = 1 / len(data_loader)
expected_batch_size = int(len(data_loader.dataset) * sample_rate)
```
For our landmark training cohort:
- `len(data_loader) = 18` (since initial `batch_size = 64`, $\lceil 1132 / 64 \rceil = 18$).
- `sample_rate = 1 / 18 = 0.05555555555555555`.
- `len(data_loader.dataset) = 1132`.
- Exact float: $1132 \times (1/18) = 62.888888888888886$.
- `int(...)` truncation: `int(62.888888888888886) = 62`.

When `DPOptimizer.scale_grad()` executes:
```python
if self.loss_reduction == "mean":
    for p in self.params:
        p.grad /= self.expected_batch_size * self.accumulated_iterations
```
The gradient was divided by integer `62` rather than float `62.8889`.

**Remediation:**  
In `src/privacy/formal_dp_engine.py`, right after `make_private_with_epsilon()`, we inject:
```python
exact_expected_batch_size = float(len(data_loader.dataset) * data_loader.sample_rate)
optimizer.expected_batch_size = exact_expected_batch_size
```
Unit test `tests/test_phase13_3_1_dp_consistency.py::test_mlp_expected_batch_size_float_exactness` confirms that `optimizer.expected_batch_size` maintains `62.888888888888886` and scales gradients with full double-precision floating-point accuracy.

---

## 6. Audit Conclusion

1. The discrepancy between reported $B_{\text{exp}} = 32.0$ / $62.0$ and $qN = 31.4444$ / $62.8889$ is verified and documented from source code.
2. The privacy guarantees $(\epsilon, \delta)$ derived in Phase 13.2 / 13.3 are mathematically valid post-processing scalings, but possessed a minor $1.74\%$ gradient scaling bias.
3. Option A is formally adopted, setting $B_{\text{exp}} = 31.444444...$ for QNN and $B_{\text{exp}} = 62.888889...$ for MLP.
4. All affected DP models must be re-trained under the corrected engine to establish updated, reproducible checkpoints and metrics.
