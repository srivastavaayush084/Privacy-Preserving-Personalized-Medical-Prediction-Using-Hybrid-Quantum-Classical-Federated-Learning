# Phase 13.3.1 — DP Mechanism Correction and Verification Report

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.1 Critical DP Verification  
**Auditor:** Senior Research Scientist & Cybersecurity Reviewer  
**Date:** November 19, 2026  
**Clinical Cohort:** 24-Hour Landmark Cohort ($N_{\text{train}} = 1,132$, $N_{\text{val}} = 234$, $N_{\text{test}} = 241$)  
**Status:** Remediated, Implemented & Fully Verified  

---

## 1. Mathematical Formulation of the Corrected DP Mechanism

The implemented differential privacy mechanism for both Hybrid DP-QNN and Classical DP-MLP is the **Poisson-Subsampled Gaussian Mechanism with Record-Level L2 Gradient Clipping and Unbiased Normalization**.

### 1.1 Step-by-Step Mechanism Execution
At each training iteration $t \in \{1, \dots, T\}$:

1. **Poisson Subsampling:**
   Each record $i \in \{1, \dots, N\}$ in the training dataset $\mathcal{D}$ is included in batch $B_t$ independently with fixed probability $q \in (0, 1]$:
   $$\mathbb{P}(i \in B_t) = q, \quad |B_t| \sim \text{Binomial}(N, q)$$
   The mathematical expectation of the batch size is:
   $$B_{\text{exp}} = \mathbb{E}[|B_t|] = qN$$

2. **Per-Sample Gradient Computation & L2 Clipping:**
   For each sample $i \in B_t$, the per-sample loss gradient $g_i(\theta_t) = \nabla_\theta \ell(\theta_t; x_i, y_i)$ is computed.
   Each gradient is individually projected onto the $L_2$ ball of radius $C = 1.0$:
   $$\bar{g}_i(\theta_t) = g_i(\theta_t) \cdot \min\left(1, \frac{C}{\|g_i(\theta_t)\|_2 + 10^{-8}}\right)$$
   This guarantees $\|\bar{g}_i(\theta_t)\|_2 \le C = 1.0$ for all $i \in B_t$.

3. **Noise Addition to the Clipped Gradient Sum:**
   The clipped gradients are summed, and spherical Gaussian noise calibrated to the $L_2$ sensitivity is added directly to the sum:
   $$\tilde{S}_t = \sum_{i \in B_t} \bar{g}_i(\theta_t) + \xi_t, \quad \xi_t \sim \mathcal{N}\left(0, \sigma^2 C^2 I_d\right)$$
   where $d$ is the number of model parameters, and $\sigma$ is the calibrated noise multiplier.

4. **Unbiased Normalization by Exact Expected Batch Size:**
   The noisy sum $\tilde{S}_t$ is normalized by the fixed, data-independent expected batch size $B_{\text{exp}} = qN$:
   $$\tilde{g}_t = \frac{\tilde{S}_t}{B_{\text{exp}}} = \frac{1}{qN} \left(\sum_{i \in B_t} \bar{g}_i(\theta_t) + \xi_t\right)$$

5. **Parameter Update & Accountant Step:**
   The model parameters are updated using the normalized gradient $\tilde{g}_t$ via Adam:
   $$\theta_{t+1} = \text{AdamStep}\left(\theta_t, \tilde{g}_t\right)$$
   Simultaneously, the Renyi Differential Privacy (RDP) accountant advances by recording one evaluation of the Subsampled Gaussian Mechanism at sampling rate $q$ and noise multiplier $\sigma$.

---

## 2. Sensitivity Analysis and Post-Processing Invariance

### 2.1 Record-Level Adjacency
We define privacy under the standard **Add/Remove-One-Record Adjacency Relation**:
$$\mathcal{D} \sim \mathcal{D}' \iff |\mathcal{D} \Delta \mathcal{D}'| = 1$$
Let $\mathcal{D}' = \mathcal{D} \cup \{z^*\}$. Under Poisson subsampling:
- If $z^* \notin B_t$ (probability $1 - q$), the batch draws are identical: $S(\mathcal{D}) = S(\mathcal{D}')$.
- If $z^* \in B_t$ (probability $q$), the sum difference is:
  $$\|S(\mathcal{D}') - S(\mathcal{D})\|_2 = \|\bar{g}(z^*)\|_2 \le C = 1.0$$
Thus, the global $L_2$ sensitivity of the un-subsampled sum query is:
$$\Delta_2 = \max_{\mathcal{D} \sim \mathcal{D}'} \|S(\mathcal{D}) - S(\mathcal{D}')\|_2 = C = 1.0$$

### 2.2 Proof of Post-Processing Invariance under $B_{\text{exp}}$ Normalization
**Theorem (Post-Processing of RDP):**  
Let $\mathcal{M}: \mathcal{D} \to \mathbb{R}^d$ be an $(\alpha, \epsilon(\alpha))$-RDP mechanism. Let $f: \mathbb{R}^d \to \mathbb{R}^d$ be any data-independent mapping. Then $f \circ \mathcal{M}$ satisfies $(\alpha, \epsilon(\alpha))$-RDP.

*Proof:*  
The Renyi divergence between the output distributions $\mathcal{M}(\mathcal{D})$ and $\mathcal{M}(\mathcal{D}')$ is:
$$D_\alpha(\mathcal{M}(\mathcal{D}) \,\|\, \mathcal{M}(\mathcal{D}')) \le \epsilon(\alpha)$$
By the data processing inequality for Renyi divergence:
$$D_\alpha(f(\mathcal{M}(\mathcal{D})) \,\|\, f(\mathcal{M}(\mathcal{D}'))) \le D_\alpha(\mathcal{M}(\mathcal{D}) \,\|\, \mathcal{M}(\mathcal{D}')) \le \epsilon(\alpha)$$
For the linear normalization function $f(x) = \frac{x}{B_{\text{exp}}}$, since $B_{\text{exp}} = qN$ is computed solely from the fixed public dataset size $N$ and the algorithm sampling rate $q$, $f$ does not depend on any private data.  
Therefore, dividing by $B_{\text{exp}} = 31.4444...$ (QNN) or $B_{\text{exp}} = 62.8889...$ (MLP) preserves the exact RDP curve and $(\epsilon, \delta)$ guarantee without any degradation. $\blacksquare$

---

## 3. Empty Batch Handling & Accountant Synchronization

Under Poisson subsampling with $q = 1/36$ ($N = 1,132$), the probability of drawing an empty batch ($|B_t| = 0$) in any step is:
$$\mathbb{P}(|B_t| = 0) = (1 - q)^N = \left(1 - \frac{1}{36}\right)^{1132} \approx 1.63 \times 10^{-14}$$
Although extremely rare, the DP guarantee mathematically assumes that the mechanism produces an output on every sampling attempt, even when $|B_t| = 0$.

### Verified Code Implementation:
In `src/privacy/formal_dp_engine.py` (lines 383–415):
```python
B = len(bx)
optimizer.zero_grad()
accum_grads = [torch.zeros_like(p) for p in params]

if B > 0:
    non_empty_batches += 1
    # Compute and accumulate clipped per-sample gradients
    for i in range(B):
        ...
        for acc_g, g in zip(accum_grads, sample_grads):
            acc_g.add_(g * clip_factor)
else:
    empty_batches += 1

if sigma > 0.0:
    for acc_g, p in zip(accum_grads, params):
        noise = torch.randn_like(acc_g) * (sigma * self.max_grad_norm)
        # When B == 0, acc_g is 0, so p.grad = noise / expected_batch_size
        p.grad = (acc_g + noise) / expected_batch_size
    if acc is not None:
        acc.step(noise_multiplier=sigma, sample_rate=sample_rate)
        accountant_steps += 1
    optimizer.step()
    optimizer_updates += 1
```

**Verification Highlights:**
1. When $B = 0$, `accum_grads` remains all zeros.
2. Gaussian noise is generated and assigned: `p.grad = noise / B_exp`.
3. The accountant advances: `acc.step(noise_multiplier=sigma, sample_rate=sample_rate)`.
4. The optimizer updates model weights: `optimizer.step()`.
5. Strict trace equality is verified:
   $$T_{\text{sampler}} = T_{\text{optimizer}} = T_{\text{accountant}} = 180 \quad (\text{QNN})$$
   $$T_{\text{sampler}} = T_{\text{optimizer}} = T_{\text{accountant}} = 270 \quad (\text{MLP})$$

---

## 4. Code Remediation Details

### 4.1 Hybrid DP-QNN (`FormalPerSampleDPQNNTrainer`)
**File:** `src/privacy/formal_dp_engine.py`  
**Before (Phase 13.2):**
```python
expected_batch_size = float(self.batch_size) # Hardcoded to 32.0!
```
**After (Phase 13.3.1):**
```python
# Exact expected batch size under Poisson subsampling with inclusion rate q: E[|B|] = q * N
# Aligns with Option A mathematical expectation (replaces nominal batch_size divisor)
expected_batch_size = float(N * sample_rate) # Exact float: 1132 / 36 = 31.444444444444443
```

### 4.2 Classical DP-MLP (`FormalOpacusMLPTrainer`)
**File:** `src/privacy/formal_dp_engine.py`  
**Before (Phase 13.2):**
```python
model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(...)
# Opacus defaulted optimizer.expected_batch_size to int(1132 * (1/18)) = 62!
```
**After (Phase 13.3.1):**
```python
model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(...)
# Correct Opacus integer truncation (int(len*q) = 62) to exact float q*N = 62.888888888888886
exact_expected_batch_size = float(len(data_loader.dataset) * data_loader.sample_rate)
optimizer.expected_batch_size = exact_expected_batch_size
```

---

## 5. Verification Checklist

| Verification Item | Requirement | Verification Method | Status |
| :--- | :--- | :--- | :--- |
| **Individual Gradient Clipping** | $\|\bar{g}_i\|_2 \le 1.0$ | Per-sample $L_2$ norm thresholding with factor $\min(1, 1.0 / \|g_i\|)$ | **VERIFIED** |
| **Noise Addition Location** | Noise added to unnormalized sum | Code inspection: `(acc_g + noise) / B_exp` | **VERIFIED** |
| **Noise Standard Deviation** | $\text{std} = \sigma \cdot C$ | `randn * (sigma * max_grad_norm)` with $C=1.0$ | **VERIFIED** |
| **Normalization Denominator** | Fixed, data-independent float $qN$ | QNN: $31.4444...$, MLP: $62.8889...$ | **VERIFIED** |
| **Empty Batch Progress** | Pure noise update + accountant step | Validated by `tests/test_phase13_2_dp.py::test_empty_batch_accounting_and_trace_equality` | **VERIFIED** |
| **Sampler / Step Parity** | $T_{\text{sampler}} = T_{\text{opt}} = T_{\text{acc}}$ | Validated by execution trace assertions | **VERIFIED** |
| **No Secondary Scaling** | No extraneous gradient scaling | Code inspection: Adam optimizer receives scaled gradient directly | **VERIFIED** |
