# PHASE 13.2 AUDIT: QUANTUM DIFFERENTIAL PRIVACY GRADIENT NORMALIZATION

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.2 — Final DP Mechanism Verification & Statistical Audit  
**Author:** Independent Statistical & Research Audit Team  
**Date:** November 2026  
**Status:** Remediated & Mathematically Verified  

---

## 1. Executive Summary & Audit Mandate

In Phase 13.1, an independent audit of the differential privacy implementation for the Hybrid Quantum-Classical Neural Network (`DPHybridQNNNet` via `FormalPerSampleDPQNNTrainer`) identified a critical theoretical question regarding gradient normalization:

$$\tilde{g}_{\text{realized}} = \frac{\sum_{i \in B} \text{clip}(g_i, C) + \mathcal{N}(0, \sigma^2 C^2 I)}{|B|}$$

Under Poisson subsampling, the realized batch size $|B|$ is a binomially distributed random variable. The Phase 13.2 audit mandate required investigating whether dividing by the random realized batch size $|B|$ strictly preserves the formal differential privacy guarantee $(\epsilon, \delta)$ provided by the Rényi Differential Privacy (RDP) accountant, or whether it violates the Post-Processing Theorem.

**Audit Finding:** Dividing the noisy gradient sum by the realized batch size $|B|$ introduces a data-dependent denominator that violates the standard post-processing formulation of the Gaussian mechanism under add/remove-one-record adjacency. Standard Opacus DP-SGD proofs require normalization by the **fixed, data-independent expected batch size** $B_{\text{exp}} = qN = \text{batch\_size}$.

**Remediation Executed:**
1. Modified `src/privacy/formal_dp_engine.py` to divide strictly by fixed expected batch size $B_{\text{exp}} = \text{self.batch\_size}$ (or $qN$).
2. Added Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ directly to the clipped gradient sum prior to normalization.
3. Verified that $L_2$ sensitivity $\Delta_2 = C = 1.0$ holds under add/remove-one-record adjacency.
4. Validated full parity with native Opacus `DPOptimizer.scale_grad()`.

---

## 2. Mathematical Formulation of Poisson Subsampled DP-SGD

### 2.1 Adjacency Notion
We adopt the standard **add/remove-one-record adjacency** relation: two datasets $\mathcal{D}, \mathcal{D}'$ are adjacent ($\mathcal{D} \sim \mathcal{D}'$) if one can be formed from the other by adding or removing exactly one record:

$$|\mathcal{D} \Delta \mathcal{D}'| = 1 \quad \iff \quad |\mathcal{D}| = N, \ |\mathcal{D}'| = N \pm 1$$

### 2.2 Poisson Subsampling Mechanism
Given dataset $\mathcal{D} = \{x_1, \dots, x_N\}$, each record $i \in \{1, \dots, N\}$ is independently sampled into mini-batch $B$ with subsampling probability $q \in (0, 1)$:

$$z_i \sim \text{Bernoulli}(q), \quad B = \{i \in \mathcal{D} : z_i = 1\}$$

The realized batch size $|B| = \sum_{i=1}^N z_i$ follows a Binomial distribution:

$$|B| \sim \text{Binomial}(N, q), \quad \mathbb{E}[|B|] = qN = B_{\text{exp}}$$

### 2.3 Per-Sample Gradient Clipping & Sensitivity
For each sample $i \in B$, the loss gradient with respect to model parameters $\theta$ is $g_i = \nabla_\theta \ell(f(x_i; \theta), y_i)$. Each per-sample gradient is clipped to maximum $L_2$ norm $C$:

$$\bar{g}_i = \text{clip}(g_i, C) = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right), \quad \|\bar{g}_i\|_2 \le C$$

The sum of clipped gradients is $S(B) = \sum_{i \in B} \bar{g}_i$. Consider adjacent datasets $\mathcal{D}$ and $\mathcal{D}' = \mathcal{D} \cup \{x^*\}$. In any step where the subsampler runs, the extra sample $x^*$ is included with probability $q$. If included ($z^* = 1$), the difference in the unnormalized sum is:

$$\|S(B \cup \{x^*\}) - S(B)\|_2 = \|\bar{g}^*\|_2 \le C$$

Thus, the global $L_2$ sensitivity of the sum under add/remove-one-record adjacency is:

$$\Delta_2(S) = \max_{\mathcal{D} \sim \mathcal{D}'} \|S(B) - S(B')\|_2 = C$$

### 2.4 The Calibrated Gaussian Mechanism
The subsampled Gaussian mechanism computes:

$$\mathcal{M}(B) = \sum_{i \in B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)$$

The Rényi Differential Privacy (RDP) accountant bounds the privacy curve $\epsilon(\alpha)$ of $\mathcal{M}$ over $T$ composition steps as a function of $(q, \sigma, T)$ using the analytical Fourier/numerical RDP formulation of Poisson subsampled Gaussian mechanisms (Mironov et al., 2019; Wang et al., 2019).

---

## 3. The Data-Dependent Normalization Defect

### 3.1 The Post-Processing Theorem
The foundational Post-Processing Theorem of Differential Privacy states:

> **Theorem (Post-Processing):** If $\mathcal{M}: \mathcal{X}^* \to \mathcal{Y}$ is $(\epsilon, \delta)$-differentially private, and $f: \mathcal{Y} \to \mathcal{Z}$ is an arbitrary **data-independent** randomized or deterministic mapping, then $f \circ \mathcal{M}: \mathcal{X}^* \to \mathcal{Z}$ is $(\epsilon, \delta)$-differentially private.

### 3.2 Violation Under Division by Realized Batch Size $|B|$
In the pre-remediation QNN implementation:

$$\tilde{g} = \frac{\mathcal{M}(B)}{|B|} = \frac{\sum_{i \in B} \bar{g}_i + \xi}{|B|}, \quad \xi \sim \mathcal{N}(0, \sigma^2 C^2 I)$$

Here, the scaling function is $f_B(y) = \frac{y}{|B|}$. However:
1. $|B| = \sum_{i \in \mathcal{D}} z_i$. The value of $|B|$ is an explicit function of the database $\mathcal{D}$ and the realization of $z^*$.
2. For $\mathcal{D}' = \mathcal{D} \cup \{x^*\}$, if $x^*$ is sampled, then $|B'| = |B| + 1$.
3. Therefore, $f_B$ is **not** a data-independent post-processing function. It is a data-dependent function that depends on the private presence of $x^*$.
4. Dividing by $|B|$ scales both the signal $\sum \bar{g}_i$ and the added noise $\xi$ by $\frac{1}{|B|}$. When $|B|$ is small (e.g., $|B| = 2$), the noise variance injected into the model update is $\frac{\sigma^2 C^2}{4}$, whereas when $|B|$ is large (e.g., $|B| = 40$), the noise variance is $\frac{\sigma^2 C^2}{1600}$.
5. Because an adversary observing the scale of update fluctuations could potentially infer information about $|B|$ (and thus about whether specific records were sampled), division by $|B|$ invalidates the closed-form RDP accountant proofs.

---

## 4. Remediation: Fixed Expected Batch Size Normalization

To strictly satisfy the Post-Processing Theorem, the normalization factor must be a predetermined, data-independent scalar.

### 4.1 Specification
We define the normalized gradient update as:

$$\tilde{g} = \frac{\sum_{i \in B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)}{B_{\text{exp}}}$$

where:

$$B_{\text{exp}} = q \cdot N = \text{self.batch\_size}$$

is fixed before training begins, depending only on the published hyperparameters $q$ and $N$, with zero dependence on the realized sample draws.

### 4.2 Proof of Privacy Preservation
Let $f(y) = \frac{y}{B_{\text{exp}}}$. Since $B_{\text{exp}} > 0$ is a fixed constant independent of the dataset $\mathcal{D}$, $f$ is a deterministic, data-independent function. By the Post-Processing Theorem:

$$\mathcal{M}_{\text{DP-SGD}}(\mathcal{D}) = f(\mathcal{M}(\mathcal{D})) = \frac{\mathcal{M}(\mathcal{D})}{B_{\text{exp}}}$$

has exactly the same $(\epsilon, \delta)$ differential privacy guarantee as $\mathcal{M}(\mathcal{D})$. The RDP bound computed by Opacus `RDPAccountant` applies unconditionally.

### 4.3 Opacus Architectural Alignment
Inspection of PyTorch Opacus (`opacus.optimizers.DPOptimizer.scale_grad`) confirms that Opacus itself uses this exact principle:

```python
# opacus/optimizers/optimizer.py
def scale_grad(self):
    if self.loss_reduction == "mean":
        for p in self.params:
            p.grad /= self.expected_batch_size * self.accumulated_iterations
```

Opacus docstring explicitly notes:
> *"expected_batch_size: batch_size used for averaging gradients. When using Poisson sampling averaging denominator can't be inferred from the actual batch size."*

Our remediated `FormalPerSampleDPQNNTrainer` now strictly mirrors Opacus's reference design.

---

## 5. Verification Matrix & Test Trace

| Audit Requirement | Pre-Remediation State | Remediated State | Test Reference |
|---|---|---|---|
| **Normalization Denominator** | Variable $|B| = \text{len}(bx)$ | Fixed $B_{\text{exp}} = \text{self.batch\_size}$ | `test_qnn_gradient_normalization_uses_expected_batch_size` |
| **Noise Addition Timing** | After clipping, divided by $|B|$ | Added to sum before $B_{\text{exp}}$ scaling | `test_qnn_gradient_normalization_uses_expected_batch_size` |
| **Adjacency Notion** | Add/Remove record | Add/Remove record ($|\mathcal{D} \Delta \mathcal{D}'| = 1$) | `test_record_level_sensitivity_under_add_remove` |
| **Sensitivity $\Delta_2$** | $C = 1.0$ | $C = 1.0$ strictly verified | `test_record_level_sensitivity_under_add_remove` |
| **Accountant Alignment** | Analytical RDP | Analytical RDP via Opacus `RDPAccountant` | `test_noise_multiplier_calibration_and_monotonicity` |

---

## 6. Conclusion

The gradient normalization defect in the quantum differential privacy engine has been fully resolved. By replacing the random realized batch size $|B|$ with the fixed data-independent expected batch size $B_{\text{exp}}$, the implementation strictly adheres to the Post-Processing Theorem and guarantees that the claimed $(\epsilon, \delta)$ privacy bounds are mathematically defensible and publication-ready.
