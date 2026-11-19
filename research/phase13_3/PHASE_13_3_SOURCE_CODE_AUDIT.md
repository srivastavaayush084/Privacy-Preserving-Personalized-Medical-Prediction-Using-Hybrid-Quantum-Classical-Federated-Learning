# PHASE 13.3 AUDIT: SOURCE CODE VERIFICATION & TRAINING SCHEDULE AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3 — Final Source-Code Verification, DP Accounting Reproducibility and Experiment Consistency Audit  
**Author:** Independent Research Software & Audit Specialist  
**Date:** November 2026  
**Status:** Audited & Reconciled  

---

## 1. Executive Summary & Audit Mandate

The Phase 13.3 audit mandate required an independent investigation into the source code, training schedules, and iteration counts of the Differential Privacy Stochastic Gradient Descent (DP-SGD) training runs conducted in Phase 13.2. Specifically, the audit investigated the relationship between:
- Training cohort size: $N = 1,132$ clinical encounters.
- Batch sizes: $B_{\text{exp}} = 32$ for Hybrid QNN, $B_{\text{exp}} = 64$ for Classical MLP.
- The reported QNN iteration count (180 steps) vs. the earlier Phase 13.1 proposal (10 epochs / 360 steps).
- The exact scheduling convention implemented in `src/privacy/formal_dp_engine.py` and `experiments/phase13_2/run_phase13_2_experiments.py`.

**Core Finding:** The reported 180 iterations for the Hybrid DP-QNN in Phase 13.2 corresponds strictly and accurately to **5 epochs** under the implementation's scheduling convention ($\lceil 1,132 / 32 \rceil = 36$ steps/epoch $\times 5$ epochs = 180 steps). The 360 iterations mentioned in the Phase 13.1 audit document was a theoretical calibration plan for 10 epochs that was modified to 5 epochs in the executable code to accommodate classical simulation constraints on CPU. The code and recorded artifacts (`dp_accounting_trace.csv`) are 100% self-consistent with the 5-epoch schedule.

---

## 2. Implementation Scheduling Convention

### 2.1 Code Inspection: `src/privacy/formal_dp_engine.py`
In `src/privacy/formal_dp_engine.py` (lines 330–355):

```python
N = len(X_train)
steps_per_epoch = int(math.ceil(N / self.batch_size))
total_steps = steps_per_epoch * self.epochs
sample_rate = 1.0 / steps_per_epoch
expected_batch_size = float(self.batch_size)
```

1. **Steps per Epoch:** `steps_per_epoch = int(math.ceil(1132 / 32)) = 36`.
2. **Subsampling Rate:** `sample_rate = 1.0 / 36 = 0.027777777777777776`.
   - *Note on sampling convention:* In Opacus and standard Poisson subsampling libraries, setting `sample_rate = 1 / steps_per_epoch` ensures that `len(UniformWithReplacementSampler) == steps_per_epoch`. If $q$ were set to $32 / 1132 \approx 0.028268$, then $\lfloor 1/q \rfloor = 35$, which would cause the sampler to yield 35 batches per epoch while calibration assumed 36, creating a 1-step per epoch accountant mismatch. Setting $q = 1 / 36$ guarantees exact structural alignment between the data loader loop and the RDP accountant.
3. **Total Steps:** For `epochs = 5`, `total_steps = 36 * 5 = 180`.

### 2.2 Classical MLP Scheduling Convention
For Classical MLP via `FormalOpacusMLPTrainer` (`batch_size = 64`):
1. **Steps per Epoch:** `steps_per_epoch = int(math.ceil(1132 / 64)) = 18`.
2. **Subsampling Rate:** `sample_rate = 1.0 / 18 = 0.05555555555555555`.
3. **Total Steps:** For `epochs = 15`, `total_steps = 18 * 15 = 270`.

---

## 3. Discrepancy Reconciliation: 180 vs. 360 QNN Iterations

### 3.1 Historical Evolution Across Phases
- **Phase 12:** Evaluated Hybrid QNN on a stratified subset ($N = 400$) for 5 epochs with batch size 32.
- **Phase 13.1:** When transitioning to the locked landmark cohort ($N = 1,132$), the analytical audit report (`research/PHASE_13_1_DP_ACCOUNTING_AUDIT.md`) drafted an aspirational table specifying 10 epochs ($10 \times 36 = 360$ steps). However, in the experiment script (`experiments/landmark/run_landmark_experiments.py` line 624), the trainer was instantiated with `epochs = 5`.
- **Phase 13.2:** The master experiment runner (`experiments/phase13_2/run_phase13_2_experiments.py` line 164) executed with `epochs = 5`, generating:
  $$\text{Sampler Iterations} = 5 \times 36 = 180$$
  Noise multipliers were calibrated for $T = 180$, and the accountant was stepped exactly 180 times.

### 3.2 Verification of Executed Parameters

The table below reconciles the executed parameters across all models in Phase 13.2 (`results/phase13_2/dp_accounting_trace.csv`):

| Model | Target $\epsilon$ | Target $\delta$ | Actual Epochs | $q$ (Subsampling Rate) | $B_{\text{exp}}$ | Sampler Iterations | Optimizer Updates | Accountant Steps | Calibrated $\sigma$ | Achieved $\epsilon$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **Classical DP-MLP** | 1.0 | $10^{-5}$ | 15 | 0.055556 | 62.0 | 270 | 270 | 270 | 3.8867 | 0.9953 |
| **Classical DP-MLP** | 2.0 | $10^{-5}$ | 15 | 0.055556 | 62.0 | 270 | 270 | 270 | 2.1924 | 1.9954 |
| **Classical DP-MLP** | 4.0 | $10^{-5}$ | 15 | 0.055556 | 62.0 | 270 | 270 | 270 | 1.3501 | 3.9985 |
| **Classical DP-MLP** | 8.0 | $10^{-5}$ | 15 | 0.055556 | 62.0 | 270 | 270 | 270 | 0.9277 | 7.9977 |
| **Classical DP-MLP** | $\infty$ | N/A | 15 | 0.055556 | 62.0 | 270 | 270 | 0 | 0.0000 | $\infty$ |
| **Hybrid DP-QNN** | 1.0 | $10^{-5}$ | 5 | 0.027778 | 32.0 | 180 | 180 | 180 | 1.8066 | 0.9940 |
| **Hybrid DP-QNN** | 2.0 | $10^{-5}$ | 5 | 0.027778 | 32.0 | 180 | 180 | 180 | 1.1890 | 1.9959 |
| **Hybrid DP-QNN** | 4.0 | $10^{-5}$ | 5 | 0.027778 | 32.0 | 180 | 180 | 180 | 0.8728 | 3.9949 |
| **Hybrid DP-QNN** | 8.0 | $10^{-5}$ | 5 | 0.027778 | 32.0 | 180 | 180 | 180 | 0.6656 | 7.9944 |
| **Hybrid DP-QNN** | $\infty$ | N/A | 5 | 0.027778 | 32.0 | 180 | 180 | 0 | 0.0000 | $\infty$ |

---

## 4. Source-Code Integrity Findings

1. **No Hardcoded Metrics:** Metrics in `results/phase13_2/privacy_utility_metrics.csv` are produced dynamically by `evaluate_model()` and `compute_empirical_privacy_leakage()`.
2. **Correct Optimizer Attachment:** In `FormalPerSampleDPQNNTrainer`, `optimizer.step()` and `acc.step()` are both called exactly inside the batch iteration loop, guaranteeing that optimizer updates and privacy budget accounting advance in lockstep.
3. **Trace Equality Confirmed:** Across all private training runs, $\text{Sampler Iterations} == \text{Optimizer Updates} == \text{Accountant Steps}$.
4. **Documentation Correction Mandate:** Any remaining textual references to "10 epochs" or "360 steps" for QNN in earlier Phase 13.1 text must be designated as an superseded preliminary plan, and the manuscript must report the audited **5 epochs / 180 steps** schedule.
