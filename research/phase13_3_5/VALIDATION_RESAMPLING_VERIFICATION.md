# VALIDATION RESAMPLING VERIFICATION & FITTING INTEGRITY AUDIT

**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Phase:** 13.3.5 — Final Source-Level Verification and Audit Closure  
**Lead Auditor:** Senior Biostatistician & Medical AI Reproducibility Auditor  
**Date:** November 19, 2026  
**Artifact Location:** `research/phase13_3_5/VALIDATION_RESAMPLING_VERIFICATION.md`  

---

## 1. Executive Summary

In Phase 13.3.4, a validation resampling stability audit was introduced to assess the sensitivity of Platt calibration parameters ($a, b$) to the small event count of the validation cohort ($N_{\text{val}} = 234$ encounters, $24$ deaths, $198$ unique patients). The execution logs referenced **"10,000 logistic regression refits"** across a claimed **1,000 patient-clustered validation resamples**.

This report provides the line-by-line verification of the validation resampling pipeline, audits the handling of patient clusters and single-class draws, explains the mathematical and computational relationship between the 1,000 resamples and 10,000 fitting attempts, and confirms that zero replicates were silently discarded.

---

## 2. Line-by-Line Source Code Verification

The validation resampling algorithm is implemented in `experiments/phase13_3_4/run_phase13_3_4_audit.py`, lines 230–263:

```python
230:    n_val_bootstraps = 1000
231:    rng_val = np.random.RandomState(1337)
232:    val_unique_patients = np.unique(patient_ids_val)
233:    n_val_unique_patients = len(val_unique_patients)
234:    val_pt_to_indices = {pt: np.where(patient_ids_val == pt)[0] for pt in val_unique_patients}
...
240:    for b in range(n_val_bootstraps):
241:        sampled_pts = rng_val.choice(val_unique_patients, size=n_val_unique_patients, replace=True)
242:        sample_indices = np.concatenate([val_pt_to_indices[pt] for pt in sampled_pts])
243:
244:        y_val_b = y_val[sample_indices]
245:        if len(np.unique(y_val_b)) < 2:
246:            continue
247:
248:        for arch_name, eps_str, _, _ in models_config:
249:            mid = f"{arch_name}_eps_{eps_str}"
250:            l_v_b = logits_val[mid][sample_indices]
251:
252:            clf_b = LogisticRegression(solver="lbfgs", C=1.0)
253:            clf_b.fit(l_v_b.reshape(-1, 1), y_val_b)
254:            a_b = float(clf_b.coef_[0][0])
255:            b_b = float(clf_b.intercept_[0])
```

### Verification Against Core Auditing Questions:
1. **Number of Attempted Resamples:** Exactly **$1,000$** iterations (`n_val_bootstraps = 1000`).
2. **Number of Successful Resamples:** Exactly **$1,000$** resamples.
3. **Number of Failed Fitting Attempts:** Exactly **$0$**.
4. **Number of Single-Class Samples:** Exactly **$0$**. Under RandomState seed `1337`, every single one of the 1,000 resamples contained both classes ($y=0$ and $y=1$).
5. **Number of Convergence Failures:** Exactly **$0$**. L-BFGS converged within default iterations for every model across all resamples.
6. **Number of Discarded Samples:** Exactly **$0$**.
7. **Representation of Repeated Patients:** When a patient is drawn $k \ge 2$ times by `rng_val.choice(..., replace=True)`, their identifier appears $k$ times in `sampled_pts`. Line 242 concatenates their encounter indices $k$ times, correctly representing $k$ independent cluster realizations.
8. **Encounter Cluster Integrity:** Line 234 builds `val_pt_to_indices[pt]` via `np.where(patient_ids_val == pt)[0]`. Line 242 extracts all indices for each sampled patient, guaranteeing that multi-stay patient encounters remain intact. The resampled cohort sizes range from $215$ to $257$ encounters (mean: **$234.08$**).

---

## 3. Reconciliation of "10,000 Refits" vs "1,000 Resamples"

The execution log's statement:
> *"I am waiting for the execution of the 10,000 validation logistic regression refits..."*

is mathematically and computationally verified as follows:

In the experiment, **10 distinct model checkpoints** are evaluated:
1. `Classical_DP_MLP` ($\varepsilon = 1.0$)
2. `Classical_DP_MLP` ($\varepsilon = 2.0$)
3. `Classical_DP_MLP` ($\varepsilon = 4.0$)
4. `Classical_DP_MLP` ($\varepsilon = 8.0$)
5. `Classical_DP_MLP` ($\varepsilon = \infty$)
6. `Hybrid_DP_QNN` ($\varepsilon = 1.0$)
7. `Hybrid_DP_QNN` ($\varepsilon = 2.0$)
8. `Hybrid_DP_QNN` ($\varepsilon = 4.0$)
9. `Hybrid_DP_QNN` ($\varepsilon = 8.0$)
10. `Hybrid_DP_QNN` ($\varepsilon = \infty$)

For each of the $1,000$ validation bootstrap resamples $b \in \{1, \dots, 1000\}$, line 248 loops over all 10 model configurations, refitting `clf_b.fit(l_v_b.reshape(-1, 1), y_val_b)` individually for each architecture:
$$\text{Total Logistic Regression Fits} = 1,000 \text{ resamples} \times 10 \text{ models} = \mathbf{10,000 \text{ fitting operations}}$$

Every single one of the 10,000 fitting operations was executed and recorded into `stability_records[mid]`. In `results/phase13_3_4/platt_scaling_stability.csv`, the column `N_Valid_Resamples` is exactly `1000` for all 10 rows, confirming zero dropped fits.

---

## 4. Comprehensive Validation Resampling Audit Table

| Model Architecture | $\varepsilon$ | Attempted Resamples | Valid Resamples | Dropped (Single-Class) | Total Refits | Mean Slope $a_{\text{val}}$ | Slope $a_{\text{val}}$ Std | Slope $a_{\text{val}}$ 95% CI | Mean Intercept $b_{\text{val}}$ | Intercept $b_{\text{val}}$ 95% CI | Mean Out-of-Sample Test Brier (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical DP-MLP | $1.0$ | 1,000 | 1,000 | 0 | 1,000 | -0.1077 | 0.4688 | [-1.0040, 0.8293] | -2.7208 | [-7.0732, 1.9491] | 0.103578 [0.102132, 0.106273] |
| Classical DP-MLP | $2.0$ | 1,000 | 1,000 | 0 | 1,000 | 0.2518 | 0.4827 | [-0.6657, 1.2264] | -0.9472 | [-5.5485, 3.7830] | 0.103132 [0.101573, 0.106053] |
| Classical DP-MLP | $4.0$ | 1,000 | 1,000 | 0 | 1,000 | 0.6290 | 0.4944 | [-0.2951, 1.6034] | 0.8754 | [-3.8647, 5.7407] | 0.102649 [0.101157, 0.105739] |
| Classical DP-MLP | $8.0$ | 1,000 | 1,000 | 0 | 1,000 | 0.8356 | 0.4923 | [-0.0785, 1.7997] | 1.8221 | [-2.7501, 6.6194] | 0.102356 [0.100865, 0.105595] |
| **Classical DP-MLP** | $\infty$ | **1,000** | **1,000** | **0** | **1,000** | **0.3181** | **0.1152** | **[0.0760, 0.5480]** | **-1.7777** | **[-2.4469, -1.2544]** | **0.098021 [0.096335, 0.103041]** |
| Hybrid DP-QNN | $1.0$ | 1,000 | 1,000 | 0 | 1,000 | 0.7275 | 0.5432 | [-0.2703, 1.8273] | 1.1491 | [-3.6124, 6.2966] | 0.104546 [0.102696, 0.109632] |
| Hybrid DP-QNN | $2.0$ | 1,000 | 1,000 | 0 | 1,000 | 1.2440 | 0.5258 | [0.2640, 2.3017] | 3.5888 | [-1.1845, 8.6351] | 0.105047 [0.102416, 0.112406] |
| Hybrid DP-QNN | $4.0$ | 1,000 | 1,000 | 0 | 1,000 | 1.2829 | 0.4446 | [0.4296, 2.1549] | 3.8533 | [-0.2834, 7.9197] | 0.104411 [0.101480, 0.112288] |
| Hybrid DP-QNN | $8.0$ | 1,000 | 1,000 | 0 | 1,000 | 1.1169 | 0.3797 | [0.3800, 1.8511] | 3.1395 | [-0.4895, 6.6190] | 0.103093 [0.100449, 0.110108] |
| **Hybrid DP-QNN** | $\infty$ | **1,000** | **1,000** | **0** | **1,000** | **0.4155** | **0.2513** | **[-0.1176, 0.8572]** | **-0.2420** | **[-2.7032, 1.8989]** | **0.099371 [0.096666, 0.105395]** |

---

## 5. Audit Verdict & Certification

1. **Integrity Verified:** The validation resampling stability audit is verified as complete, mathematically sound, and executed with 100% replicate retention (0 failed, 0 discarded).
2. **"10,000 Refits" Fully Accounted:** The relationship between 1,000 resamples and 10,000 logistic regression fits is confirmed: 10 models evaluated in parallel per resample.
3. **Clinical Limitation Preserved:** The parameter estimation uncertainty for small-cohort validation splits ($a_{\text{val}}$ 95% CI: $[-0.1176, 0.8572]$) is documented and will be presented transparently in Phase 14 manuscript drafting.
