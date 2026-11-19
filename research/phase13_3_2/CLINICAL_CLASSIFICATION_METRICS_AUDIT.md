# Clinical Classification Metrics & Thresholding Audit

**Phase:** 13.3.2  
**Task:** In-Depth Audit of Classification Thresholds, Discrimination, and F1 Behavior  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor & Machine Learning Specialist  

---

## 1. Executive Summary & Problem Formulation

In Phase 13.3.1, the non-private Hybrid QNN model demonstrated strong clinical discrimination:
- **AUROC:** $0.7658$
- **AUPRC:** $0.2957$ (compared to test base rate of $0.1162$)

However, the default decision threshold ($\tau = 0.50$) yielded an F1 score of **$0.0000$**, with:
- $\text{Sensitivity} = 0.0000$
- $\text{Specificity} = 1.0000$
- $\text{Accuracy} = 0.8838$

This audit was conducted to determine whether this behavior constitutes an architectural defect, training pathology, or standard classification threshold misalignment under severe class imbalance and gradient-regularized training.

---

## 2. Root Cause Analysis: The Probability Compression Phenomenon

### 2.1 Inspection of Predicted Probability Distributions
Evaluation of raw predicted risk scores $\hat{p} = \sigma(\text{logit})$ on the test set ($N=241$) reveals severe probability compression across models:

| Model Architecture | Privacy $\epsilon$ | Min Prob | Q25 Prob | Median Prob | Mean Prob | Q75 Prob | Max Prob | Std Prob |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.005351$ | $0.006371$ | $0.006950$ | $0.007449$ | $0.008039$ | $0.017770$ | $0.001713$ |
| **Classical DP-MLP** | $2.0$ | $0.005362$ | $0.006050$ | $0.006633$ | $0.007106$ | $0.007556$ | $0.018216$ | $0.001639$ |
| **Classical DP-MLP** | $4.0$ | $0.006006$ | $0.006566$ | $0.007099$ | $0.007583$ | $0.007976$ | $0.018577$ | $0.001633$ |
| **Classical DP-MLP** | $8.0$ | $0.006569$ | $0.007171$ | $0.007669$ | $0.008154$ | $0.008480$ | $0.017954$ | $0.001594$ |
| **Classical DP-MLP** | $\infty$ (Non-private) | $0.071124$ | $0.073459$ | $0.078306$ | $0.233719$ | $0.142276$ | **$0.937286$** | $0.301073$ |
| **Hybrid DP-QNN** | $1.0$ | $0.007107$ | $0.007992$ | $0.009050$ | $0.010517$ | $0.011338$ | $0.043324$ | $0.004244$ |
| **Hybrid DP-QNN** | $2.0$ | $0.006062$ | $0.007400$ | $0.008477$ | $0.009888$ | $0.010979$ | $0.029938$ | $0.003986$ |
| **Hybrid DP-QNN** | $4.0$ | $0.004827$ | $0.005967$ | $0.007424$ | $0.009439$ | $0.011151$ | $0.033150$ | $0.005038$ |
| **Hybrid DP-QNN** | $8.0$ | $0.004140$ | $0.004948$ | $0.006301$ | $0.009296$ | $0.011167$ | $0.037381$ | $0.006428$ |
| **Hybrid DP-QNN** | $\infty$ (Non-private) | $0.006269$ | $0.006302$ | $0.006398$ | $0.013490$ | $0.007338$ | **$0.136655$** | $0.021915$ |

### 2.2 Mathematical Explanation
1. **Logit Calibration under Imbalance:** In unweighted binary cross-entropy loss, the optimal Bayes logit under low prevalence ($\pi \approx 0.08$) is heavily negative ($\ln(\pi / (1-\pi)) \approx -2.44$).
2. **Quantum Layer Output Bounds:** In the hybrid QNN, the PQC measures Pauli-Z expectation values $\langle Z_i \rangle \in [-1, 1]$. The subsequent linear classification head regularizes logits towards conservative negative values. The maximum predicted probability produced by the non-private QNN on the test set is **$0.136655$**, which is strictly below $\tau = 0.50$.
3. **Differential Privacy Regularization:** For all DP models ($\epsilon \in \{1, 2, 4, 8\}$), per-sample gradient clipping ($C = 1.0$) and calibrated Gaussian noise severely constrain parameter magnitudes, compressing all probabilities below $0.045$.
4. **Conclusion:** Because $\max(\hat{p}) = 0.1367 < 0.50$, setting an arbitrary threshold at $\tau = 0.50$ predicts zero positive encounters ($\text{Pred Pos} = 0$). This produces $TP = 0, FP = 0, TN = 213, FN = 28$, resulting in $F1 = 0.0000$ and trivial accuracy of $213 / 241 = 0.8838$.
5. **Finding:** This is **not a software defect or model corruption**. The model successfully ranks positive cases ahead of negative cases (AUROC $= 0.7658$, AUPRC $= 0.2957$), but requires threshold selection matched to the empirical risk scale.

---

## 3. Comprehensive Evaluation: Default Threshold ($\tau = 0.50$)

Table 1 summarizes all 10 trained models evaluated at the unadjusted standard threshold $\tau = 0.50$:

| Model Architecture | $\epsilon$ | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced Acc | AUROC | AUPRC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5857 | 0.1772 |
| **Classical DP-MLP** | $2.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5778 | 0.1898 |
| **Classical DP-MLP** | $4.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6016 | 0.1945 |
| **Classical DP-MLP** | $8.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6281 | 0.2006 |
| **Classical DP-MLP** | $\infty$ | 11 | 32 | 181 | 17 | 0.3929 | 0.8498 | 0.2558 | 0.3099 | 0.7967 | 0.6213 | 0.7166 | 0.2722 |
| **Hybrid DP-QNN** | $1.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.5904 | 0.1382 |
| **Hybrid DP-QNN** | $2.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6184 | 0.1522 |
| **Hybrid DP-QNN** | $4.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6410 | 0.1675 |
| **Hybrid DP-QNN** | $8.0$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.6648 | 0.1817 |
| **Hybrid DP-QNN** | $\infty$ | 0 | 0 | 213 | 28 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.8838 | 0.5000 | 0.7658 | 0.2957 |

---

## 4. Validation-Locked Threshold Optimization

### 4.1 Methodology (Zero Test Leakage)
To establish clinical utility without test-label leakage:
1. Optimal decision threshold $\tau^*$ is determined **strictly on the independent validation set** ($N=234, 24\text{ deaths}$) by maximizing Youden's J statistic:
   $$J(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1$$
2. The selected threshold $\tau^*$ is permanently locked.
3. The locked threshold is evaluated **exactly once** on the frozen test set ($N=241, 28\text{ deaths}$).

### 4.2 Out-of-Sample Performance at Validation-Locked Thresholds

Table 2 presents test set performance when evaluated at the pre-locked validation thresholds:

| Model Architecture | $\epsilon$ | Locked $\tau^*$ | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1 | Balanced Acc | Pred Pos Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical DP-MLP** | $1.0$ | $0.007880$ | 14 | 55 | 158 | 14 | 0.5000 | 0.7418 | 0.2029 | 0.2887 | 0.6209 | 28.63% |
| **Classical DP-MLP** | $2.0$ | $0.007110$ | 15 | 71 | 142 | 13 | 0.5357 | 0.6667 | 0.1744 | 0.2632 | 0.6012 | 35.68% |
| **Classical DP-MLP** | $4.0$ | $0.007849$ | 14 | 50 | 163 | 14 | 0.5000 | 0.7653 | 0.2188 | 0.3043 | 0.6326 | 26.56% |
| **Classical DP-MLP** | $8.0$ | $0.008374$ | 14 | 51 | 162 | 14 | 0.5000 | 0.7606 | 0.2154 | 0.3011 | 0.6303 | 26.97% |
| **Classical DP-MLP** | $\infty$ | $0.080205$ | 21 | 82 | 131 | 7 | 0.7500 | 0.6150 | 0.2039 | 0.3206 | 0.6825 | 42.74% |
| **Hybrid DP-QNN** | $1.0$ | $0.009810$ | 12 | 85 | 128 | 16 | 0.4286 | 0.6009 | 0.1237 | 0.1920 | 0.5148 | 40.25% |
| **Hybrid DP-QNN** | $2.0$ | $0.010198$ | 15 | 63 | 150 | 13 | 0.5357 | 0.7042 | 0.1923 | 0.2830 | 0.6200 | 32.37% |
| **Hybrid DP-QNN** | $4.0$ | $0.007749$ | 20 | 97 | 116 | 8 | 0.7143 | 0.5446 | 0.1709 | 0.2759 | 0.6294 | 48.55% |
| **Hybrid DP-QNN** | $8.0$ | $0.011079$ | 12 | 50 | 163 | 16 | 0.4286 | 0.7653 | 0.1935 | 0.2667 | 0.5969 | 25.73% |
| **Hybrid DP-QNN** | $\infty$ | $0.006521$ | 21 | 75 | 138 | 7 | **0.7500** | **0.6479** | **0.2188** | **0.3387** | **0.6989** | 39.83% |

---

## 5. Statistical & Clinical Implications

1. **Restoration of Clinical Classification:** Applying the pre-locked Youden threshold restores robust binary decision performance:
   - For non-private QNN, sensitivity rises from $0.0000$ to **$75.00\%$** ($21/28$ deaths captured), with **$64.79\%$** specificity and balanced accuracy of **$69.89\%$**.
   - F1 score rises from $0.0000$ to **$0.3387$** (substantially exceeding the random prevalence baseline of $0.1162$).
2. **Threshold Sensitivity in Production:** Because raw probabilities are compressed, any real-world deployment must incorporate either:
   - Isotonic regression / Platt probability recalibration.
   - Clinical operating threshold selection keyed directly to acceptable false positive rates (e.g., triage protocols).
3. **Manuscript Reporting Directive:** The IEEE manuscript must explicitly report both default threshold metrics ($\tau = 0.50$) and validation-locked operating metrics ($\tau^*$) to ensure full scientific transparency.
