# Prevalence-Only Calibration Baseline & Model Calibration Audit

**Phase:** 13.3.3  
**Task:** Rigorous Probabilistic Calibration Assessment Against Constant Prevalence Baseline  
**Date:** November 19, 2026  
**Auditor:** Independent Senior Research Auditor, Biostatistician & Clinical Informatics Reviewer  

---

## 1. Executive Summary

A critical question in clinical machine learning is whether complex neural network models provide probabilistic risk assessments that improve upon a trivial **prevalence-only constant baseline**. In medical prognosis, an uncalibrated model may exhibit good discriminative ranking (AUROC $> 0.75$) while generating systematically distorted probabilities that produce severe squared-error penalties and mislead clinical risk stratification.

This audit independently establishes that:
1. The **prevalence-only baseline** for the frozen test cohort ($N=241, 28\text{ deaths}$) with constant predicted risk $p_i = 28/241 \approx 0.116183$ achieves:
   - **Brier Score:** **$0.102684$**
   - **Log Loss:** **$0.359249$**
   - **Expected Calibration Error (ECE, 10 bins):** **$0.000000$**
2. In its **raw uncalibrated state**, the non-private Hybrid QNN achieves a Brier score of **$0.110555$**, which is **worse (higher squared error) than the prevalence-only baseline** ($0.1106$ vs. $0.1027$, $\Delta = +0.007871$).
3. The previous report's description of QNN's Brier score ($0.1106$) as "superior" rested entirely on a relative comparison against an overconfident Classical MLP ($0.1621$). **Relative superiority over a miscalibrated model does not constitute good absolute calibration.**
4. When recalibrated out-of-sample using **validation-fit Platt scaling** (strictly zero test-data leakage), the QNN's Brier score drops to **$0.097960$** (Log Loss: **$0.339444$**), successfully outperforming the prevalence baseline while preserving its discriminative ranking (AUROC $= 0.7658$).

---

## 2. Mathematical Definition of the Prevalence Baseline

Let the test cohort outcomes be $y_i \in \{0, 1\}$ for $i \in \{1, \dots, N\}$, where $N = 241$ and $\sum_{i=1}^N y_i = 28$.  
The sample prevalence is $\bar{y} = \frac{28}{241} \approx 0.11618257$.

### 2.1 Baseline Brier Score
For a constant prediction $\hat{p}_i = \bar{y}$, the Brier score equals the sample variance of the binary Bernoulli outcome:
$$\text{Brier}(\bar{y}) = \frac{1}{N} \sum_{i=1}^N (y_i - \bar{y})^2 = \bar{y}(1 - \bar{y}) = \left(\frac{28}{241}\right) \left(\frac{213}{241}\right) = \frac{5,964}{58,081} \approx \mathbf{0.102684}$$

### 2.2 Baseline Log Loss
$$\text{LogLoss}(\bar{y}) = - \left[ \bar{y} \ln(\bar{y}) + (1 - \bar{y}) \ln(1 - \bar{y}) \right] \approx \mathbf{0.359249}$$

### 2.3 Baseline Calibration Error (ECE)
Because every prediction is identical ($\hat{p}_i = 0.1162$), all 241 samples fall into a single bin ($[0.1, 0.2]$). The average predicted risk in this bin is $0.1162$ and the observed event rate is $28/241 = 0.1162$. Thus:
$$\text{ECE} = \frac{241}{241} |0.116183 - 0.116183| = \mathbf{0.000000}$$

---

## 3. Comprehensive Calibration Audit Across All Models

Table 1 details Brier score, Log Loss, ECE, calibration intercept, and calibration slope for the prevalence baseline, all 10 raw models, and all 10 validation-calibrated models:

| Model Architecture | $\epsilon$ | Variant | Brier Score | Log Loss | ECE (10 bins) | Calib. Intercept ($\alpha$) | Calib. Slope ($\beta$) | Mean Pred. Prob. | Observed Prev. | $\Delta$ Brier vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prevalence Baseline** | N/A | Constant | **$0.102684$** | **$0.359249$** | **$0.000000$** | $0.0000$ | $1.0000$ | $0.116183$ | $0.116183$ | $0.000000$ |
| **Classical DP-MLP** | $1.0$ | Raw | $0.114408$ | $0.572211$ | $0.108733$ | $+2.8687$ | $1.3359$ | $0.007449$ | $0.116183$ | $+0.011724$ |
| **Classical DP-MLP** | $1.0$ | Platt | $0.102982$ | $0.360839$ | $0.013557$ | $+0.1393$ | $-13.4414$ | $0.102626$ | $0.116183$ | $+0.000298$ |
| **Classical DP-MLP** | $2.0$ | Raw | $0.114466$ | $0.576255$ | $0.109077$ | $+2.9162$ | $1.5676$ | $0.007106$ | $0.116183$ | $+0.011782$ |
| **Classical DP-MLP** | $2.0$ | Platt | $0.102535$ | $0.358474$ | $0.018403$ | $+0.1423$ | $5.8719$ | $0.102353$ | $0.116183$ | **$-0.000150$** |
| **Classical DP-MLP** | $4.0$ | Raw | $0.114350$ | $0.568109$ | $0.108600$ | $+2.8500$ | $1.9183$ | $0.007583$ | $0.116183$ | $+0.011665$ |
| **Classical DP-MLP** | $4.0$ | Platt | $0.102020$ | $0.355886$ | $0.025802$ | $+0.1440$ | $2.9430$ | $0.102217$ | $0.116183$ | **$-0.000664$** |
| **Classical DP-MLP** | $8.0$ | Raw | $0.114214$ | $0.559596$ | $0.108029$ | $+2.7760$ | $2.2831$ | $0.008154$ | $0.116183$ | $+0.011530$ |
| **Classical DP-MLP** | $8.0$ | Platt | $0.101726$ | $0.354368$ | $0.024905$ | $+0.1422$ | $2.6586$ | $0.102387$ | $0.116183$ | **$-0.000958$** |
| **Classical DP-MLP** | $\infty$ | Raw | **$0.162129$** | **$0.518477$** | **$0.123177$** | $-1.7558$ | $0.3377$ | $0.233719$ | $0.116183$ | **$+0.059444$** |
| **Classical DP-MLP** | $\infty$ | Platt | **$0.096438$** | **$0.335121$** | **$0.016929$** | $+0.0766$ | $1.0494$ | $0.108936$ | $0.116183$ | **$-0.006246$** |
| **Hybrid DP-QNN** | $1.0$ | Raw | $0.113879$ | $0.542331$ | $0.105665$ | $+2.5309$ | $0.2400$ | $0.010517$ | $0.116183$ | $+0.011195$ |
| **Hybrid DP-QNN** | $1.0$ | Platt | $0.103318$ | $0.361085$ | $0.018664$ | $+0.1166$ | $0.3145$ | $0.104810$ | $0.116183$ | $+0.000634$ |
| **Hybrid DP-QNN** | $2.0$ | Raw | $0.113892$ | $0.543881$ | $0.106295$ | $+2.5938$ | $0.6830$ | $0.009888$ | $0.116183$ | $+0.011208$ |
| **Hybrid DP-QNN** | $2.0$ | Platt | $0.103668$ | $0.359158$ | $0.039777$ | $+0.0900$ | $0.5357$ | $0.107470$ | $0.116183$ | $+0.000984$ |
| **Hybrid DP-QNN** | $4.0$ | Raw | $0.113806$ | $0.543830$ | $0.106743$ | $+2.6535$ | $0.8519$ | $0.009439$ | $0.116183$ | $+0.011122$ |
| **Hybrid DP-QNN** | $4.0$ | Platt | $0.102973$ | $0.353747$ | $0.044713$ | $+0.0646$ | $0.6519$ | $0.110000$ | $0.116183$ | $+0.000289$ |
| **Hybrid DP-QNN** | $8.0$ | Raw | $0.113632$ | $0.541903$ | $0.106886$ | $+2.6890$ | $0.8404$ | $0.009296$ | $0.116183$ | $+0.010948$ |
| **Hybrid DP-QNN** | $8.0$ | Platt | $0.101607$ | $0.347965$ | $0.028608$ | $+0.0489$ | $0.7404$ | $0.111523$ | $0.116183$ | **$-0.001078$** |
| **Hybrid DP-QNN** | $\infty$ | Raw | **$0.110555$** | **$0.502657$** | **$0.102692$** | **$+2.4826$** | **$0.7526$** | **$0.013490$** | $0.116183$ | **$+0.007871$** |
| **Hybrid DP-QNN** | $\infty$ | Platt | **$0.097960$** | **$0.339444$** | **$0.044539$** | **$+0.1081$** | **$1.7096$** | **$0.105722$** | $0.116183$ | **$-0.004724$** |

---

## 4. Deep Forensic Analysis of the Calibration Failure

### 4.1 Why Does Raw QNN Fail Against the Prevalence Baseline?
1. **Severe Under-Prediction of Risk:**  
   The mean predicted probability of the non-private QNN across the 241 test encounters is only **$0.013490$** ($1.35\%$), whereas the true empirical mortality rate is **$0.116183$** ($11.62\%$).  
   The calibration intercept (calibration-in-the-large, with slope fixed at 1) is **$\alpha = +2.4826$**. This large positive value formally proves that raw logits are severely negative, systematically deflating estimated death probabilities.
2. **Asymmetric Squared-Error Penalty:**  
   For the 28 patients who died ($y_i = 1$), the raw QNN predicts probabilities around $0.01$ to $0.08$. The squared error $(1 - \hat{p}_i)^2$ is near $0.90 - 0.98$.  
   In contrast, the prevalence baseline assigns $\hat{p}_i = 0.1162$, yielding $(1 - 0.1162)^2 = 0.7811$.  
   Thus, on the positive cases, the raw QNN accumulates substantially worse squared error than the naive baseline, outweighing its small gains on the negative cases.
3. **The Danger of Misleading Relative Benchmarking:**  
   The Classical MLP ($\epsilon=\infty$) suffered from the opposite pathology: extreme overconfidence ($\bar{p} = 0.2337$, $\max \hat{p} = 0.9373$), producing massive false alarms and an inflated Brier score of **$0.1621$**.  
   Describing QNN's Brier score of $0.1106$ as "superior" merely because it beat an overconfident MLP concealed the fact that **neither raw model outperformed a simple baseline assignment of $11.62\%$ risk**.

---

## 5. Efficacy of Validation-Fit Out-of-Sample Recalibration

When Platt scaling is trained strictly on the independent validation split ($N=234, 24\text{ deaths}$) and applied out-of-sample to test predictions:
1. **Calibration-in-the-Large Restored:** The calibration intercept for QNN shifts from **$+2.4826$** to **$+0.1081$** (near the ideal $0.0$). Mean predicted risk rises to **$0.105722$**, closely matching the test prevalence of $0.1162$.
2. **Brier Score Genuine Improvement:** QNN Brier score drops from **$0.110555$** to **$0.097960$**, representing a true **$-0.004724$ absolute improvement over the prevalence baseline** ($0.0980 < 0.1027$).
3. **Log Loss Genuine Improvement:** QNN Log Loss drops from **$0.502657$** to **$0.339444$**, beating the baseline Log Loss of $0.359249$.
4. **Preservation of Discrimination:** Because Platt scaling is a strictly monotonic affine transformation of logits ($z_{\text{cal}} = a z + b$ with $a > 0$), the model's discriminative capacity is mathematically identical (AUROC $= 0.7658$).

---

## 6. Audit Conclusion & Publication Mandate

1. **Formal Retraction of Absolute Calibration Claims:** Historical statements claiming the raw QNN possessed "good calibration" based on Brier $= 0.1106$ are scientifically invalid and are formally retracted.
2. **Publication Guideline:** The IEEE manuscript must explicitly state that **raw neural network predictions (both QNN and MLP) fail to beat the prevalence-only baseline in Brier score and Log Loss**, and that post-hoc validation recalibration (Platt scaling or isotonic regression) is strictly required to translate discriminative capacity into well-calibrated clinical probabilities.
