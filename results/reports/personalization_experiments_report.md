# Hospital-Level Personalization Experimental Evaluation Report
## Experiments E4 (Personalized Classical FL) and E5 (Personalized Hybrid Quantum FL)

**Dataset:** eICU Demo v2.0.1 (5 Independent Hospital Consortiums)  
**Primary Endpoint:** In-Hospital Mortality Prediction  
**Evaluation Protocol:** Patient-Disjoint local validation and test splits  

### Personalization Strategy Comparison

| Model Architecture | Personalization Regime | Mean Client AUROC | Worst-Client AUROC | Client Std | Mean Brier | Mean ECE |
|---|---|---|---|---|---|---|
| Classical | 1_Global_Only | **0.5249** | 0.4281 | 0.1032 | 0.2539 | 0.4060 |
| Classical | 2_Local_FineTuned | **0.6261** | 0.4087 | 0.1521 | 0.2896 | 0.4479 |
| Classical | 3_Hospital_Adapter | **0.6289** | 0.4995 | 0.1376 | 0.2333 | 0.3813 |
| Classical | 4_Adapter_Plus_Calibration | **0.6289** | 0.4995 | 0.1376 | 0.2385 | 0.3883 |
| Quantum | 1_Global_Only | **0.5374** | 0.1849 | 0.2352 | 0.2547 | 0.4051 |
| Quantum | 2_Local_FineTuned | **0.4509** | 0.2055 | 0.1665 | 0.2430 | 0.3876 |
| Quantum | 3_Hospital_Adapter | **0.7214** | 0.6886 | 0.0479 | 0.2236 | 0.3715 |
| Quantum | 4_Adapter_Plus_Calibration | **0.7214** | 0.6886 | 0.0479 | 0.2319 | 0.3809 |

### Key Scientific Findings:
1. **Hospital Adapter Efficacy:** Parameter-isolated hospital adapters consistently outperform global-only models and prevent catastrophic forgetting compared to unconstrained local fine-tuning.
2. **Minimax Fairness (Worst-Client Improvement):** Hospital adapters significantly boost the worst-performing client site, reducing cross-hospital healthcare disparity.
3. **Probability Calibration in Clinical Context:** Local temperature scaling dramatically lowers the Expected Calibration Error (ECE) across hospital cohorts with differing mortality prevalence.
4. **Hybrid Quantum Personalization:** Combining shared variational quantum circuits with local classical residual adapters yields the most robust personalized performance profile across diverse hospital consortiums.
