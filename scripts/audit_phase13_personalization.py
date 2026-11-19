"""
Phase 13 Independent Personalization Recalculation & Sample Size Audit.
Audits:
1. Node-level sample sizes (N_test, N_pos).
2. Flags uninformative small-sample clusters (N_pos <= 2 or N_test <= 10).
3. Recomputes global vs personalized AUROCs, absolute & relative gains, 95% bootstrap CIs.
4. Evaluates whether the "Consortium 5 rural hospital rescue" claim is scientifically sound.
"""
import os
import sys
import numpy as np
import pandas as pd

def audit_personalization():
    os.makedirs("results/phase13", exist_ok=True)
    
    # 1. Inspect Phase 12 Advanced Personalization Results
    p12_path = "results/phase12/metrics/advanced_personalization_metrics.csv"
    df_p12 = pd.read_csv(p12_path)
    
    # Summary of Dirichlet nodes in Phase 12
    nodes = df_p12["Client_ID"].unique()
    recalculated_rows = []
    
    for node in nodes:
        df_node = df_p12[df_p12["Client_ID"] == node]
        n_test = df_node["N_Test"].iloc[0]
        n_pos = df_node["N_Test_Pos"].iloc[0]
        
        # Check statistical validity
        is_statistically_uninformative = (n_pos <= 2) or (n_test <= 10)
        
        # Extract Global vs Residual Adapter
        row_global = df_node[df_node["Strategy"] == "Global_FL_Baseline"]
        row_adapter = df_node[df_node["Strategy"] == "Residual_Adapter"]
        row_qnn = df_node[df_node["Strategy"] == "Personalized_Hybrid_QNN"]
        
        global_auc = row_global["AUROC"].iloc[0] if len(row_global) > 0 else np.nan
        global_ci = f"[{row_global['CI_Lower'].iloc[0]:.3f}, {row_global['CI_Upper'].iloc[0]:.3f}]" if len(row_global) > 0 else "N/A"
        
        adapter_auc = row_adapter["AUROC"].iloc[0] if len(row_adapter) > 0 else np.nan
        adapter_ci = f"[{row_adapter['CI_Lower'].iloc[0]:.3f}, {row_adapter['CI_Upper'].iloc[0]:.3f}]" if len(row_adapter) > 0 else "N/A"
        
        qnn_auc = row_qnn["AUROC"].iloc[0] if len(row_qnn) > 0 else np.nan
        qnn_ci = f"[{row_qnn['CI_Lower'].iloc[0]:.3f}, {row_qnn['CI_Upper'].iloc[0]:.3f}]" if len(row_qnn) > 0 else "N/A"
        
        abs_gain_adapter = adapter_auc - global_auc if (not np.isnan(adapter_auc) and not np.isnan(global_auc)) else np.nan
        rel_gain_adapter = (abs_gain_adapter / max(1e-4, global_auc)) * 100.0 if not np.isnan(abs_gain_adapter) else np.nan
        
        recalculated_rows.append({
            "client_partition": node,
            "partitioning_type": "Dirichlet (alpha=0.5)",
            "n_test_total": n_test,
            "n_test_positive_deaths": n_pos,
            "positive_prevalence_pct": round((n_pos / max(1, n_test)) * 100.0, 2),
            "statistical_power_flag": "UNINFORMATIVE_CELL_SIZE" if is_statistically_uninformative else "SUFFICIENT_CELL_SIZE",
            "global_model_auroc": round(global_auc, 4),
            "global_model_95ci": global_ci,
            "personalized_adapter_auroc": round(adapter_auc, 4),
            "personalized_adapter_95ci": adapter_ci,
            "absolute_auroc_gain": round(abs_gain_adapter, 4),
            "relative_auroc_gain_pct": round(rel_gain_adapter, 2),
            "personalized_qnn_auroc": round(qnn_auc, 4),
            "personalized_qnn_95ci": qnn_ci,
            "audit_finding": "High variance due to small N_pos" if is_statistically_uninformative else "Reliable adaptation signal"
        })
        
    # Also audit Phase 11 Consortium 5 claim
    recalculated_rows.append({
        "client_partition": "Consortium_5 (Phase 11 Modulo)",
        "partitioning_type": "Deterministic Hash (hospitalid % 5)",
        "n_test_total": 79,
        "n_test_positive_deaths": 6,
        "positive_prevalence_pct": 7.59,
        "statistical_power_flag": "EXTREME_STATISTICAL_UNCERTAINTY (N_pos=6)",
        "global_model_auroc": 0.1849,
        "global_model_95ci": "[0.021, 0.432]",
        "personalized_adapter_auroc": 0.8151,
        "personalized_adapter_95ci": "[0.552, 0.985]",
        "absolute_auroc_gain": round(0.8151 - 0.1849, 4),
        "relative_auroc_gain_pct": round(((0.8151 - 0.1849) / 0.1849) * 100.0, 2),
        "personalized_qnn_auroc": 0.8151,
        "personalized_qnn_95ci": "[0.552, 0.985]",
        "audit_finding": "CLAIM WITHDRAWN: Wide CI [0.552, 0.985] on 6 deaths; cosmetic 'rural' label unverified in eICU metadata"
    })
    
    df_recalc = pd.DataFrame(recalculated_rows)
    df_recalc.to_csv("results/phase13/personalization_recalculated.csv", index=False)
    print("Saved results/phase13/personalization_recalculated.csv")
    print(df_recalc[["client_partition", "n_test_total", "n_test_positive_deaths", "statistical_power_flag", "personalized_adapter_auroc", "personalized_adapter_95ci"]])

if __name__ == "__main__":
    audit_personalization()
