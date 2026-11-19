"""
Phase 13 Model Complexity & Communication Audit Script.
Exhaustively decomposes parameter counts, byte footprints, and communication costs
for Classical MLP and Hybrid QNN.
"""
import os
import sys
import io
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical.baselines import ClinicalMLPNet
from src.privacy.formal_dp_engine import ClinicalDPMLPNet, DPHybridQNNNet
from src.models.quantum.hybrid_qnn import HybridQNNNet

def audit_complexity():
    os.makedirs("results/phase13", exist_ok=True)
    
    # 1. Inspect Classical Models
    mlp_dp = ClinicalDPMLPNet(input_dim=41, hidden_dims=[64, 32, 16], dropout=0.20)
    mlp_orig = ClinicalMLPNet(input_dim=41, hidden_dims=[64, 32, 16], dropout=0.20)
    
    dp_params = sum(p.numel() for p in mlp_dp.parameters() if p.requires_grad)
    orig_params = sum(p.numel() for p in mlp_orig.parameters() if p.requires_grad)
    
    # Layer-by-layer breakdown for DP-MLP
    dp_mlp_layers = []
    for name, param in mlp_dp.named_parameters():
        dp_mlp_layers.append({
            "component": "Classical_DP_MLP",
            "layer_name": name,
            "shape": list(param.shape),
            "parameters": param.numel(),
            "trainable": param.requires_grad
        })
        
    # 2. Inspect Hybrid QNN Models
    qnn_dp = DPHybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=2)
    qnn_orig = HybridQNNNet(input_dim=41, num_qubits=4, circuit_depth=2)
    
    qnn_dp_params = sum(p.numel() for p in qnn_dp.parameters() if p.requires_grad)
    qnn_orig_params = sum(p.numel() for p in qnn_orig.parameters() if p.requires_grad)
    
    encoder_params = sum(p.numel() for p in qnn_dp.encoder.parameters() if p.requires_grad)
    pqc_params = sum(p.numel() for p in qnn_dp.quantum_layer.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in qnn_dp.head.parameters() if p.requires_grad)
    
    qnn_layers = []
    for name, param in qnn_dp.named_parameters():
        qnn_layers.append({
            "component": "Hybrid_DP_QNN",
            "layer_name": name,
            "shape": list(param.shape),
            "parameters": param.numel(),
            "trainable": param.requires_grad
        })
        
    # Measure serialized state_dict size in memory
    buf_mlp = io.BytesIO()
    torch.save(mlp_dp.state_dict(), buf_mlp)
    mlp_state_dict_bytes = buf_mlp.getbuffer().nbytes
    
    buf_qnn = io.BytesIO()
    torch.save(qnn_dp.state_dict(), buf_qnn)
    qnn_state_dict_bytes = buf_qnn.getbuffer().nbytes
    
    # Build model_complexity_comparison.csv
    complexity_rows = [
        {
            "model_architecture": "Classical_DP_MLP (GroupNorm)",
            "encoder_parameters": 0,
            "pqc_quantum_parameters": 0,
            "head_parameters": 0,
            "total_trainable_parameters": dp_params,
            "quantum_parameter_percentage": 0.0,
            "serialized_state_dict_bytes": mlp_state_dict_bytes,
            "serialized_state_dict_kb": round(mlp_state_dict_bytes / 1024.0, 2),
            "architectural_notes": "Input (41) -> Linear(64) -> GN(64) -> Linear(32) -> GN(32) -> Linear(16) -> GN(16) -> Out(1)"
        },
        {
            "model_architecture": "Classical_MLP_Baseline (BatchNorm)",
            "encoder_parameters": 0,
            "pqc_quantum_parameters": 0,
            "head_parameters": 0,
            "total_trainable_parameters": orig_params,
            "quantum_parameter_percentage": 0.0,
            "serialized_state_dict_bytes": mlp_state_dict_bytes,
            "serialized_state_dict_kb": round(mlp_state_dict_bytes / 1024.0, 2),
            "architectural_notes": "Input (41) -> Linear(64) -> BN(64) -> Linear(32) -> BN(32) -> Linear(16) -> BN(16) -> Out(1)"
        },
        {
            "model_architecture": "Hybrid_DP_QNN (4-qubit, depth-2)",
            "encoder_parameters": encoder_params,
            "pqc_quantum_parameters": pqc_params,
            "head_parameters": head_params,
            "total_trainable_parameters": qnn_dp_params,
            "quantum_parameter_percentage": round((pqc_params / qnn_dp_params) * 100.0, 2),
            "serialized_state_dict_bytes": qnn_state_dict_bytes,
            "serialized_state_dict_kb": round(qnn_state_dict_bytes / 1024.0, 2),
            "architectural_notes": f"Classical Encoder ({encoder_params} params) + PQC Circuit ({pqc_params} params) + Head ({head_params} params)"
        },
        {
            "model_architecture": "PQC_Quantum_Sublayer_Only",
            "encoder_parameters": 0,
            "pqc_quantum_parameters": pqc_params,
            "head_parameters": 0,
            "total_trainable_parameters": pqc_params,
            "quantum_parameter_percentage": 100.0,
            "serialized_state_dict_bytes": 24 * 4,  # 24 float32 values = 96 bytes raw
            "serialized_state_dict_kb": 0.09,
            "architectural_notes": "The '24 parameters' claim applies SOLELY to this variational circuit layer, NOT the full hybrid model!"
        }
    ]
    df_complexity = pd.DataFrame(complexity_rows)
    df_complexity.to_csv("results/phase13/model_complexity_comparison.csv", index=False)
    print("Saved results/phase13/model_complexity_comparison.csv")
    print(df_complexity[["model_architecture", "total_trainable_parameters", "pqc_quantum_parameters", "serialized_state_dict_kb"]])
    
    # 3. Federated Communication Recalculation
    # 5 hospital clients, 10 communication rounds
    # In each round:
    # 1. Server broadcasts global model to 5 clients (5 downlinks)
    # 2. 5 clients upload local model updates to server (5 uplinks)
    # Total transmissions = 10 rounds * 5 clients * 2 (up + down) = 100 transmissions
    
    # Actual payload from Phase 12 federated runner:
    mlp_total_mb = 2.880764
    qnn_total_mb = 0.795650
    
    mlp_per_round_mb = mlp_total_mb / 10.0
    qnn_per_round_mb = qnn_total_mb / 10.0
    
    mlp_per_client_round_kb = (mlp_total_mb * 1024.0) / (10 * 5 * 2)  # Single payload
    qnn_per_client_round_kb = (qnn_total_mb * 1024.0) / (10 * 5 * 2)
    
    reduction_pct = ((mlp_total_mb - qnn_total_mb) / mlp_total_mb) * 100.0
    reduction_factor = mlp_total_mb / qnn_total_mb
    
    comm_rows = [
        {
            "metric": "Number of Participating Hospital Clients",
            "classical_mlp": 5,
            "hybrid_qnn": 5,
            "reduction_or_ratio": "1.0x"
        },
        {
            "metric": "Number of Federated Training Rounds",
            "classical_mlp": 10,
            "hybrid_qnn": 10,
            "reduction_or_ratio": "1.0x"
        },
        {
            "metric": "Total Transmissions (Uplinks + Downlinks)",
            "classical_mlp": 100,
            "hybrid_qnn": 100,
            "reduction_or_ratio": "1.0x"
        },
        {
            "metric": "Single Payload Size (KB)",
            "classical_mlp": round(mlp_per_client_round_kb, 2),
            "hybrid_qnn": round(qnn_per_client_round_kb, 2),
            "reduction_or_ratio": f"{reduction_factor:.2f}x smaller"
        },
        {
            "metric": "Communication Bandwidth Per Round (MB)",
            "classical_mlp": round(mlp_per_round_mb, 4),
            "hybrid_qnn": round(qnn_per_round_mb, 4),
            "reduction_or_ratio": f"{reduction_factor:.2f}x smaller"
        },
        {
            "metric": "Total Consortium Bandwidth Across 10 Rounds (MB)",
            "classical_mlp": round(mlp_total_mb, 4),
            "hybrid_qnn": round(qnn_total_mb, 4),
            "reduction_or_ratio": f"{reduction_factor:.2f}x smaller ({reduction_pct:.1f}% reduction)"
        },
        {
            "metric": "Total Trainable Parameters in Model",
            "classical_mlp": dp_params,
            "hybrid_qnn": qnn_dp_params,
            "reduction_or_ratio": f"{dp_params / qnn_dp_params:.2f}x fewer parameters"
        },
        {
            "metric": "Variational Quantum Parameters Transmitted",
            "classical_mlp": 0,
            "hybrid_qnn": pqc_params,
            "reduction_or_ratio": "24 PQC parameters"
        }
    ]
    df_comm = pd.DataFrame(comm_rows)
    df_comm.to_csv("results/phase13/communication_recalculation.csv", index=False)
    print("Saved results/phase13/communication_recalculation.csv")
    print(df_comm[["metric", "classical_mlp", "hybrid_qnn", "reduction_or_ratio"]])

if __name__ == "__main__":
    audit_complexity()
