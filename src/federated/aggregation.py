"""
Federated aggregation algorithms (FedAvg, FedProx, Trimmed Mean, Server Optimizer).
"""
import copy
import logging
from typing import Dict, List, Optional
import torch

from src.federated.communication import ClientUpdateMessage

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    client_updates: List[ClientUpdateMessage],
    global_model_state: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """Computes sample-weighted Federated Averaging (FedAvg) over client updates.
    
    Formula:
        W_{t+1} = sum_{k=1}^K (n_k / N) * W_k^{t+1}
        where N = sum_{k=1}^K n_k
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty list of client updates.")

    total_samples = sum(u.num_samples for u in client_updates)
    if total_samples <= 0:
        raise ValueError("Total client sample count must be positive.")

    # Initialize aggregated state with zeros matching the first client's parameter shapes
    base_state = client_updates[0].model_state_dict
    aggregated_state = {}

    for key, val in base_state.items():
        if val.dtype in (torch.float32, torch.float64, torch.float16):
            aggregated_state[key] = torch.zeros_like(val)
        else:
            # For non-floating types (like buffers/indices), keep base value
            aggregated_state[key] = val.clone()

    for update in client_updates:
        weight = float(update.num_samples) / float(total_samples)
        for key in aggregated_state.keys():
            if aggregated_state[key].dtype in (torch.float32, torch.float64, torch.float16):
                aggregated_state[key] += weight * update.model_state_dict[key]

    return aggregated_state


def trimmed_mean_aggregate(
    client_updates: List[ClientUpdateMessage],
    trim_fraction: float = 0.1
) -> Dict[str, torch.Tensor]:
    """Computes coordinate-wise trimmed mean across client updates.
    Provides Byzantine and outlier robustness against corrupted clients.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty list of client updates.")

    base_state = client_updates[0].model_state_dict
    aggregated_state = {}
    k = len(client_updates)
    num_to_trim = max(1, int(round(k * trim_fraction))) if (trim_fraction > 0 and k >= 3) else int(k * trim_fraction)

    for key, val in base_state.items():
        if val.dtype in (torch.float32, torch.float64, torch.float16):
            # Stack across clients: shape (K, *param_shape)
            stacked = torch.stack([u.model_state_dict[key] for u in client_updates], dim=0)
            if num_to_trim > 0 and 2 * num_to_trim < k:
                # Sort along client dimension and trim lowest and highest
                sorted_vals, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_vals[num_to_trim : k - num_to_trim]
                aggregated_state[key] = torch.mean(trimmed, dim=0)
            else:
                aggregated_state[key] = torch.mean(stacked, dim=0)
        else:
            aggregated_state[key] = val.clone()

    return aggregated_state
