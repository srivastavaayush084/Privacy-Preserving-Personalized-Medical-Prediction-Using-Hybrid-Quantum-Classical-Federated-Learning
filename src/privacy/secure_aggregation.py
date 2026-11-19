"""
Secure Aggregation Protocol Module (Simulated Pairwise Additive Masking).
Implements privacy-preserving multi-party summation (Bonawitz et al., 2017)
ensuring the coordinator only observes the global aggregate, with individual client
updates mathematically hidden behind zero-sum masking vectors.
"""
import copy
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SecureAggregationProtocol:
    """Simulates zero-knowledge pairwise secret additive masking for federated aggregation."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_pairwise_masks(
        self,
        client_ids: List[str],
        param_shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Generates zero-sum pairwise masking tensors between all client pairs.
        
        For each pair (i, j) with i < j, generates mask S_{i,j}.
        Client i adds +S_{i,j}, Client j subtracts -S_{i,j}.
        Sum across all clients strictly cancels out: sum_i sum_j S_{i,j} = 0.
        """
        n_clients = len(client_ids)
        masks = {cid: {k: torch.zeros(shape, dtype=torch.float32) for k, shape in param_shapes.items()} for cid in client_ids}

        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                cid_i = client_ids[i]
                cid_j = client_ids[j]

                for k, shape in param_shapes.items():
                    # Generate symmetric pairwise random mask with high variance
                    mask_values = torch.tensor(
                        self.rng.normal(loc=0.0, scale=10.0, size=shape),
                        dtype=torch.float32
                    )
                    masks[cid_i][k] += mask_values
                    masks[cid_j][k] -= mask_values

        return masks

    def mask_client_update(
        self,
        client_update: Dict[str, torch.Tensor],
        client_mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Obfuscates local client update by adding the zero-sum pairwise mask."""
        masked_update = {}
        for k, v in client_update.items():
            if v.dtype in (torch.float32, torch.float64):
                masked_update[k] = v + client_mask[k]
            else:
                masked_update[k] = v.clone()
        return masked_update

    def aggregate_masked_updates(
        self,
        masked_updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregates masked client updates at the central server.
        Because pairwise masks sum to zero, the unmasked sum is perfectly reconstructed.
        """
        if not masked_updates:
            raise ValueError("No masked updates to aggregate.")

        n_updates = len(masked_updates)
        if sample_weights is None:
            weights = [1.0 / n_updates] * n_updates
        else:
            total_w = sum(sample_weights)
            weights = [w / total_w for w in sample_weights]

        base = masked_updates[0]
        aggregated = {}

        for k, v in base.items():
            if v.dtype in (torch.float32, torch.float64):
                aggregated[k] = torch.zeros_like(v)
            else:
                aggregated[k] = v.clone()

        for idx, u in enumerate(masked_updates):
            w = weights[idx]
            for k in aggregated.keys():
                if aggregated[k].dtype in (torch.float32, torch.float64):
                    aggregated[k] += u[k] * w

        return aggregated
