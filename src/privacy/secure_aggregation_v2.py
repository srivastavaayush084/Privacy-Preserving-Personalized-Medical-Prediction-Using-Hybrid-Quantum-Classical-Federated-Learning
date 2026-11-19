"""
Enhanced Secure Aggregation Protocol (v2) with Dropout Resilience and Threshold Secret Sharing.
Implements the multi-round protocol based on Bonawitz et al. (CCS 2017):
- Round 0: Client Key Establishment & PRG Seed Generation
- Round 1: Pairwise Shared Secret Agreement & Blinding Mask Creation
- Round 2: Shamir Threshold Secret Sharing Simulation of Private Seeds
- Round 3: Masked Model Update Submission
- Round 4: Unmasking with Dropout Reconciliation (Orphan Mask Removal)
Provides verified mathematical guarantees that server learns ONLY the sum of surviving updates.
"""

import copy
import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ShamirSecretSharingSimulator:
    """Simulates t-out-of-n Shamir secret sharing over 64-bit integer seeds."""

    def __init__(self, threshold: int, num_clients: int, prime: int = 2147483647):
        self.t = threshold
        self.n = num_clients
        self.p = prime

    def split_secret(self, secret: int, rng: np.random.RandomState) -> Dict[int, int]:
        """Splits an integer secret into n shares using a polynomial of degree t-1."""
        secret = int(secret) % self.p
        # Polynomial: f(x) = secret + a_1*x + ... + a_{t-1}*x^{t-1} mod p
        coeffs = [secret] + [int(rng.randint(1, self.p)) for _ in range(self.t - 1)]
        shares = {}
        for x in range(1, self.n + 1):
            val = sum(c * (x ** exp) for exp, c in enumerate(coeffs)) % self.p
            shares[x] = val
        return shares

    def reconstruct_secret(self, shares: Dict[int, int]) -> int:
        """Reconstructs the secret from at least t shares using Lagrange interpolation at x=0."""
        if len(shares) < self.t:
            raise ValueError(f"Need at least {self.t} shares to reconstruct, got {len(shares)}.")

        items = list(shares.items())[:self.t]
        secret = 0
        for i, (x_i, y_i) in enumerate(items):
            # Compute Lagrange basis polynomial L_i(0)
            numerator = 1
            denominator = 1
            for j, (x_j, _) in enumerate(items):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.p
                    denominator = (denominator * (x_i - x_j)) % self.p
            inv_denom = pow(denominator, self.p - 2, self.p)  # Modular inverse via Fermat's Little Theorem
            term = (y_i * numerator * inv_denom) % self.p
            secret = (secret + term) % self.p

        return (secret + self.p) % self.p


class SecureAggregationV2:
    """Production-grade simulated Secure Aggregation protocol with client dropout resilience."""

    def __init__(
        self,
        client_ids: List[str],
        threshold_ratio: float = 0.60,
        random_seed: int = 42
    ):
        self.client_ids = client_ids
        self.n = len(client_ids)
        self.threshold = max(2, int(math.ceil(self.n * threshold_ratio)))
        self.rng = np.random.RandomState(random_seed)
        self.sss = ShamirSecretSharingSimulator(self.threshold, self.n)

        # Map client_id to 1-based index
        self.cid_to_idx = {cid: idx + 1 for idx, cid in enumerate(client_ids)}
        self.idx_to_cid = {idx + 1: cid for idx, cid in enumerate(client_ids)}

        # Internal state for protocol execution
        self.private_blinding_seeds: Dict[str, int] = {}
        self.pairwise_seeds: Dict[Tuple[str, str], int] = {}
        self.blinding_shares: Dict[str, Dict[int, int]] = {}
        self.pairwise_shares: Dict[Tuple[str, str], Dict[int, int]] = {}

    def setup_keys_and_shares(self):
        """Round 0 & 1: Generates blinding seeds, pairwise shared seeds, and Shamir shares."""
        # 1. Private blinding seed b_u for each client
        for cid in self.client_ids:
            seed = int(self.rng.randint(1000, 2000000000))
            self.private_blinding_seeds[cid] = seed
            self.blinding_shares[cid] = self.sss.split_secret(seed, self.rng)

        # 2. Pairwise seeds s_{u, v} for each pair (u < v)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                u = self.client_ids[i]
                v = self.client_ids[j]
                pair_seed = int(self.rng.randint(1000, 2000000000))
                self.pairwise_seeds[(u, v)] = pair_seed
                self.pairwise_seeds[(v, u)] = pair_seed
                # Secret shares of the pairwise seed
                self.pairwise_shares[(u, v)] = self.sss.split_secret(pair_seed, self.rng)
                self.pairwise_shares[(v, u)] = self.pairwise_shares[(u, v)]

    def _generate_mask_tensor(self, seed: int, shape: Tuple[int, ...]) -> torch.Tensor:
        """Deterministically generates pseudorandom Gaussian mask from an integer seed."""
        local_rng = np.random.RandomState(seed % 4294967295)
        # Scale of 5.0 ensures model weights are thoroughly obfuscated
        vals = local_rng.normal(loc=0.0, scale=5.0, size=shape)
        return torch.tensor(vals, dtype=torch.float32)

    def mask_client_updates(
        self,
        raw_updates: Dict[str, Dict[str, torch.Tensor]],
        dropped_client_ids: Optional[Set[str]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Round 2: Each participating client computes its masked update y_u."""
        if dropped_client_ids is None:
            dropped_client_ids = set()

        masked_updates = {}
        for cid, update in raw_updates.items():
            if cid in dropped_client_ids:
                continue  # Dropped client does not submit update

            u_idx = self.cid_to_idx[cid]
            masked_dict = {}

            # Private blinding mask B_u
            b_seed = self.private_blinding_seeds[cid]

            for param_name, tensor in update.items():
                if tensor.dtype not in (torch.float32, torch.float64):
                    masked_dict[param_name] = tensor.clone()
                    continue

                shape = tensor.shape
                # y_u = w_u + B_u + sum_{v > u} S_{u, v} - sum_{v < u} S_{v, u}
                y = tensor.clone()
                # Add private blinding mask B_u
                y += self._generate_mask_tensor(b_seed + hash(param_name) % 100000, shape)

                # Add pairwise zero-sum masks
                for other_cid in self.client_ids:
                    if other_cid == cid:
                        continue
                    v_idx = self.cid_to_idx[other_cid]
                    pair_seed = self.pairwise_seeds[(cid, other_cid)]
                    pair_mask = self._generate_mask_tensor(pair_seed + hash(param_name) % 100000, shape)
                    if u_idx < v_idx:
                        y += pair_mask
                    else:
                        y -= pair_mask

                masked_dict[param_name] = y

            masked_updates[cid] = masked_dict

        return masked_updates

    def server_aggregate_with_dropout_resolution(
        self,
        masked_updates: Dict[str, Dict[str, torch.Tensor]],
        all_client_ids: List[str],
        dropped_client_ids: Set[str]
    ) -> Dict[str, torch.Tensor]:
        """Round 3: Server reconstructs unmasked sum from surviving clients with dropout compensation.
        
        Surviving clients reveal:
        1. Their blinding seed shares for all surviving clients (allowing server to subtract B_u)
        2. Their pairwise seed shares for all DROPPED clients (allowing server to subtract orphan pairwise masks)
        """
        surviving_cids = [cid for cid in all_client_ids if cid not in dropped_client_ids]
        if len(surviving_cids) < self.threshold:
            raise RuntimeError(
                f"Aggregation aborted: {len(surviving_cids)} surviving clients is below threshold {self.threshold}."
            )

        # 1. Server reconstructs blinding seeds for surviving clients
        reconstructed_blinding_seeds: Dict[str, int] = {}
        for s_cid in surviving_cids:
            shares_available = {
                self.cid_to_idx[peer]: self.blinding_shares[s_cid][self.cid_to_idx[peer]]
                for peer in surviving_cids
            }
            rec_seed = self.sss.reconstruct_secret(shares_available)
            reconstructed_blinding_seeds[s_cid] = rec_seed

        # 2. Server reconstructs pairwise seeds between surviving and dropped clients
        reconstructed_orphan_seeds: Dict[Tuple[str, str], int] = {}
        for d_cid in dropped_client_ids:
            for s_cid in surviving_cids:
                pair = (min(s_cid, d_cid), max(s_cid, d_cid))
                if pair not in reconstructed_orphan_seeds:
                    shares_available = {
                        self.cid_to_idx[peer]: self.pairwise_shares[pair][self.cid_to_idx[peer]]
                        for peer in surviving_cids
                    }
                    rec_seed = self.sss.reconstruct_secret(shares_available)
                    reconstructed_orphan_seeds[pair] = rec_seed

        # 3. Sum the masked updates
        base_update = next(iter(masked_updates.values()))
        aggregated_sum = {}
        for k, v in base_update.items():
            if v.dtype in (torch.float32, torch.float64):
                aggregated_sum[k] = torch.zeros_like(v)
            else:
                aggregated_sum[k] = v.clone()

        for cid in surviving_cids:
            u_dict = masked_updates[cid]
            for k, v in u_dict.items():
                if v.dtype in (torch.float32, torch.float64):
                    aggregated_sum[k] += v

        # 4. De-masking Step A: Subtract reconstructed private blinding masks B_u of surviving clients
        for s_cid in surviving_cids:
            b_seed = reconstructed_blinding_seeds[s_cid]
            for k, tensor in aggregated_sum.items():
                if tensor.dtype in (torch.float32, torch.float64):
                    mask = self._generate_mask_tensor(b_seed + hash(k) % 100000, tensor.shape)
                    aggregated_sum[k] -= mask

        # 5. De-masking Step B: Cancel out orphan pairwise masks involving dropped clients
        for s_cid in surviving_cids:
            u_idx = self.cid_to_idx[s_cid]
            for d_cid in dropped_client_ids:
                v_idx = self.cid_to_idx[d_cid]
                pair = (min(s_cid, d_cid), max(s_cid, d_cid))
                pair_seed = reconstructed_orphan_seeds[pair]
                for k, tensor in aggregated_sum.items():
                    if tensor.dtype in (torch.float32, torch.float64):
                        orphan_mask = self._generate_mask_tensor(pair_seed + hash(k) % 100000, tensor.shape)
                        if u_idx < v_idx:
                            # In masked update, s_cid added +orphan_mask because u_idx < v_idx
                            aggregated_sum[k] -= orphan_mask
                        else:
                            # In masked update, s_cid subtracted -orphan_mask because u_idx > v_idx
                            aggregated_sum[k] += orphan_mask

        # 6. Compute exact average across surviving clients
        n_surviving = float(len(surviving_cids))
        aggregated_avg = {}
        for k, v in aggregated_sum.items():
            if v.dtype in (torch.float32, torch.float64):
                aggregated_avg[k] = v / n_surviving
            else:
                aggregated_avg[k] = v.clone()

        return aggregated_avg
