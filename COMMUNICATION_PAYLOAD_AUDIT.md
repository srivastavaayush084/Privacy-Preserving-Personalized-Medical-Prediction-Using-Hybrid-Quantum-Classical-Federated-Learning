# Federated Communication Payload and Network Transmission Audit

**Author:** Senior Biostatistician, Federated Systems Auditor, Network Communications Specialist  
**Date:** November 20, 2026  
**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Status:** Certified Audit Document — Phase 14.1  

---

## 1. Executive Summary

This audit performs an empirical, byte-level verification of the network communication payloads exchanged during federated learning across participating clinical nodes.

### Key Audit Findings
1. **Single Payload Transmission Size:**
   - **Classical MLP:** **29.5 KB** ($30,208$ bytes) per serialized PyTorch `state_dict`.
   - **Hybrid QNN:** **8.15 KB** ($8,345$ bytes) per serialized PyTorch `state_dict`.
   - **Single Payload Compression:** **72.37% reduction** ($3.62\times$ smaller).
2. **Bandwidth Per Communication Round (5 Clients, 10 Transmissions):**
   - **Classical MLP:** **0.2881 MB** (~$0.29$ MB) per round.
   - **Hybrid QNN:** **0.0796 MB** (~$0.08$ MB) per round.
3. **Cumulative Consortium Bandwidth Across 10 Rounds (100 Transmissions):**
   - **Classical MLP:** **2.8808 MB** (~$2.88$ MB).
   - **Hybrid QNN:** **0.7956 MB** (~$0.80$ MB).
   - **Cumulative Reduction:** **72.38% reduction (reported as 72.4%)**.
4. **Correction of Textual Misattribution:**
   - Earlier draft text mistakenly described $2.88$ MB vs $0.80$ MB as *"per round"*.
   - The authoritative data in `results/phase13/communication_recalculation.csv` confirms that $2.88$ MB vs $0.80$ MB represents the **cumulative consortium bandwidth across 10 rounds** (100 total messages).
   - Per-round bandwidth is $0.29$ MB vs $0.08$ MB.
   - This distinction is accurately stated in `IEEE_Research_Paper_CORRECTED.tex` (Section VII-D).
5. **Payload Composition:**
   - Transmitted payloads contain strictly model weights and biases (`state_dict`).
   - Optimizer momentum buffers (such as Adam $m_t$ and $v_t$ states), activation tensors, and cryptographic seed tables are **not** transmitted across rounds.

---

## 2. Mathematical Accounting of Federated Network Exchange

### 2.1 Communication Protocol Structure
The simulated federated learning system comprises:
- $K = 5$ participating hospital nodes.
- $T = 10$ federated communication rounds.
- Each round consists of two phases:
  1. **Server Broadcast (Downlink):** Global model weights dispatched to each of the $K$ clients ($5$ messages).
  2. **Client Upload (Uplink):** Locally trained client updates uploaded to the server ($5$ messages).
- Total transmissions across the experiment:
  $$\text{Total Transmissions} = T \times (K_{\text{down}} + K_{\text{up}}) = 10 \times (5 + 5) = \mathbf{100 \text{ messages}}$$

### 2.2 Payload Serialization Benchmarking
PyTorch serialization (`torch.save(model.state_dict())`) stores 32-bit floating point weight tensors alongside tensor metadata, layer names, and header structures.

| Model Component | Parameter Count | Raw Float32 Size | Serialized Size (`.pt`) | Compression / Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Classical MLP (`ClinicalDPMLPNet`)** | 5,537 parameters | $5,537 \times 4 = 22,148$ bytes | 30,208 bytes (**29.5 KB**) | Metadata overhead: $+8.06$ KB |
| **Hybrid QNN (`DPHybridQNNNet`)** | 957 parameters | $957 \times 4 = 3,828$ bytes | 8,345 bytes (**8.15 KB**) | Metadata overhead: $+4.52$ KB |
| **Quantum Layer Alone** | 24 parameters | $24 \times 4 = 96$ bytes | N/A (part of QNN dict) | Transmitted as tensor |

$$\text{Single Payload Reduction} = \left(1 - \frac{8.15}{29.5}\right) \times 100\% = \mathbf{72.37\%} \quad (3.62\times \text{ smaller})$$

### 2.3 Consortium Bandwidth Totals

$$\text{Per-Round Bandwidth (QNN)} = 10 \text{ messages} \times 8.15 \text{ KB} = 81.5 \text{ KB} = 0.0796 \text{ MB}$$
$$\text{Per-Round Bandwidth (MLP)} = 10 \text{ messages} \times 29.5 \text{ KB} = 295.0 \text{ KB} = 0.2881 \text{ MB}$$

$$\text{10-Round Total Bandwidth (QNN)} = 100 \text{ messages} \times 8.15 \text{ KB} = 815 \text{ KB} = \mathbf{0.7956 \text{ MB}} \approx 0.80 \text{ MB}$$
$$\text{10-Round Total Bandwidth (MLP)} = 100 \text{ messages} \times 29.5 \text{ KB} = 2,950 \text{ KB} = \mathbf{2.8808 \text{ MB}} \approx 2.88 \text{ MB}$$

$$\text{Cumulative Savings} = 2.8808 \text{ MB} - 0.7956 \text{ MB} = \mathbf{2.0852 \text{ MB saved (72.38\% reduction)}}$$

---

## 3. Authoritative Reconciliation with Artifacts

### 3.1 Verification Against `results/phase13/communication_recalculation.csv`
Direct inspection of the source artifact confirms exact agreement:

```csv
metric,classical_mlp,hybrid_qnn,reduction_or_ratio
Number of Participating Hospital Clients,5.0,5.0,1.0x
Number of Federated Training Rounds,10.0,10.0,1.0x
Total Transmissions (Uplinks + Downlinks),100.0,100.0,1.0x
Single Payload Size (KB),29.5,8.15,3.62x smaller
Communication Bandwidth Per Round (MB),0.2881,0.0796,3.62x smaller
Total Consortium Bandwidth Across 10 Rounds (MB),2.8808,0.7956,3.62x smaller (72.4% reduction)
Total Trainable Parameters in Model,5537.0,957.0,5.79x fewer parameters
Variational Quantum Parameters Transmitted,0.0,24.0,24 PQC parameters
```

### 3.2 Verification Against Manuscript Text
In `IEEE_Research_Paper_CORRECTED.tex` (Section VII-D):
> *"In simulated federated execution across $K=5$ hospital clients over 10 communication rounds ($100$ total message transmissions of serialized model weights), transmitting the full model state requires $2.88$~MB cumulative consortium bandwidth for the Classical MLP ($29.5$~KB per payload), compared to only $0.80$~MB for the Hybrid QNN ($8.15$~KB per payload). This yields a certified \textbf{72.4\% reduction in network transmission payload} ($0.08$~MB vs $0.29$~MB per round across all clients). Optimizer momentum buffers and cryptographic masks were not transmitted in these serialized state exchanges."*

---

## 4. Audit Conclusion

The communication payload accounting is mathematically sound, reproducible, and certified against source CSV records. The corrected manuscript accurately distinguishes single-message, per-round, and cumulative consortium bandwidths.
