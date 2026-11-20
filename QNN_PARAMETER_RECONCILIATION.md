# Source-Code Verification and Parameter Accounting of Hybrid Quantum and Classical Neural Network Architectures

**Author:** Senior Biostatistician, Quantum Machine Learning Specialist, Model Architecture Auditor  
**Date:** November 20, 2026  
**Project:** Privacy-Preserving Personalized Medical Prediction Using Hybrid Quantum-Classical Federated Learning  
**Status:** Certified Audit Document — Phase 14.1  

---

## 1. Executive Summary

This audit establishes the definitive, layer-by-layer parameter accounting for both the **Hybrid Quantum-Classical Neural Network (`DPHybridQNNNet`)** and the **Classical Baseline Deep Tabular MLP (`ClinicalDPMLPNet`)** as implemented in `src/privacy/formal_dp_engine.py` and utilized in all differential privacy and federated learning evaluations.

### Key Audit Findings
1. **Hybrid QNN Trainable Parameters:** Exactly **957** parameters, partitioned into:
   - **Classical Feature Encoder:** **772** parameters (including 32 affine parameters for GroupNorm).
   - **Parameterized Quantum Circuit (PQC):** **24** variational angles ($2 \times 4 \times 3$).
   - **Classical Classification Head:** **161** parameters ($144 + 17$).
2. **Classical MLP Trainable Parameters:** Exactly **5,537** parameters (including 224 affine parameters across three GroupNorm layers).
3. **Parameter Footprint Reduction:** The Hybrid QNN achieves an **82.72% reduction** in trainable parameters relative to the classical MLP ($5.79\times$ fewer parameters), while maintaining comparable discrimination ($0.7658$ vs $0.7166$ AUROC, paired bootstrap $p = 0.290$).
4. **Resolution of Historical Discrepancies:** Historical working drafts occasionally omitted GroupNorm affine parameters (reporting 740 for encoder and 5,313 for MLP) or misaggregated head layers. The current accounting reflects direct PyTorch introspection of instantiated model state dictionaries.

---

## 2. Hybrid QNN Parameter Decomposition (`DPHybridQNNNet`)

The hybrid QNN architecture bridges high-dimensional tabular clinical data and low-depth variational quantum circuits via a classical bottleneck and skip-connection head.

### 2.1 Layer-by-Layer Parameter Inventory

```python
class DPHybridQNNNet(nn.Module):
    # Encoder: Linear(41, 16) -> GroupNorm(2, 16) -> ReLU -> Linear(16, 4) -> Tanh
    # Quantum: 4-qubit Strongly Entangling PQC (depth=2, weight shape [2, 4, 3])
    # Head: Linear(8, 16) -> ReLU -> Dropout(0.20) -> Linear(16, 1)
```

| Sub-Module | PyTorch Parameter Identifier | Tensor Shape | Element Count | Sub-Module Total |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder** | `encoder.0.weight` | `torch.Size([16, 41])` | 656 | |
| | `encoder.0.bias` | `torch.Size([16])` | 16 | |
| | `encoder.1.weight` (GroupNorm Scale $\gamma$) | `torch.Size([16])` | 16 | |
| | `encoder.1.bias` (GroupNorm Shift $\beta$) | `torch.Size([16])` | 16 | |
| | `encoder.3.weight` | `torch.Size([4, 16])` | 64 | |
| | `encoder.3.bias` | `torch.Size([4])` | 4 | **772** |
| **Quantum Layer** | `quantum_layer.weights` | `torch.Size([2, 4, 3])` | 24 | **24** |
| **Classification Head** | `head.0.weight` | `torch.Size([16, 8])` | 128 | |
| | `head.0.bias` | `torch.Size([16])` | 16 | |
| | `head.3.weight` | `torch.Size([1, 16])` | 16 | |
| | `head.3.bias` | `torch.Size([1])` | 1 | **161** |
| **Total Model** | — | — | — | **957** |

### 2.2 Mathematical Structure of the Components
- **Encoder:** The first dense layer projects $41 \to 16$ ($41 \times 16 + 16 = 672$). The GroupNorm layer normalizes across 2 groups of 8 channels without mixing batch samples, adding 16 learnable scale factors ($\gamma$) and 16 biases ($\beta$) ($32$ parameters). The second dense layer compresses $16 \to 4$ ($16 \times 4 + 4 = 68$). Tanh activation bounds features to $[-1, 1]$, which are scaled by $\pi$ to serve as rotation angles for state preparation.
  $$\text{Encoder Total} = 672 + 32 + 68 = \mathbf{772}$$
- **Quantum Circuit:** A 4-qubit circuit with depth $L = 2$. Each layer applies arbitrary single-qubit rotations $R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_X(\alpha)$ requiring 3 angles per qubit ($3 \times 4 = 12$ angles per layer). Layers are interleaved with a circular entangling ladder of 4 CNOT gates ($0 \to 1, 1 \to 2, 2 \to 3, 3 \to 0$).
  $$\text{Quantum Total} = 2 \times 4 \times 3 = \mathbf{24}$$
- **Head:** Takes the concatenation of 4 quantum expectation values $\langle Z_i \rangle \in [-1, 1]$ and 4 bottleneck classical encoder features (8 dimensions total). Dense layer 1 projects $8 \to 16$ ($8 \times 16 + 16 = 144$). Dense layer 2 projects $16 \to 1$ ($16 \times 1 + 1 = 17$).
  $$\text{Head Total} = 144 + 17 = \mathbf{161}$$
- **Grand Total:** $772 + 24 + 161 = \mathbf{957}$.

---

## 3. Classical MLP Parameter Decomposition (`ClinicalDPMLPNet`)

The classical baseline is a fully connected deep tabular network designed with equivalent per-sample DP-SGD compatibility (GroupNorm with 2 groups per layer, eliminating batch dimension dependencies).

### 3.1 Layer-by-Layer Parameter Inventory

| Sub-Module | PyTorch Parameter Identifier | Tensor Shape | Element Count | Block Subtotal |
| :--- | :--- | :--- | :--- | :--- |
| **Block 1 ($41 \to 64$)** | `network.0.weight` | `torch.Size([64, 41])` | 2,624 | |
| | `network.0.bias` | `torch.Size([64])` | 64 | |
| | `network.1.weight` (GroupNorm Scale $\gamma$) | `torch.Size([64])` | 64 | |
| | `network.1.bias` (GroupNorm Shift $\beta$) | `torch.Size([64])` | 64 | **2,816** |
| **Block 2 ($64 \to 32$)** | `network.4.weight` | `torch.Size([32, 64])` | 2,048 | |
| | `network.4.bias` | `torch.Size([32])` | 32 | |
| | `network.5.weight` (GroupNorm Scale $\gamma$) | `torch.Size([32])` | 32 | |
| | `network.5.bias` (GroupNorm Shift $\beta$) | `torch.Size([32])` | 32 | **2,144** |
| **Block 3 ($32 \to 16$)** | `network.8.weight` | `torch.Size([16, 32])` | 512 | |
| | `network.8.bias` | `torch.Size([16])` | 16 | |
| | `network.9.weight` (GroupNorm Scale $\gamma$) | `torch.Size([16])` | 16 | |
| | `network.9.bias` (GroupNorm Shift $\beta$) | `torch.Size([16])` | 16 | **560** |
| **Output Layer ($16 \to 1$)** | `network.12.weight` | `torch.Size([1, 16])` | 16 | |
| | `network.12.bias` | `torch.Size([1])` | 1 | **17** |
| **Total Model** | — | — | — | **5,537** |

---

## 4. Architectural Comparison and Compression Ratios

| Metric | Classical Baseline (MLP) | Hybrid Architecture (QNN) | Compression / Reduction Ratio |
| :--- | :--- | :--- | :--- |
| **Input Dimension ($d$)** | 41 | 41 | 1.00x |
| **Feature Extraction Parameters** | 5,520 (Layers 1–3) | 772 (Encoder) | 7.15x fewer parameters |
| **Core Non-Linear / Latent Parameters** | N/A (Classical ReLUs) | 24 (Variational Quantum Angles) | Expressive Hilbert space encoding |
| **Classification Output Parameters** | 17 | 161 | 9.47x more capacity in head |
| **Total Trainable Parameters** | **5,537** | **957** | **82.72% reduction (5.79x fewer)** |
| **Single Serialized Payload (KB)** | 29.5 KB | 8.15 KB | **72.37% reduction (3.62x smaller)** |

---

## 5. Verification Against Paper Figures and Artifacts

1. **Figure 1 (`figures/fig1_architecture.png`):**
   Explicitly annotates the three modular blocks:
   - Encoder: 772 parameters
   - Quantum Circuit: 24 variational parameters
   - Head: 161 parameters
   - Total: 957 parameters.
2. **Figure 2 (`figures/fig2_quantum_circuit.png`):**
   Illustrates the 4-qubit Strongly Entangling PQC with 2 layers of $R(\alpha, \beta, \gamma)$ rotations and 4 entangling CNOTs, explicitly stating $2 \times 4 \times 3 = 24$ variational weights.
3. **Corrected Manuscript Text (`IEEE_Research_Paper_CORRECTED.tex`):**
   - Section IV-A and Section VII-D accurately report the 772 / 24 / 161 decomposition.
   - Table II correctly reports 957 parameters for all QNN variants and 5,537 parameters for all MLP variants.

---

## 6. Audit Conclusion

The parameter counts for both models are verified with mathematical precision directly from the source code. All reporting in the corrected manuscript is completely consistent with the audited implementation.
