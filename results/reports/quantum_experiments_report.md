# Quantum Machine Learning Benchmark & Circuit Profiling Report

## System Architecture: Hybrid Quantum-Classical Neural Network
- **Input Dimension:** 41 clinical features (First 24h ICU)
- **Bottleneck Projection:** Compressed to $N_{qubits}$ rotation angles via `nn.Tanh() * π`
- **Quantum Execution:** PennyLane `default.qubit` statevector simulation with analytic backpropagation
- **Measurement Strategy:** Pauli-$Z$ expectation values $\langle Z_i \rangle \in [-1, 1]$

## Experimental Results Matrix

| Experiment | Qubits | Depth | Gates | Train Time (s) | Test AUROC | Test PR-AUC | Test Accuracy | Brier Score | ECE |
|---|---|---|---|---|---|---|---|---|---|
| **Hybrid QNN (4-Qubit Depth-2 Strong)** | 4 | 2 | 36 | 33.42s | **0.8346** | 0.3311 | 0.8193 | 0.1395 | 0.1484 |
| **Hybrid QNN (4-Qubit Depth-1 Light)** | 4 | 1 | 20 | 21.06s | **0.7756** | 0.2193 | 0.6768 | 0.2101 | 0.241 |
| **Hybrid QNN (4-Qubit Depth-3 Deep)** | 4 | 3 | 52 | 36.78s | **0.7533** | 0.2947 | 0.7379 | 0.1726 | 0.2019 |
| **Hybrid QNN (6-Qubit Scaled Register)** | 6 | 2 | 54 | 44.03s | **0.7863** | 0.2844 | 0.7252 | 0.1877 | 0.2231 |
| **Hybrid QNN (4-Qubit Ring Entangling)** | 4 | 2 | 20 | 21.35s | **0.7543** | 0.2746 | 0.715 | 0.1911 | 0.2215 |

## Quantum vs Classical Matched Comparison
- **Best Classical Baseline:** XGBoost (Test AUROC: 0.8445, PR-AUC: 0.3849) and Random Forest (Test AUROC: 0.8411, PR-AUC: 0.3403).
- **Hybrid QNN Performance:** Reaches competitive discrimination while compressing 41 features into a low-dimensional Hilbert space (4–6 qubits).
- **Resource Scaling:** Adding qubits and layers expands the gate count linearly while scaling simulation memory as $\mathcal{O}(2^N)$.
