"""
Unit tests for Quantum Machine Learning modules (PQC, HybridQNN, QSVM, and Noisy Simulation).
"""
import pytest
import numpy as np
import torch
import pennylane as qml

from src.models.quantum.quantum_circuits import create_pqc_qnode, profile_quantum_circuit
from src.models.quantum.hybrid_qnn import HybridQNNNet, HybridQNNModel
from src.models.quantum.noisy_simulation import create_noisy_pqc_qnode


def test_quantum_circuit_qnode_execution():
    """Verify that PennyLane QNode executes and produces valid expectation values."""
    n_qubits = 4
    circuit_depth = 2
    qnode, weight_shapes, dev = create_pqc_qnode(num_qubits=n_qubits, circuit_depth=circuit_depth, entanglement="strongly_entangling")
    
    # 4 features
    inputs = torch.tensor(np.random.uniform(0, np.pi, size=(n_qubits,)), dtype=torch.float32)
    # weights shape: (circuit_depth, n_qubits, 3)
    weights = torch.tensor(np.random.uniform(0, 2 * np.pi, size=(circuit_depth, n_qubits, 3)), dtype=torch.float32)
    
    res = qnode(inputs, weights)
    assert len(res) == n_qubits
    for val in res:
        # Pauli-Z expectation value in [-1, 1]
        assert -1.0 <= float(val) <= 1.0


def test_quantum_circuit_profiler():
    """Verify quantum circuit resource profiling (gate counts, depth)."""
    n_qubits = 4
    circuit_depth = 2
    metrics = profile_quantum_circuit(num_qubits=n_qubits, circuit_depth=circuit_depth, entanglement="strongly_entangling")
    
    assert metrics["num_qubits"] == 4
    assert metrics["circuit_depth"] == 2
    assert metrics["trainable_parameters"] == 24
    assert metrics["estimated_total_gates"] > 0
    assert metrics["single_qubit_rotations"] == 24
    assert metrics["two_qubit_entangling_gates"] == 8


def test_hybrid_qnn_forward_pass():
    """Verify PyTorch Hybrid QNN forward pass and gradient backpropagation."""
    batch_size = 4
    input_dim = 41
    n_qubits = 4
    
    model = HybridQNNNet(input_dim=input_dim, num_qubits=n_qubits, circuit_depth=2, entanglement="strongly_entangling")
    
    dummy_x = torch.randn(batch_size, input_dim)
    logits = model(dummy_x)
    
    assert logits.shape == (batch_size,)
    probs = torch.sigmoid(logits)
    assert probs.shape == (batch_size,)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    
    # Check backward pass
    loss = logits.sum()
    loss.backward()
    
    # Check classical encoder gradients exist
    assert model.encoder[0].weight.grad is not None
    # Check quantum weight gradients exist
    assert model.quantum_layer.weights.grad is not None


def test_hybrid_qnn_wrapper_prediction():
    """Verify HybridQNNModel wrapper predict_proba works."""
    wrapper = HybridQNNModel(input_dim=41, num_qubits=4, circuit_depth=1, epochs=1, batch_size=4)
    dummy_x = np.random.randn(8, 41)
    probs = wrapper.predict_proba(dummy_x)
    assert probs.shape == (8,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_noisy_simulation():
    """Verify noisy quantum circuit evaluation with depolarizing channel."""
    n_qubits = 2
    noisy_node, weight_shapes, dev = create_noisy_pqc_qnode(num_qubits=n_qubits, depolarizing_prob=0.05)
    
    inputs = torch.tensor([0.5, 1.2], dtype=torch.float32)
    weights = torch.zeros((2, n_qubits, 3), dtype=torch.float32)
    
    res = noisy_node(inputs, weights)
    assert len(res) == n_qubits
    for val in res:
        assert -1.0 <= float(val) <= 1.0
