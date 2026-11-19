"""Noisy quantum simulation module using PennyLane's density matrix backend.
Benchmarks the impact of NISQ decoherence and gate noise on medical prediction.
"""

import logging
from typing import Dict, Optional, Tuple
import pennylane as qml
import torch

logger = logging.getLogger(__name__)


def create_noisy_pqc_qnode(
    num_qubits: int = 4,
    circuit_depth: int = 2,
    depolarizing_prob: float = 0.02,
    bit_flip_prob: float = 0.01
):
    """Creates a noise-aware quantum circuit using density matrix simulation.
    
    Args:
        num_qubits: Number of quantum bits.
        circuit_depth: PQC layers.
        depolarizing_prob: Probability of depolarizing channel after two-qubit gates.
        bit_flip_prob: Probability of bit flip after single-qubit rotations.
    """
    dev_noisy = qml.device("default.mixed", wires=num_qubits)
    weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}

    @qml.qnode(dev_noisy, interface="torch")
    def circuit(inputs, weights):
        # 1. State preparation
        qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")

        # 2. Parameterized noisy layers
        for layer in range(circuit_depth):
            for i in range(num_qubits):
                qml.Rot(*weights[layer, i], wires=i)
                if bit_flip_prob > 0:
                    qml.BitFlip(bit_flip_prob, wires=i)

            # Entangling gates with depolarizing channel
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=i)
                    qml.DepolarizingChannel(depolarizing_prob, wires=(i + 1) % num_qubits)

        # 3. Expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit, weight_shapes, dev_noisy
