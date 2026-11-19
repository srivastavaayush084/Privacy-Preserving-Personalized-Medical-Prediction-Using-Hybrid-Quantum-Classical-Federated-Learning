"""Parameterized Quantum Circuit (PQC) definitions and resource profiler using PennyLane.
Executes real quantum state transformations and Pauli expectation value measurements.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pennylane as qml
import torch

logger = logging.getLogger(__name__)


def create_pqc_qnode(
    num_qubits: int = 4,
    circuit_depth: int = 2,
    entanglement: str = "strongly_entangling",
    shots: Optional[int] = None,
    device_name: str = "default.qubit"
):
    """Creates a parameterized quantum circuit QNode.
    
    Args:
        num_qubits: Number of quantum bits (e.g. 4, 6).
        circuit_depth: Number of variational PQC layers.
        entanglement: 'strongly_entangling' or 'basic_ring'.
        shots: Number of measurement shots (None = exact analytic statevector).
        device_name: PennyLane device backend.
        
    Returns:
        Tuple of (qnode, weight_shapes_dict, dev)
    """
    dev = qml.device(device_name, wires=num_qubits, shots=shots)

    if entanglement == "strongly_entangling":
        weight_shapes = {"weights": (circuit_depth, num_qubits, 3)}
        
        @qml.qnode(dev, interface="torch", diff_method="backprop" if shots is None else "parameter-shift")
        def circuit(inputs, weights):
            # 1. Angle Encoding: maps real clinical features to rotation angles around X or Y
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            # 2. Parameterized Entangling Layers
            qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
            # 3. Measurement: Pauli-Z expectation value on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    elif entanglement == "basic_ring":
        weight_shapes = {"weights": (circuit_depth, num_qubits)}
        
        @qml.qnode(dev, interface="torch", diff_method="backprop" if shots is None else "parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
            for layer in range(circuit_depth):
                for i in range(num_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Ring CNOTs
                for i in range(num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % num_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    else:
        raise ValueError(f"Unsupported entanglement type: {entanglement}")

    return circuit, weight_shapes, dev


def profile_quantum_circuit(
    num_qubits: int = 4,
    circuit_depth: int = 2,
    entanglement: str = "strongly_entangling"
) -> Dict[str, Any]:
    """Profiles quantum circuit resources: qubit count, gate count, circuit depth."""
    dev = qml.device("default.qubit", wires=num_qubits)
    circuit, weight_shapes, _ = create_pqc_qnode(
        num_qubits=num_qubits,
        circuit_depth=circuit_depth,
        entanglement=entanglement
    )

    dummy_inputs = torch.zeros(num_qubits)
    dummy_weights = torch.zeros(weight_shapes["weights"])

    specs = qml.specs(circuit)(dummy_inputs, dummy_weights)

    # Calculate gate breakdown
    num_rotations = circuit_depth * num_qubits * (3 if entanglement == "strongly_entangling" else 1)
    num_cnots = circuit_depth * num_qubits
    total_gates = num_rotations + num_cnots + num_qubits  # + angle embedding gates

    specs_depth = getattr(specs, "depth", circuit_depth * 3)
    if callable(specs_depth):
        try:
            specs_depth = specs_depth()
        except Exception:
            specs_depth = circuit_depth * 3

    profile = {
        "num_qubits": num_qubits,
        "circuit_depth": circuit_depth,
        "entanglement": entanglement,
        "trainable_parameters": int(np.prod(weight_shapes["weights"])),
        "estimated_total_gates": total_gates,
        "single_qubit_rotations": num_rotations,
        "two_qubit_entangling_gates": num_cnots,
        "specs_depth": int(specs_depth)
    }
    return profile
