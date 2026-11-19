"""Quantum Kernel Support Vector Machine (QSVM).
Computes quantum state fidelity kernels and trains precomputed kernel SVM.
"""

import logging
import time
from typing import Dict, Optional, Tuple
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class QuantumKernelSVM:
    """QSVM: computes quantum transition amplitude kernel matrix using fidelity."""

    def __init__(self, num_qubits: int = 4, C: float = 1.0, random_state: int = 42):
        self.num_qubits = num_qubits
        self.C = C
        self.random_state = random_state
        self.pca = PCA(n_components=num_qubits)

        # Setup PennyLane device and QNode for inner product / fidelity
        self.dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # Encode x1
            qml.AngleEmbedding(x1, wires=range(num_qubits), rotation="Y")
            # Adjoint encode x2
            qml.adjoint(qml.AngleEmbedding)(x2, wires=range(num_qubits), rotation="Y")
            # Return probability of all zeros state |0...0>
            return qml.probs(wires=range(num_qubits))

        self.kernel_qnode = kernel_circuit
        self.X_train_reduced: Optional[np.ndarray] = None
        self.classifier: Optional[CalibratedClassifierCV] = None

    def _compute_gram_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes pairwise quantum kernel matrix."""
        n1 = len(X1)
        n2 = len(X2)
        K = np.zeros((n1, n2))

        # Normalize features to [-pi, pi]
        X1_norm = np.clip(X1, -np.pi, np.pi)
        X2_norm = np.clip(X2, -np.pi, np.pi)

        for i in range(n1):
            for j in range(n2):
                if X1 is X2 and j < i:
                    K[i, j] = K[j, i]
                else:
                    # Prob of 0...0 state is fidelity |<psi(x1)|psi(x2)>|^2
                    probs = self.kernel_qnode(X1_norm[i], X2_norm[j])
                    K[i, j] = probs[0]
        return K

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        start_time = time.time()
        # Compress clinical features to match quantum register size
        self.X_train_reduced = self.pca.fit_transform(X_train)

        # Compute training Gram matrix
        logger.info(f"Computing Quantum Kernel Gram matrix for {len(X_train)} samples...")
        K_train = self._compute_gram_matrix(self.X_train_reduced, self.X_train_reduced)

        base_svc = SVC(
            C=self.C,
            kernel="precomputed",
            class_weight="balanced",
            random_state=self.random_state
        )
        self.classifier = CalibratedClassifierCV(base_svc, cv=3)
        self.classifier.fit(K_train, y_train)
        self.training_time = time.time() - start_time
        logger.info(f"Quantum Kernel SVM trained in {self.training_time:.2f} seconds.")
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if self.classifier is None or self.X_train_reduced is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_test_reduced = self.pca.transform(X_test)
        K_test = self._compute_gram_matrix(X_test_reduced, self.X_train_reduced)
        return self.classifier.predict_proba(K_test)[:, 1]
