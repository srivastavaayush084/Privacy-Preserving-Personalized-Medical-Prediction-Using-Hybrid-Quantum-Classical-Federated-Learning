import os
import gzip
import pandas as pd
import numpy as np

# 1. Create directory structure
dirs = [
    'data/raw',
    'data/processed',
    'data/metadata',
    'data_audit',
    'configs',
    'src/data',
    'src/models/classical',
    'src/models/quantum',
    'src/federated',
    'src/personalization',
    'src/privacy',
    'src/evaluation',
    'src/inference',
    'experiments/classical_baselines',
    'experiments/quantum_models',
    'experiments/federated_learning',
    'experiments/personalization',
    'experiments/privacy',
    'experiments/robustness',
    'experiments/ablation',
    'checkpoints',
    'results/metrics',
    'results/figures',
    'results/predictions',
    'results/reports',
    'dashboard',
    'tests',
    'research',
    'scripts'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
print("Created directory structure successfully.")

