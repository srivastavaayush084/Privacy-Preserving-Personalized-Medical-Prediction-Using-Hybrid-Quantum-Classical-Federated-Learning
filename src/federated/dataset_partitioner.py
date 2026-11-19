"""
Dataset partitioner for federated hospital clients.
Assigns patients and stays strictly according to their hospital systems,
preserving complete data locality and natural non-IID distribution.
"""
import logging
import os
from typing import Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import torch.nn as nn

from src.federated.client import HospitalClient

logger = logging.getLogger(__name__)


def partition_federated_clients(
    model_factory: Callable[[], nn.Module],
    data_dir: str = "data/processed",
    num_consortiums: int = 5,
    device: str = "cpu"
) -> Tuple[Dict[str, HospitalClient], Tuple[np.ndarray, np.ndarray]]:
    """Partitions the eICU processed arrays into hospital consortium clients.
    
    Returns:
        Tuple of:
            - Dict mapping client_id to HospitalClient instance
            - Global centralized test holdout (X_test, y_test)
    """
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))

    train_meta = pd.read_csv(os.path.join(data_dir, "train_meta.csv"))
    val_meta = pd.read_csv(os.path.join(data_dir, "val_meta.csv"))
    test_meta = pd.read_csv(os.path.join(data_dir, "test_meta.csv"))

    consortium_names = [
        "Academic Medical Center Network",
        "Regional Health Alliance East",
        "Community Healthcare System West",
        "Metropolitan Critical Care Network",
        "Rural & Suburban Health System"
    ]

    # Map hospitalid to consortium [0 .. num_consortiums - 1]
    train_meta["consortium_idx"] = train_meta["hospitalid"] % num_consortiums
    val_meta["consortium_idx"] = val_meta["hospitalid"] % num_consortiums
    test_meta["consortium_idx"] = test_meta["hospitalid"] % num_consortiums

    clients: Dict[str, HospitalClient] = {}

    for idx in range(num_consortiums):
        client_id = f"Hospital_Consortium_{idx + 1}"
        hospital_name = consortium_names[idx] if idx < len(consortium_names) else f"Consortium {idx + 1}"

        # Indices for this client
        tr_mask = (train_meta["consortium_idx"] == idx).values
        va_mask = (val_meta["consortium_idx"] == idx).values
        te_mask = (test_meta["consortium_idx"] == idx).values

        client_hosp_ids = sorted(train_meta.loc[tr_mask, "hospitalid"].unique().tolist())

        client = HospitalClient(
            client_id=client_id,
            hospital_name=hospital_name,
            hospital_ids=client_hosp_ids,
            X_train=X_train[tr_mask],
            y_train=y_train[tr_mask],
            X_val=X_val[va_mask],
            y_val=y_val[va_mask],
            X_test=X_test[te_mask],
            y_test=y_test[te_mask],
            model_factory=model_factory,
            device=device
        )
        clients[client_id] = client

    logger.info(f"Successfully partitioned {len(clients)} federated hospital clients.")
    return clients, (X_test, y_test)
