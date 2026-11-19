"""
Federated communication protocols and payload telemetry.
Ensures zero exchange of raw patient records across hospital boundaries.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pickle


@dataclass
class ClientRegistrationMessage:
    """Message sent by hospital client to coordinator upon joining."""
    client_id: str
    hospital_name: str
    hospital_ids: List[int]
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int
    mortality_rate: float


@dataclass
class GlobalBroadcastMessage:
    """Message sent from coordinator to hospital clients each round."""
    round_idx: int
    global_parameters: Dict[str, Any]
    strategy_config: Dict[str, Any]


@dataclass
class ClientUpdateMessage:
    """Message sent from hospital client to coordinator after local training.
    Strictly contains model parameter weights or deltas, never raw patient data.
    """
    client_id: str
    round_idx: int
    num_samples: int
    model_state_dict: Dict[str, Any]
    train_loss: float
    local_val_metrics: Dict[str, float]
    local_test_metrics: Dict[str, float]


class CommunicationTracker:
    """Tracks network bandwidth and payload exchange between clients and coordinator."""

    def __init__(self):
        self.bytes_sent_server_to_clients = 0
        self.bytes_sent_clients_to_server = 0
        self.round_history: List[Dict[str, Any]] = []

    def record_downlink(self, round_idx: int, message: GlobalBroadcastMessage, num_recipients: int):
        payload_bytes = len(pickle.dumps(message.global_parameters)) * num_recipients
        self.bytes_sent_server_to_clients += payload_bytes

    def record_uplink(self, round_idx: int, message: ClientUpdateMessage):
        payload_bytes = len(pickle.dumps(message.model_state_dict))
        self.bytes_sent_clients_to_server += payload_bytes

    def get_summary(self) -> Dict[str, float]:
        return {
            "total_downlink_mb": self.bytes_sent_server_to_clients / (1024 * 1024),
            "total_uplink_mb": self.bytes_sent_clients_to_server / (1024 * 1024),
            "total_bandwidth_mb": (self.bytes_sent_server_to_clients + self.bytes_sent_clients_to_server) / (1024 * 1024),
        }
