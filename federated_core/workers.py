import gc
import copy
from typing import Dict, List, Optional, Tuple, Any

class BaseWorker:
    """Quản lý pin và vòng đời của AUV — CHUNG cho 1D và 2D."""
    def __init__(self, auv_id: int, battery_init: float = 500.0):
        self.auv_id = auv_id
        self.battery = battery_init
        self.alive = True

    def deduct_battery(self, energy_joules: float, min_battery: float = 0.0):
        """Khấu trừ pin và check death."""
        self.battery -= energy_joules
        if self.battery <= min_battery:
            self.alive = False

    def train_and_get_payload(self, global_state, **kwargs):
        """Abstract — 1D trả SparseINT8Payload, 2D trả LoRA state_dict (sau Lazy Filter)."""
        raise NotImplementedError


class BaseRelayNode:
    """Tổng hợp nội cụm và hợp tác liên cụm — CHUNG."""
    def __init__(self, relay_id: int, cluster_members: List[int]):
        self.relay_id = relay_id
        self.cluster_members = list(cluster_members)
        self.cluster_size = len(cluster_members)
        self.intra_state_dict = None
        self.final_state_dict = None

    def cooperate(
        self,
        rule: str,
        mean_cluster_size: float,
        cluster_sizes: Dict[int, int],
        feasibility_graph: Dict,
        all_relays_intra_states: Dict[int, Any],
        q1_distance: Optional[float] = None,
    ) -> Tuple[bool, Optional[int]]:
        """Logic HFL-Selective/Nearest."""
        from federated_core.hfl_rules import should_cooperate, find_coop_partner, blend_state_dicts
        if rule == 'nocoop':
            return False, None

        if rule == 'selective' and not should_cooperate(self.cluster_size, mean_cluster_size):
            return False, None

        alpha = 0.7 if rule == 'nearest' else 0.8
        dist_filter = q1_distance if rule == 'selective' else None

        partner_id = find_coop_partner(
            self.relay_id, cluster_sizes, feasibility_graph,
            q1_distance=dist_filter,
        )

        if partner_id is not None and partner_id in all_relays_intra_states:
            neighbor_sd = all_relays_intra_states[partner_id]
            self.final_state_dict = blend_state_dicts(
                self.intra_state_dict, neighbor_sd, alpha=alpha
            )
            del neighbor_sd
            gc.collect()
            return True, partner_id
        return False, None

    def aggregate_intra_cluster(self, global_state_dict, payloads, auv_n_samples, **kwargs):
        """Abstract — 1D dùng delta decompress, 2D dùng state_dict trực tiếp."""
        raise NotImplementedError


class BaseGateway:
    """Tổng hợp toàn cục — CHUNG."""
    def __init__(self, initial_state):
        # initial_state can be state_dict
        import copy
        self.global_state_dict = copy.deepcopy(initial_state)

    def aggregate_global(self, relay_final_states: Dict[int, Any], cluster_total_samples: Dict[int, int]):
        """fedavg_global"""
        from federated_core.aggregator import fedavg_global
        states_list = []
        samples_list = []
        for relay_id, state in relay_final_states.items():
            states_list.append(state)
            samples_list.append(cluster_total_samples[relay_id])

        if states_list:
            self.global_state_dict = fedavg_global(states_list, samples_list)
