import gc
import copy
from typing import Dict, List, Optional, Tuple, Any

class BaseWorker:
    """Quản lý pin và vòng đời của AUV — CHUNG cho 1D và 2D."""
    def __init__(self, auv_id: int, battery_init: float = 2000.0):
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
    def __init__(self, relay_id: int, cluster_members: List[int],
                 battery_init: float = 2000.0):
        self.relay_id = relay_id
        self.cluster_members = list(cluster_members)
        self.cluster_size = len(cluster_members)
        self.intra_state_dict = None
        self.final_state_dict = None
        self.battery = battery_init
        self.alive = True

    def deduct_battery(self, energy_joules: float, min_battery: float = 50.0):
        """Khấu trừ pin Relay và check death."""
        self.battery -= energy_joules
        if self.battery <= min_battery:
            self.alive = False

    def cooperate(
        self,
        rule: str,
        mean_cluster_size: float,
        cluster_sizes: Dict[int, int],
        feasibility_graph: Dict,
        all_relays_intra_states: Dict[int, Any],
        q1_distance: Optional[float] = None,
        transport_state=None,
    ) -> Tuple[bool, Optional[int]]:
        """Logic HFL-Selective/Nearest."""
        from federated_core.hfl_rules import should_cooperate, find_coop_partner, blend_state_dicts
        if rule == 'nocoop':
            return False, None

        effective_cluster_size = cluster_sizes.get(self.relay_id, self.cluster_size)
        if rule == 'selective' and not should_cooperate(effective_cluster_size, mean_cluster_size):
            return False, None

        from config.settings import fed_cfg
        neighbor_weight = (
            fed_cfg.COOP_NEIGHBOR_WEIGHT_NEAREST
            if rule == 'nearest'
            else fed_cfg.COOP_NEIGHBOR_WEIGHT_SELECTIVE
        )
        alpha = 1.0 - float(neighbor_weight)
        dist_filter = q1_distance if rule == 'selective' else None

        partner_id = find_coop_partner(
            self.relay_id, cluster_sizes, feasibility_graph,
            q1_distance=dist_filter,
            require_larger_cluster=(rule == 'selective'),
        )

        if partner_id is not None and partner_id in all_relays_intra_states:
            neighbor_sd = all_relays_intra_states[partner_id]
            if transport_state is not None:
                neighbor_sd = transport_state(neighbor_sd)
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

    def aggregate_global(
        self,
        relay_final_states: Dict[int, Any],
        cluster_total_samples: Dict[int, int],
        lora_aggregation: str = "svd",
        server_mix_beta: float = 1.0,
    ):
        """fedavg_global"""
        from federated_core.aggregator import fedavg_global, mix_server_state
        states_list = []
        samples_list = []
        delta_c_list = []
        scaffold_client_counts = []
        
        for relay_id, state in relay_final_states.items():
            if isinstance(state, dict) and '__scaffold_delta_c__' in state:
                delta_c_list.append(state['__scaffold_delta_c__'])
                scaffold_client_counts.append(int(state.get('__scaffold_client_count__', 1)))
                state = {
                    key: value
                    for key, value in state.items()
                    if key not in {'__scaffold_delta_c__', '__scaffold_client_count__'}
                }
            states_list.append(state)
            samples_list.append(cluster_total_samples[relay_id])

        if states_list:
            old_global_state = copy.deepcopy(self.global_state_dict)
            aggregated_state = fedavg_global(
                states_list,
                samples_list,
                lora_aggregation=lora_aggregation,
            )
            self.global_state_dict = mix_server_state(
                old_global_state,
                aggregated_state,
                beta=server_mix_beta,
                lora_aggregation=lora_aggregation,
            )
            
        if delta_c_list:
            total_clients = sum(scaffold_client_counts)
            weights = [count / total_clients for count in scaffold_client_counts]
            delta_c_agg = {}
            for k in delta_c_list[0].keys():
                delta_c_agg[k] = sum(d[k] * w for d, w in zip(delta_c_list, weights))
            self.global_state_dict['__scaffold_delta_c__'] = delta_c_agg
