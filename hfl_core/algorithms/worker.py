"""
worker.py
Các thực thể mô phỏng: SensorWorker, FogNode, SurfaceGateway.
Quản lý trạng thái nội bộ (pin, model cục bộ, error buffer).
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from hfl_core.models.autoencoder import get_model_state_dict_copy
from hfl_core.algorithms.local_trainer import local_sgd
from hfl_core.knowledge_compression.topk_sparsification import TopKCompressor
from hfl_core.knowledge_compression.int8_quantization import SparseINT8Payload
from hfl_core.algorithms.aggregator import fedavg_intra_cluster, fedavg_global
from hfl_core.algorithms.hfl_rules import (
    should_cooperate, find_coop_partner, blend_state_dicts, compute_q1_fog_distance,
)


class SensorWorker:
    """
    AUV cảm biến tại biên. Thực hiện:
    1. Trừ pin sau mỗi round (death logic).
    2. Huấn luyện cục bộ (FedAvg/FedProx).
    3. Nén Top-K + INT8.
    """
    def __init__(
        self,
        sensor_id: int,
        dataloader: DataLoader,
        model_template: nn.Module,
        battery_init: float = 500.0,
        rho_s: float = 0.05,
    ):
        self.sensor_id = sensor_id
        self.dataloader = dataloader
        self.battery = battery_init
        self.alive = True
        self.n_samples = len(dataloader.dataset)

        # Cấu trúc model (copy để huấn luyện cục bộ)
        self.model = copy.deepcopy(model_template)
        
        # Compressor với error buffer riêng
        total_params = sum(p.numel() for p in model_template.parameters())
        self.compressor = TopKCompressor(total_params=total_params, rho_s=rho_s)

    def train_and_compress(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        global_model: Optional[nn.Module] = None,
        epochs: int = 5,
        lr: float = 0.01,
        mu: float = 0.0,
        device: str = 'cpu',
    ) -> Tuple[Optional[SparseINT8Payload], float]:
        """
        Huấn luyện và nén gửi lên Fog.
        Nếu pin <= 0, trả về None, 0.0.
        """
        if not self.alive:
            return None, 0.0

        # Load global weights
        self.model.load_state_dict(global_state_dict)

        # Local SGD
        delta_theta, avg_loss = local_sgd(
            model=self.model,
            dataloader=self.dataloader,
            epochs=epochs,
            lr=lr,
            global_model=global_model,
            mu=mu,
            device=device,
        )

        # Nén Top-K + Error Feedback
        topk_indices, topk_values = self.compressor.compress(delta_theta)

        # INT8 Quantize
        payload = SparseINT8Payload(
            topk_indices=topk_indices,
            topk_values=topk_values,
            total_params=self.compressor.total_params,
        )
        return payload, avg_loss

    def deduct_battery(self, energy_joules: float, min_battery: float = 0.0):
        """Khấu trừ pin và check death."""
        self.battery -= energy_joules
        if self.battery <= min_battery:
            self.alive = False


class FogNode:
    """
    Trạm Fog trung gian. Thực hiện:
    1. Gom payload từ cụm, giải nén.
    2. FedAvg nội cụm.
    3. Quyết định HFL-Selective.
    """
    def __init__(self, fog_id: int, cluster_members: List[int], model_template: nn.Module):
        self.fog_id = fog_id
        self.cluster_members = cluster_members
        self.cluster_size = len(cluster_members)
        self.model_template = model_template

        # Trạng thái sau khi intra-cluster agg
        self.intra_state_dict = None
        # Trạng thái cuối cùng gửi lên Gateway
        self.final_state_dict = None

    def aggregate_intra_cluster(
        self,
        global_state_dict: Dict[str, torch.Tensor],
        payloads: Dict[int, SparseINT8Payload],
        sensor_n_samples: Dict[int, int],
    ):
        """Giải nén và FedAvg nội cụm."""
        client_deltas = []
        for sensor_id, payload in payloads.items():
            if sensor_id in self.cluster_members:
                dense_delta = payload.decompress()
                n_i = sensor_n_samples[sensor_id]
                client_deltas.append((dense_delta, n_i))

        self.intra_state_dict = fedavg_intra_cluster(
            global_state_dict=global_state_dict,
            client_deltas=client_deltas,
            model_template=self.model_template,
        )
        # Tạm thời final = intra (để fallback nếu ko coop)
        self.final_state_dict = copy.deepcopy(self.intra_state_dict)

    def cooperate(
        self,
        rule: str,
        mean_cluster_size: float,
        cluster_sizes: Dict[int, int],
        feasibility_graph: Dict,
        all_fogs_intra_states: Dict[int, Dict[str, torch.Tensor]],
        q1_distance: Optional[float] = None,
    ) -> Tuple[bool, Optional[int]]:
        """
        Kiểm tra và thực hiện giao tiếp liên cụm.

        Alpha theo paper:
            - HFL-Nearest  : α = 0.7 (fixed mixing weights (0.7, 0.3) — paper Sec. V-B)
            - HFL-Selective: α = 0.8 (θ̃_m = 0.8·θ_m + 0.2·θ_j — Eq. 29)

        Returns:
            (did_cooperate, partner_id) — partner_id dùng để tính E_f2f trong simulator.
        """
        if rule == 'nocoop':
            return False, None

        # HFL-Selective: kiểm tra điều kiện Eq. 28 trước
        if rule == 'selective' and not should_cooperate(self.cluster_size, mean_cluster_size):
            return False, None

        # Chọn alpha theo rule (paper Sec. V-B)
        alpha = 0.7 if rule == 'nearest' else 0.8

        # HFL-Selective truyền q1_distance để lọc thêm, HFL-Nearest thì không
        dist_filter = q1_distance if rule == 'selective' else None

        # Cả 'nearest' và 'selective' (đã pass điều kiện) đều tìm neighbor
        partner_id = find_coop_partner(
            self.fog_id, cluster_sizes, feasibility_graph,
            q1_distance=dist_filter,
        )

        if partner_id is not None and partner_id in all_fogs_intra_states:
            neighbor_sd = all_fogs_intra_states[partner_id]
            self.final_state_dict = blend_state_dicts(
                self.intra_state_dict, neighbor_sd, alpha=alpha
            )
            return True, partner_id
        return False, None


class SurfaceGateway:
    """Trạm mặt nước tổng hợp toàn cục."""
    def __init__(self, model_template: nn.Module):
        self.global_state_dict = get_model_state_dict_copy(model_template)

    def aggregate_global(
        self,
        fog_final_states: Dict[int, Dict[str, torch.Tensor]],
        cluster_total_samples: Dict[int, int],
    ):
        """Tổng hợp từ các Fog."""
        states_list = []
        samples_list = []
        for fog_id, state in fog_final_states.items():
            states_list.append(state)
            samples_list.append(cluster_total_samples[fog_id])

        if states_list:
            self.global_state_dict = fedavg_global(states_list, samples_list)

    def aggregate_global_flat(
        self,
        payloads: Dict[int, SparseINT8Payload],
        sensor_n_samples: Dict[int, int],
        model_template: nn.Module,
    ):
        """Tổng hợp mạng phẳng (FedAvg/FedProx) trực tiếp từ Sensor."""
        client_deltas = []
        for sensor_id, payload in payloads.items():
            dense_delta = payload.decompress()
            n_i = sensor_n_samples[sensor_id]
            client_deltas.append((dense_delta, n_i))

        self.global_state_dict = fedavg_intra_cluster(
            global_state_dict=self.global_state_dict,
            client_deltas=client_deltas,
            model_template=model_template,
        )
