"""
simulator.py
Vòng lặp mô phỏng chính cho Scenario 1: Tái hiện Omeke et al. 2026.
Tích hợp: Logic Tử vong pin (Device Death), Flat Topology Filtering, HFL-Selective.
"""

import copy
import torch
import numpy as np
from typing import Dict, List, Tuple

from config.settings import FedKDLConfig, NetworkConfig, AcousticChannelConfig, EnergyConfig
from physics_models.topology import (
    Topology3D, build_feasibility_graph, nearest_feasible_association, flat_topology_association, build_clusters
)
from kdl_core.algorithms.knowledge_association import knowledge_aware_association
from physics_models.energy import e_tx, e_comp_simple
from kdl_core.algorithms.hfl_rules import compute_q1_fog_distance
from kdl_core.data.dataloader_1d import load_dataset, SlidingWindowDataset, non_iid_partition, make_client_loaders, make_val_loader
from kdl_core.models.autoencoder import SmallAutoencoder
from kdl_core.algorithms.worker import SensorWorker, FogNode, SurfaceGateway
from kdl_core.metrics import EnergyTracker, LatencyTracker, MetricsLogger, anomaly_threshold, point_adjusted_f1


class Scenario1Simulator:
    """
    Simulator cho tác vụ Anomaly Detection 1D.
    Hỗ trợ 2 baseline chính cho Scenario 1: 'hfl_selective' và 'fedprox'.
    """

    def __init__(self, net_cfg: NetworkConfig, ac_cfg: AcousticChannelConfig,
                 en_cfg: EnergyConfig, fed_cfg: FedKDLConfig,
                 baseline: str = 'hfl_selective', seed: int = 42):
        
        self.net_cfg = net_cfg
        self.ac_cfg = ac_cfg
        self.en_cfg = en_cfg
        self.fed_cfg = fed_cfg
        self.baseline = baseline
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"--- Khởi tạo Simulator: Baseline = {baseline} ---")
        
        # 1. Khởi tạo Topology & Feasibility Graph
        self.topology = Topology3D(net_cfg, ac_cfg, seed)
        self.G = build_feasibility_graph(self.topology, ac_cfg)
        
        # 3. Chuẩn bị Data
        print(f"Load dataset: {fed_cfg.DATASETS_1D[0]}...")
        train_ds, val_ds, total_sensors, train_labels = load_dataset(fed_cfg.DATASETS_1D[0])
        
        # Phân chia Non-IID cho train
        client_indices = non_iid_partition(train_ds, total_sensors, alpha=fed_cfg.NON_IID_ALPHA, seed=seed)
        self.client_loaders = make_client_loaders(train_ds, client_indices)
        self.val_loader = make_val_loader(val_ds)
        
        # DataLoader cho Centralised
        self.centralised_loader = make_val_loader(train_ds)  # Re-use val_loader func for full batching
        
        # Tạo Label Histograms cho D_joint EMD (Knowledge-Aware Association)
        sensor_hists = np.ones((total_sensors, 2)) * 0.5 # Default uniform
        for i, loader in self.client_loaders.items():
            if len(loader.dataset) > 0:
                y = np.array([train_labels[idx] for idx in loader.dataset.indices])
                counts = np.bincount(y, minlength=2)
                sensor_hists[i] = counts / max(1, counts.sum())
        fog_hists = np.ones((self.topology.M, 2)) * 0.5 # Uniform initialization for fogs

        # 2. Association rule (Chuyển xuống đây để dùng hists)
        if baseline in ['fedprox', 'fedavg']:
            self.association = flat_topology_association(self.topology, self.G)
        else: # 'hfl_selective', 'hfl_nearest', 'hfl_nocoop'
            self.association = knowledge_aware_association(
                self.topology, self.G, sensor_hists, fog_hists, beta=fed_cfg.BETA_EMD
            )
            
        self.clusters = build_clusters(self.association, self.topology.M)
        
        connected = len(self.association)
        print(f"Topology: {connected}/{total_sensors} sensors có liên kết khả thi.")
        if baseline not in ['fedprox', 'fedavg', 'centralised']:
            print(f"Cluster sizes: {[len(c) for c in self.clusters.values()]}")
        
        # 4. Khởi tạo Mô hình & Thực thể
        input_dim = train_ds.D
        self.model_template = SmallAutoencoder(input_dim).to(self.device)
        
        # Sensors
        self.sensors: Dict[int, SensorWorker] = {}
        for i in range(total_sensors):
            if i in self.association and i in self.client_loaders:
                # Chỉ tạo worker cho sensor khả thi
                self.sensors[i] = SensorWorker(
                    sensor_id=i,
                    dataloader=self.client_loaders[i],
                    model_template=self.model_template,
                    battery_init=en_cfg.E_INIT,
                    rho_s=0.05  # Top-K = 5%
                )
                
        # Fogs
        self.fogs: Dict[int, FogNode] = {}
        for m in range(self.topology.M):
            self.fogs[m] = FogNode(m, self.clusters[m], self.model_template)
            
        # Gateway
        self.gateway = SurfaceGateway(self.model_template)
        
        # Metrics trackers
        self.energy_tracker = EnergyTracker()
        self.latency_tracker = LatencyTracker(sound_speed=ac_cfg.SOUND_SPEED)
        self.metrics_logger = MetricsLogger()

        # Pre-compute Q1 fog-fog distance (dùng cho HFL-Selective filter — Eq. 29)
        self.q1_fog_distance = compute_q1_fog_distance(self.G)

        # Kích thước model fog (full-precision, dùng cho e_f2g và e_f2f — paper: fog exchange là full-precision)
        # L_g = d × 32 bits  (d ≈ 1350 params, 32-bit float)
        self.fog_model_bits = sum(p.numel() for p in self.model_template.parameters()) * 32
        
    def evaluate_global_model(self) -> Tuple[float, float, float]:
        """Đánh giá PA-F1 trên tập validation."""
        self.model_template.load_state_dict(self.gateway.global_state_dict)
        self.model_template.eval()
        
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                errors = self.model_template.reconstruction_error(x)
                all_errors.extend(errors.cpu().numpy())
                all_labels.extend(y.numpy())
                
        all_errors = np.array(all_errors)
        all_labels = np.array(all_labels)
        
        # Tính ngưỡng từ data bình thường (label == 0) trong val set
        normal_errors = all_errors[all_labels == 0]
        threshold = anomaly_threshold(normal_errors, percentile=99.0)
        
        pa_f1, prec, rec = point_adjusted_f1(all_labels, all_errors, threshold)
        return pa_f1, prec, rec

    def run(self, T_rounds: int):
        print("\n=== Bắt đầu Huấn luyện ===")
        
        if self.baseline == 'centralised':
            return self.run_centralised(T_rounds)
            
        for t in range(T_rounds):
            # Lọc các sensor còn sống
            alive_sensors = [s_id for s_id, s in self.sensors.items() if s.alive]
            participation = len(alive_sensors) / self.topology.N
            
            # --- Phase 1: Local Training + Compression ---
            payloads = {}
            e_s2f_total = 0.0
            e_comp_total = 0.0
            sensor_n_samples = {}
            
            for s_id in alive_sensors:
                sensor = self.sensors[s_id]
                mu = 0.01 if self.baseline == 'fedprox' else 0.0
                
                # Huấn luyện và nén
                res = sensor.train_and_compress(
                    global_state_dict=self.gateway.global_state_dict,
                    global_model=self.model_template,  # Để freeze param cho proximal term
                    epochs=self.fed_cfg.LOCAL_EPOCHS,
                    mu=mu,
                    device=self.device
                )
                
                if res is not None:
                    payload, avg_loss = res
                    payloads[s_id] = payload
                    sensor_n_samples[s_id] = sensor.n_samples
                    
                    # Tính chi phí phát từ sensor lên fog/gateway
                    target_id = self.association[s_id]
                    if self.baseline in ['fedprox', 'fedavg']:
                        link_key = ('sensor', s_id, 'gateway', 0)
                    else:
                        link_key = ('sensor', s_id, 'fog', target_id)
                        
                    if link_key in self.G:
                        link = self.G[link_key]
                        e_tx_cost = e_tx(payload.payload_bits, link.R_bps, link.SL_min, 
                                         self.en_cfg.ETA_EA, self.en_cfg.P_C_TX)
                    else:
                        e_tx_cost = 0.0
                        
                    e_comp_cost = e_comp_simple(self.fed_cfg.LOCAL_EPOCHS, self.en_cfg.E_COMP_EPOCH)
                    
                    # Trừ pin
                    sensor.deduct_battery(e_tx_cost + e_comp_cost)
                    e_s2f_total += e_tx_cost
                    e_comp_total += e_comp_cost

            # --- Phase 2: Fog Aggregation & Cooperation ---
            e_f2f_total = 0.0
            e_f2g_total = 0.0
            cooperation_partners: Dict[int, int] = {}  # fog_id → partner_id (cho latency)
            
            if self.baseline not in ['fedprox', 'fedavg']:
                # Intra-cluster
                for m, fog in self.fogs.items():
                    fog.aggregate_intra_cluster(self.gateway.global_state_dict, payloads, sensor_n_samples)
                    
                # HFL Cooperation
                cluster_sizes = {m: fog.cluster_size for m, fog in self.fogs.items()}
                mean_c = np.mean([s for s in cluster_sizes.values() if s > 0]) if any(s > 0 for s in cluster_sizes.values()) else 1.0
                all_intra = {m: fog.intra_state_dict for m, fog in self.fogs.items()}
                
                rule_map = {'hfl_selective': 'selective', 'hfl_nearest': 'nearest', 'hfl_nocoop': 'nocoop'}
                coop_rule = rule_map.get(self.baseline, 'nocoop')
                
                for m, fog in self.fogs.items():
                    did_coop, partner_id = fog.cooperate(
                        coop_rule, mean_c, cluster_sizes, self.G, all_intra,
                        q1_distance=self.q1_fog_distance,
                    )

                    # Ghi nhận partner để tính latency
                    if did_coop and partner_id is not None:
                        cooperation_partners[m] = partner_id

                    # Chi phí fog-to-fog: dùng e_tx() với link vật lý thực — Eq. 18
                    if did_coop and partner_id is not None:
                        key_fwd = ('fog', m, 'fog', partner_id)
                        key_bwd = ('fog', partner_id, 'fog', m)
                        f2f_key = key_fwd if key_fwd in self.G else key_bwd
                        if f2f_key in self.G:
                            f2f_link = self.G[f2f_key]
                            # Cả hai chiều: m nhận từ partner (partner phát)
                            e_f2f_total += e_tx(
                                self.fog_model_bits, f2f_link.R_bps, f2f_link.SL_min,
                                self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                            )

                    # Chi phí fog-to-gateway: full-precision model — Eq. 19
                    link_key = ('fog', m, 'gateway', 0)
                    if link_key in self.G:
                        link = self.G[link_key]
                        e_f2g_total += e_tx(
                            self.fog_model_bits, link.R_bps, link.SL_min,
                            self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                        )

            # --- Phase 3: Global Aggregation ---
            if self.baseline in ['fedprox', 'fedavg']:
                self.gateway.aggregate_global_flat(payloads, sensor_n_samples, self.model_template)
            else:
                fog_final = {m: fog.final_state_dict for m, fog in self.fogs.items() if fog.final_state_dict is not None}
                cluster_samples = {m: sum(sensor_n_samples.get(s_id, 0) for s_id in fog.cluster_members) for m, fog in self.fogs.items()}
                self.gateway.aggregate_global(fog_final, cluster_samples)

            # --- Concept Drift Checking ---
            if self.baseline in ['hfl_selective', 'hfl_nearest']:
                drift_detected = False
                for s_id in alive_sensors:
                    if self.sensors[s_id].drift_monitor.check_drift(s_id):
                        drift_detected = True
                        break
                
                if drift_detected:
                    print(f"[Round {t}] ⚠️ Báo động Concept Drift! Kích hoạt Re-clustering / Reset Error Buffers (Eq. 34).")
                    for s_id in alive_sensors:
                        self.sensors[s_id].compressor.reset_error_buffer()
                        self.sensors[s_id].drift_monitor.clear()
                
            # --- Phase 4: Logging & Evaluation ---
            self.energy_tracker.add_round(t, e_s2f_total, e_f2f_total, e_f2g_total, e_comp_total)

            # Latency Eq. 21: dùng payload bits trung bình của round này
            avg_payload_bits = (
                np.mean([p.payload_bits for p in payloads.values()]) if payloads else self.fog_model_bits
            )
            tau_round = self.latency_tracker.compute_round_latency(
                G=self.G,
                association={s: self.association[s] for s in alive_sensors if s in self.association},
                cooperation_partners=cooperation_partners,
                n_local_epochs=self.fed_cfg.LOCAL_EPOCHS,
                sensor_payload_bits=avg_payload_bits,
                fog_model_bits=self.fog_model_bits,
            )
            self.latency_tracker.add_round(t, tau_round)

            pa_f1, prec, rec = self.evaluate_global_model()
            
            self.metrics_logger.log(t, {
                'PA-F1': pa_f1,
                'Precision': prec,
                'Recall': rec,
                'Participation': participation,
                'Cumul_Energy': self.energy_tracker.cumulative_energy,
                'Tau_Round_s': tau_round,
            })
            
            self.metrics_logger.print_latest()
            
            if participation == 0:
                print(f"Mạng sụp đổ hoàn toàn tại vòng {t} do cạn pin (Tỷ lệ tham gia = 0%).")
                break
                
        print("\n=== Hoàn thành Huấn luyện ===")
        return (
            self.metrics_logger.get_dataframe(),
            self.energy_tracker.get_dataframe(),
            self.latency_tracker.get_dataframe(),
        )

    def run_centralised(self, T_rounds: int):
        """Huấn luyện tập trung (không qua mạng)."""
        optimizer = torch.optim.SGD(self.model_template.parameters(), lr=self.fed_cfg.LOCAL_LR)
        loss_fn = torch.nn.MSELoss()
        
        for t in range(T_rounds):
            self.model_template.train()
            # Simulate 'local epochs' effect by iterating E times per round
            for _ in range(self.fed_cfg.LOCAL_EPOCHS):
                for x, _ in self.centralised_loader:
                    x = x.to(self.device)
                    optimizer.zero_grad()
                    x_hat = self.model_template(x)
                    loss = loss_fn(x_hat, x)
                    loss.backward()
                    optimizer.step()
            
            self.gateway.global_state_dict = copy.deepcopy(self.model_template.state_dict())
            pa_f1, prec, rec = self.evaluate_global_model()
            
            self.metrics_logger.log(t, {
                'PA-F1': pa_f1,
                'Precision': prec,
                'Recall': rec,
                'Participation': 1.0,
                'Cumul_Energy': 0.0,
                'Tau_Round_s': 0.0,  # Centralised: không có trễ mạng
            })
            self.metrics_logger.print_latest()
            
        print("\n=== Hoàn thành Huấn luyện (Centralised) ===")
        import pandas as pd
        return (
            self.metrics_logger.get_dataframe(),
            pd.DataFrame(),                         # energy_df rỗng
            self.latency_tracker.get_dataframe(),   # latency_df rỗng
        )
