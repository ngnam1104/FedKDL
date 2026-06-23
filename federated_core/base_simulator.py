import gc
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from federated_core.metrics import (
    EnergyTracker,
    LatencyTracker,
    MetricsLogger,
    physical_joint_cost,
)
from federated_core.hfl_rules import compute_mean_cluster_size, compute_q1_relay_distance
try:
    from tasks.detection_2d.baselines import (
        BASELINE_CONFIGS as DETECTION_BASELINE_CONFIGS,
        parse_baseline_config as parse_2d_baseline_config,
    )
except Exception:
    DETECTION_BASELINE_CONFIGS = {}
    parse_2d_baseline_config = None

class BaseSimulator(ABC):
    """
    Khung mô phỏng chung cho 1D và 2D.
    Dùng Template Method Pattern: các class kế thừa override _process_auv và evaluate.
    """
    def __init__(
        self,
        topo_path: str,
        baseline: str,
        device: str = "cpu",
    ):
        self.device = device
        self.topo_path = topo_path
        self.baseline = baseline
        
        self.net_cfg = network_cfg
        self.ac_cfg = acoustic_cfg
        self.en_cfg = energy_cfg
        self.fed_cfg = fed_cfg

        self._load_environment()

        self.energy_tracker = EnergyTracker()
        self.latency_tracker = LatencyTracker(
            sound_speed=self.ac_cfg.SOUND_SPEED,
        )
        self.metrics_logger = MetricsLogger()
        
        # Abstract properties that should be initialized by subclasses
        self.auvs = {}
        self.relays = {}
        self.gateway = None
        self.task_key = "1D" # default, override in subclass

    def _load_environment(self):
        from utils.env_manager import EnvironmentManager
        from physics_models.topology import Topology3D
        
        topo = EnvironmentManager.load_topology(self.topo_path)
        
        # Đồng bộ cấu hình với topology thực tế trong file pkl để tránh lỗi mismatch M_RELAYS
        self.net_cfg.N_AUVS = topo.N
        self.net_cfg.M_RELAYS = topo.M
        
        # Phục hồi lại đối tượng Topology3D để có thể gọi hàm step_mobile_auvs
        self.topology = Topology3D(self.net_cfg, self.ac_cfg, seed=topo.seed)
        # Khôi phục vị trí y nguyên từ file pkl
        self.topology.auv_positions = topo.auv_positions
        self.topology.relay_positions = topo.relay_positions
        self.topology.gateway_position = topo.gateway_position
        
        self.N_actual = self.topology.N
        self.auv_positions = self.topology.auv_positions
        self.relay_positions = self.topology.relay_positions
        self.gateway_position = self.topology.gateway_position
        self.G = EnvironmentManager.restore_graph(topo)
        from physics_models.topology import gateway_disconnected_relays
        missing_gateway_relays = gateway_disconnected_relays(self.topology, self.G)
        if missing_gateway_relays:
            print(
                "[Warning] Relay(s) without a feasible gateway uplink: "
                f"{missing_gateway_relays}. Their R2G transmission is skipped."
            )
        
        if self.baseline in DETECTION_BASELINE_CONFIGS:
            self.is_flat = not parse_2d_baseline_config(self.baseline).hfl
        else:
            self.is_flat = self.baseline in ['fedavg', 'fedprox', 'fedkd', 'centralized']
        
        if self.is_flat:
            self.association = topo.flat_association
            self.clusters = {0: list(range(topo.N))}
        else:
            self.association = topo.hfl_association
            self.clusters = topo.clusters

    def get_flop_multiplier(self) -> float:
        """Cho phép các lớp con (ví dụ 2D Simulator) định nghĩa lại Flop Multiplier dựa theo logic của Local KD hay LoRA."""
        return self.fed_cfg.FLOP_MULTIPLIER[self.task_key]

    @abstractmethod
    def _process_auv(self, s_id: int) -> Tuple[int, Any, float, int, float, float]:
        """
        Huấn luyện và nén tại một auv.
        Returns:
            s_id, payload (hoặc state_dict), avg_loss, n_samples, e_tx, e_comp
        """
        pass

    @abstractmethod
    def _aggregate_intra_relay(self, m: int, relay, payloads, auv_n_samples) -> float:
        """
        Nội cụm + tính e_r2r.
        Returns: e_r2r_cost
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Đánh giá model trên tập test."""
        pass
        
    @abstractmethod
    def _compute_payload_bits(self, payloads: Dict) -> float:
        """Tính trung bình kích thước payload."""
        pass

    @abstractmethod
    def _compute_relay_model_bits(self) -> float:
        """Tính kích thước model relay (bits)."""
        pass

    def _payload_to_bits(self, payload: Any) -> float:
        """Return physical payload size in bits for sparse, bytes, and tensor payloads."""
        if payload is None:
            return 0.0
        if hasattr(payload, 'payload_bits'):
            return float(payload.payload_bits)
        if isinstance(payload, (bytes, bytearray)):
            return float(len(payload) * 8)
        if torch.is_tensor(payload):
            return float(payload.numel() * payload.element_size() * 8)
        if isinstance(payload, dict):
            return sum(self._payload_to_bits(item) for item in payload.values())
        return 0.0

    def _gateway_knowledge_distillation(self):
        """
        Hook: Gateway-side Knowledge Distillation sau Global Aggregation.

        Mặc định: no-op (dùng cho Simulator1D hoặc baseline không dùng KD).
        Simulator2D (baseline='fedkdl') sẽ override để chạy Teacher KD tại Tier 3.
        """
        pass

    def _transport_relay_state(self, state, relay_id=None):
        """Hook for task-specific R2R/R2G serialization and reconstruction."""
        return state

    def run(self, T_rounds: int, baseline: str = None) -> list:
        if baseline is not None:
            self.baseline = baseline
        import concurrent.futures
        
        if getattr(self.fed_cfg, 'LOG_ROUND_TOPOLOGY', False):
            print("\n" + "="*60)
            print("[*] CLUSTER TOPOLOGY INFO:")
            if self.is_flat:
                print("    (Flat Topology: AUVs connect directly or via relays to Gateway)")
            for relay_id, relay in self.relays.items():
                print(f"    - Relay {relay_id} manages auvs: {relay.cluster_members}")
            print("="*60 + "\n")
        
        cumulative_payload = 0.0
        cumulative_joint_cost = 0.0  # Σ full objective cost^t
        cumulative_physical_joint_cost = 0.0  # Σ physical cost^t
        
        import math
        initial_lr = self.fed_cfg.LOCAL_LR

        if (
            self.task_key == "2D"
            and getattr(self.fed_cfg, 'PREWARM_YOLO_LABEL_CACHE', False)
            and hasattr(self, '_pre_warm_dataset_cache')
        ):
            self._pre_warm_dataset_cache()

        pending_mobility_metrics = {
            'e_move': 0.0,
            'e_move_cumul': 0.0,
            'avg_move_m': 0.0,
            'max_move_m': 0.0,
            'avg_speed_mps': 0.0,
            'mobility_recluster_changes': 0,
            'mobility_recluster_rate': 0.0,
            'mobility_connected_after': len(self.association),
        }
        movement_energy_cumul = 0.0

        for t in range(1, T_rounds + 1):
            self.current_round = t
            # Tính toán Cosine Annealing Learning Rate cho toàn bộ quá trình FL
            progress = (t - 1) / max(1, T_rounds - 1)
            # LRF (Learning Rate Fraction) = 0.1 (warm restart floor)
            self.current_lr = initial_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.task_key == "2D":
                self.current_round = t  # Track current round for KD skipping
                print(f"\n{'='*60}")
                print(f"|  [Simulator] BẮT ĐẦU VÒNG {t}/{T_rounds}  |")
                print(f"{'='*60}\n")
                print(
                    f"[*] LR Schedule | round={t}/{T_rounds} | "
                    f"base_lr={initial_lr:.6f} -> current_lr={self.current_lr:.8f}"
                )
                if getattr(self.fed_cfg, 'LOG_ROUND_TOPOLOGY', False):
                    print("\n" + "="*60)
                    print("[*] CLUSTER TOPOLOGY INFO:")
                    if self.is_flat:
                        print("    (Flat Topology: AUVs connect directly or via relays to Gateway)")
                    for relay_id, relay in self.relays.items():
                        auv_details = []
                        for sid in relay.cluster_members:
                            n_imgs = self.auvs[sid].n_samples if sid in self.auvs else '?'
                            auv_details.append(f"{sid} ({n_imgs} imgs)")
                        print(f"    - Relay {relay_id} manages auvs: [{', '.join(auv_details)}]")
                    print("="*60 + "\n")
            else:
                print(
                    f"   -> [FL Simulator 1D] Round {t}/{T_rounds} | "
                    f"current_lr={self.current_lr:.8f}",
                    flush=True,
                )

            # --- [NEW] Logging Trajectories to File (Trước khi bắt đầu vòng FL) ---
            if self.task_key == "2D" and getattr(self.fed_cfg, 'LOG_TRAJECTORIES', False):
                import os
                alpha_str = str(self.alpha).replace('.', 'p') if hasattr(self, 'alpha') else "unknown"
                filename = f"auv_trajectories_N{self.N_actual}_{self.baseline}_a{alpha_str}.txt"
                traj_log_path = os.path.join("results", "train_logs", "kdl", filename)
                os.makedirs(os.path.dirname(traj_log_path), exist_ok=True)
                mode = 'w' if t == 1 else 'a'
                with open(traj_log_path, mode, encoding='utf-8') as f:
                    if t == 1 and hasattr(self, 'data_part') and hasattr(self.data_part, 'log_text') and self.data_part.log_text:
                        f.write("============================================================\n")
                        f.write("DATA PARTITIONING SUMMARY\n")
                        f.write("============================================================\n")
                        f.write(self.data_part.log_text + "\n\n")

                    f.write(f"--- Round {t} ---\n")
                    for s_id in range(self.topology.N):
                        pos = self.topology.auv_positions[s_id]
                        assoc = self.association.get(s_id, -1)
                        
                        counts_str = ""
                        if hasattr(self, 'auv_label_counts') and self.auv_label_counts is not None:
                            counts = self.auv_label_counts[s_id]
                            counts_str = " | Counts: [" + ", ".join([f"{int(x):3d}" for x in counts]) + "]"
                            
                        f.write(f"AUV {s_id:2d}: X={pos[0]:.0f}, Y={pos[1]:.0f}, Z={pos[2]:.0f} | Relay: {assoc:2d}{counts_str}\n")
                    
                    if hasattr(self, 'auv_label_counts') and self.auv_label_counts is not None:
                        relay_summaries = {}
                        relay_positions = {}
                        for s_id, r_id in self.association.items():
                            if r_id not in relay_summaries:
                                relay_summaries[r_id] = np.zeros(len(self.auv_label_counts[s_id]), dtype=np.float32)
                                relay_positions[r_id] = self.topology.relay_positions[r_id]
                            relay_summaries[r_id] += self.auv_label_counts[s_id]
                        if relay_summaries:
                            f.write("\n--- Relay Summaries ---\n")
                            for r_id in sorted(relay_summaries.keys()):
                                counts = relay_summaries[r_id]
                                pos = relay_positions[r_id]
                                counts_str = " | Total Counts: [" + ", ".join([f"{int(x):4d}" for x in counts]) + "]"
                                f.write(f"Relay {r_id:2d}: X={pos[0]:.0f}, Y={pos[1]:.0f}, Z={pos[2]:.0f}{counts_str}\n")
                    f.write("\n")
            # --- Phase 1: AUV Tier ---
            alive_auvs = [
                s.auv_id
                for s in self.auvs.values()
                if s.alive and s.auv_id in self.association
            ]
            dead_auvs = [s.auv_id for s in self.auvs.values() if not s.alive]
            missing_auvs = [i for i in range(self.N_actual) if i not in self.association]

            if self.task_key == "2D":
                if missing_auvs:
                    missing_details = [f"{sid} ({self.auvs[sid].n_samples if sid in self.auvs else '?'} imgs)" for sid in missing_auvs]
                    print(f"[!] BỎ QUA {len(missing_auvs)} AUVS (Out of Range / Mất kết nối): [{', '.join(missing_details)}]")
                if dead_auvs:
                    dead_details = [f"{sid} ({self.auvs[sid].n_samples if sid in self.auvs else '?'} imgs)" for sid in dead_auvs]
                    print(f"[!] BỎ QUA {len(dead_auvs)} AUVS (Đã chết / Hết pin): [{', '.join(dead_details)}]")
                print(f"[*] AUVS ĐANG HOẠT ĐỘNG ({len(alive_auvs)}): {alive_auvs}\n")
            else:
                total = self.N_actual
                connected = len(alive_auvs)
                out_of_range = len(missing_auvs)
                dead = len(dead_auvs)
                participation = connected / total if total > 0 else 0
                print(f"   -> [1D Round {t}/{T_rounds}] Active={connected}/{total} ({participation:.0%}) | OOR={out_of_range} | Dead={dead}", flush=True)

            if not alive_auvs:
                print(f"[Simulator] All auvs depleted at round {t}. Stopping.")
                break

            payloads = {}
            auv_n_samples = {}
            avg_losses = []
            auv_local_metrics = {}

            
            e_a2r_total = 0.0
            e_comp_total = 0.0

            if self.task_key == "2D":
                # Chạy tuần tự cho YOLO 2D vì Ultralytics trainer gây deadlock/crash nếu chạy đa luồng CUDA
                for s_id in alive_auvs:
                    sid, payload, loss, n_samp, e_tx_cost, e_comp_cost, local_metrics = self._process_auv(s_id)
                    if payload is not None:
                        payloads[sid] = payload
                        auv_n_samples[sid] = n_samp
                        avg_losses.append(loss)
                        if local_metrics:
                            auv_local_metrics[sid] = local_metrics
                        self.auvs[sid].deduct_battery(e_tx_cost + e_comp_cost, min_battery=self.en_cfg.E_MIN)
                    e_a2r_total += e_tx_cost
                    e_comp_total += e_comp_cost
            else:
                import os
                torch.set_num_threads(1) # RẤT QUAN TRỌNG: Ngăn chặn PyTorch sinh thêm luồng ngầm gây chết CPU
                max_w = 4 if self.device == 'cuda' else (os.cpu_count() or 8)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                    futures = {executor.submit(self._process_auv, s_id): s_id for s_id in alive_auvs}
                    for future in concurrent.futures.as_completed(futures):
                        sid, payload, loss, n_samp, e_tx_cost, e_comp_cost, local_metrics = future.result()
                        
                        if payload is not None:
                            payloads[sid] = payload
                            auv_n_samples[sid] = n_samp
                            avg_losses.append(loss)
                            if local_metrics:
                                auv_local_metrics[sid] = local_metrics
                            
                            self.auvs[sid].deduct_battery(e_tx_cost + e_comp_cost, min_battery=self.en_cfg.E_MIN)
                            
                        e_a2r_total += e_tx_cost
                        e_comp_total += e_comp_cost

            # TÍNH NĂNG LƯỢNG NHẬN (E_RX) CHO CÁC RELAY KHI NHẬN TỪ MEMBER
            from physics_models.energy import e_rx
            e_a2r_rx_total = 0.0
            for sid, payload in payloads.items():
                m = self.association.get(sid, -1)
                if m != -1 and m in self.relays:
                    link_key = ('auv', sid, 'relay', m)
                    if link_key in self.G:
                        if hasattr(payload, 'payload_bits'):
                            s_bits = self._payload_to_bits(payload)
                        else:
                            s_bits = len(payload) * 8 if self.baseline_cfg.use_int8 else len(payload) * 32
                        link = self.G[link_key]
                        e_recv = e_rx(s_bits, link.R_bps, self.en_cfg.P_C_RX)
                        e_a2r_rx_total += e_recv
                        self.relays[m].deduct_battery(e_recv, min_battery=self.en_cfg.RELAY_E_MIN)
            e_a2r_total += e_a2r_rx_total

            # --- Phase 2: Relay Tier ---
            e_r2r_total = 0.0
            e_r2g_total = 0.0
            e_r2r_rx_total = 0.0
            e_r2g_rx_total = 0.0
            cooperation_partners = {}

            cluster_sizes = {
                m: sum(1 for sid in relay.cluster_members if sid in payloads)
                for m, relay in self.relays.items()
            }
            mean_c = compute_mean_cluster_size(cluster_sizes)
            q1_dist = compute_q1_relay_distance(self.G)

            # Nội cụm (Intra-cluster) + deduct SVD energy from Relay battery
            # [OPTION A] Song song hóa relay SVD aggregation bằng ThreadPoolExecutor
            # Mỗi relay hoàn toàn độc lập → thread-safe.
            from physics_models.energy import e_tx, e_svd, e_rx
            dead_relays = []
            e_svd_total = 0.0
            svd_calls_by_relay = {}

            def _intra_relay_job(args):
                m, relay = args
                if not relay.alive:
                    return m, 'dead', 0.0, 0.0
                r2r_cost = self._aggregate_intra_relay(m, relay, payloads, auv_n_samples)
                has_updates = any(sid in payloads for sid in relay.cluster_members)
                uses_relay_svd = not self.is_flat
                if self.baseline in DETECTION_BASELINE_CONFIGS:
                    cfg = parse_2d_baseline_config(self.baseline)
                    uses_relay_svd = cfg.use_lora and cfg.lora_aggregation == "svd"
                svd_cost = 0.0
                if uses_relay_svd and has_updates:
                    svd_cost = e_svd(
                        d_out=256, d_in=128, n_svd_calls=1,
                        epsilon_op=self.en_cfg.EPSILON_OP.get(self.task_key, 2.0e-12)
                    )
                return m, 'ok', r2r_cost, svd_cost

            relay_items = list(self.relays.items())
            # Relay aggregation là CPU-bound (SVD thuần NumPy) → ThreadPoolExecutor phù hợp
            _relay_workers = min(len(relay_items), 8)
            with concurrent.futures.ThreadPoolExecutor(max_workers=_relay_workers) as _relay_ex:
                _relay_futures = [_relay_ex.submit(_intra_relay_job, item) for item in relay_items]
                for fut in concurrent.futures.as_completed(_relay_futures):
                    m, status, r2r_cost, svd_cost = fut.result()
                    if status == 'dead':
                        dead_relays.append(m)
                        relay = self.relays[m]
                        print(f"[!] Relay {m} ĐÃ CHẾT (pin = {relay.battery:.1f} J). Cụm {relay.cluster_members} bỏ qua vòng này.")
                    else:
                        e_r2r_total += r2r_cost
                        self.relays[m].deduct_battery(svd_cost, min_battery=self.en_cfg.RELAY_E_MIN)
                        e_svd_total += svd_cost
                        if svd_cost > 0.0:
                            svd_calls_by_relay[m] = 1

            if self.baseline in DETECTION_BASELINE_CONFIGS:
                coop_rule = parse_2d_baseline_config(self.baseline).coop_rule
            elif 'nocoop' in self.baseline:
                coop_rule = 'nocoop'
            elif 'selective' in self.baseline:
                coop_rule = 'selective'
            else:
                coop_rule = 'nearest'
            
            # Chỉ lấy state dict của relay còn sống
            all_intra = {
                m: relay.intra_state_dict
                for m, relay in self.relays.items()
                if relay.alive
                and relay.intra_state_dict is not None
                and cluster_sizes.get(m, 0) > 0
            }
            
            for m, relay in self.relays.items():
                if not relay.alive or relay.intra_state_dict is None:
                    continue
                if not any(sid in payloads for sid in relay.cluster_members):
                    continue
                did_coop, partner_id = relay.cooperate(
                    rule=coop_rule,
                    mean_cluster_size=mean_c,
                    cluster_sizes=cluster_sizes,
                    feasibility_graph=self.G,
                    all_relays_intra_states=all_intra,
                    q1_distance=q1_dist,
                    transport_state=self._transport_relay_state,
                )
                if did_coop and partner_id is not None:
                    cooperation_partners[m] = partner_id
                    uses_relay_svd = False
                    if self.baseline in DETECTION_BASELINE_CONFIGS:
                        cfg = parse_2d_baseline_config(self.baseline)
                        uses_relay_svd = cfg.use_lora and cfg.lora_aggregation == "svd"
                    if uses_relay_svd:
                        final_svd_cost = e_svd(
                            d_out=256,
                            d_in=128,
                            n_svd_calls=1,
                            epsilon_op=self.en_cfg.EPSILON_OP.get(self.task_key, 2.0e-12),
                        )
                        e_svd_total += final_svd_cost
                        svd_calls_by_relay[m] = svd_calls_by_relay.get(m, 0) + 1
                        relay.deduct_battery(
                            final_svd_cost,
                            min_battery=self.en_cfg.RELAY_E_MIN,
                        )
                    link_key_fwd = ('relay', partner_id, 'relay', m)
                    link_key_bwd = ('relay', m, 'relay', partner_id)
                    key = link_key_fwd if link_key_fwd in self.G else (link_key_bwd if link_key_bwd in self.G else None)
                    if key:
                        link = self.G[key]
                        s_bits = self._compute_relay_model_bits()
                        e_coop_tx = e_tx(
                            s_bits, link.R_bps, link.SL_min,
                            self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                        )
                        e_r2r_total += e_coop_tx
                        self.relays[partner_id].deduct_battery(
                            e_coop_tx,
                            min_battery=self.en_cfg.RELAY_E_MIN,
                        )
                        
                        # Trừ pin nhận (e_rx) cho Relay m (người yêu cầu hợp tác)
                        e_coop_rx = e_rx(s_bits, link.R_bps, self.en_cfg.P_C_RX)
                        e_r2r_rx_total += e_coop_rx
                        e_r2r_total += e_coop_rx
                        self.relays[m].deduct_battery(e_coop_rx, min_battery=self.en_cfg.RELAY_E_MIN)
            
            # Tính năng lượng gửi Relay -> Gateway và trừ vào pin Relay
            if not self.is_flat:
                for m, relay in self.relays.items():
                    if not relay.alive or relay.final_state_dict is None:
                        continue
                    if not any(sid in payloads for sid in relay.cluster_members):
                        continue
                    link_key = ('relay', m, 'gateway', 0)
                    if link_key in self.G:
                        link = self.G[link_key]
                        e_r2g_cost = e_tx(
                            self._compute_relay_model_bits(), link.R_bps, link.SL_min,
                            self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                        )
                        e_r2g_total += e_r2g_cost
                        e_gateway_rx = e_rx(
                            self._compute_relay_model_bits(),
                            link.R_bps,
                            self.en_cfg.P_C_RX,
                        )
                        e_r2g_rx_total += e_gateway_rx
                        e_r2g_total += e_gateway_rx
                        relay.deduct_battery(e_r2g_cost, min_battery=self.en_cfg.RELAY_E_MIN)
                    else:
                        print(
                            f"\n{'='*60}\n"
                            f"[! WARNING !] Relay {m} has no feasible gateway uplink.\n"
                            f"-> Skipping R2G transmission for this round!\n"
                            f"{'='*60}\n"
                        )

            # --- Phase 3: Global Aggregation ---
            relay_final = {
                m: self._transport_relay_state(relay.final_state_dict, relay_id=m)
                for m, relay in self.relays.items()
                if relay.final_state_dict is not None
                and any(sid in payloads for sid in relay.cluster_members)
            }
            cluster_samples = {m: sum(auv_n_samples.get(s_id, 0) for s_id in relay.cluster_members) for m, relay in self.relays.items()}
            lora_aggregation = "svd"
            server_mix_beta = 1.0
            if self.baseline in DETECTION_BASELINE_CONFIGS:
                detection_cfg = parse_2d_baseline_config(self.baseline)
                lora_aggregation = detection_cfg.lora_aggregation
                if detection_cfg.server_mix:
                    server_mix_beta = getattr(self.fed_cfg, 'SERVER_MIX_BETA', 0.90)
            self.gateway.aggregate_global(
                relay_final,
                cluster_samples,
                lora_aggregation=lora_aggregation,
                server_mix_beta=server_mix_beta,
            )

            # --- [SCAFFOLD] Cập nhật Global Control Variates ---
            if hasattr(self, 'global_c') and '__scaffold_delta_c__' in self.gateway.global_state_dict:
                delta_c_agg = self.gateway.global_state_dict.pop('__scaffold_delta_c__')
                ratio = len(payloads) / self.net_cfg.N_AUVS if self.net_cfg.N_AUVS > 0 else 1.0
                for k in self.global_c:
                    if k in delta_c_agg:
                        if not torch.is_floating_point(self.global_c[k]):
                            continue
                        update = delta_c_agg[k].to(
                            device=self.global_c[k].device,
                            dtype=self.global_c[k].dtype,
                        )
                        self.global_c[k] += update * ratio

            # KD sẽ được gọi SAU evaluate() để Adaptive Dropout Gate có metrics của vòng này

            # --- Phase 4: Logging ---
            self.energy_tracker.add_round(
                t,
                e_a2r_total,
                e_r2r_total,
                e_r2g_total,
                e_comp_total,
                e_svd_total,
                e_move=pending_mobility_metrics.get('e_move', 0.0),
                e_a2r_rx=e_a2r_rx_total,
                e_r2r_rx=e_r2r_rx_total,
                e_r2g_rx=e_r2g_rx_total,
            )
            
            avg_payload_bits = self._compute_payload_bits(payloads)
            relay_model_bits = self._compute_relay_model_bits()
            
            payload_kb = avg_payload_bits / 8.0 / 1024.0
            cumulative_payload += payload_kb
            
            # Đồng bộ FL phải chờ AUV tham gia có khối lượng tính toán lớn nhất.
            from physics_models.latency import comp_delay_dynamic, max_participant_samples
            max_n_samples = max_participant_samples(auv_n_samples.values())
            tau_comp = comp_delay_dynamic(
                n_samples=int(max_n_samples),
                n_local_epochs=self.fed_cfg.LOCAL_EPOCHS,
                flops_per_sample=self.fed_cfg.MODEL_FLOPS_PER_SAMPLE[self.task_key],
                flop_multiplier=self.get_flop_multiplier(),
                f_cpu=self.en_cfg.F_CPU,
                n_cores=getattr(self.en_cfg, 'N_CORES', 6),
                flops_per_cycle=getattr(self.en_cfg, 'FLOPS_PER_CYCLE', 4.0)
            )
            # Độ trễ điện toán tại Relay — τ_comp,m (physics_models/latency.py)
            from physics_models.latency import relay_comp_delay
            uses_relay_svd = not self.is_flat
            if self.baseline in DETECTION_BASELINE_CONFIGS:
                cfg = parse_2d_baseline_config(self.baseline)
                uses_relay_svd = cfg.use_lora and cfg.lora_aggregation == "svd"
            tau_svd = 0.0
            if uses_relay_svd and payloads:
                max_svd_calls = max(svd_calls_by_relay.values(), default=0)
                tau_svd = relay_comp_delay(
                    n_svd_calls=max_svd_calls,
                    f_cpu=self.en_cfg.F_CPU,
                    n_cores=getattr(self.en_cfg, 'N_CORES', 6),
                    flops_per_cycle=getattr(self.en_cfg, 'FLOPS_PER_CYCLE', 4.0)
                )

            latency_info = self.latency_tracker.compute_round_latency(
                G=self.G,
                association={s: self.association[s] for s in payloads if s in self.association},
                cooperation_partners=cooperation_partners,
                tau_comp=tau_comp,
                tau_svd=tau_svd,
                auv_payload_bits=avg_payload_bits,
                relay_model_bits=relay_model_bits,
            )
            tau_round = latency_info['tau_round']
            self.latency_tracker.add_round(t, latency_info)

            kd_active = getattr(self.fed_cfg, 'KD_ACTIVE', False)
            global_ft_active = getattr(self.fed_cfg, 'GLOBAL_FT', False)
            gateway_will_train = kd_active or global_ft_active
            gateway_modifies_model = gateway_will_train

            # [WiSE-FT] Nếu Gateway sẽ train vòng này, WiSE-FT sẽ dung
            # gateway.pure_aggregated_state (FedAvg thuần, chưa bị server_mix)
            # để blend thay vì dùng global_state_dict (đã bị server_mix trung gian).
            # Việc này tránh double-momentum mà không cần hoàn tác mix.
            if gateway_will_train:
                print(f"[server_mix] Gateway will train → WiSE-FT will use pure FedAvg state for blending.")

            pre_kd_metrics = {}
            if gateway_modifies_model:
                # --- Đánh giá Global Model TRƯỚC khi KD để so sánh ---
                print(f"\n[Simulator] Evaluating Global Model PRE-KD (Round {t})...")
                pre_eval = self.evaluate()
                pre_kd_metrics = {f"pre_kd_{k}": v for k, v in pre_eval.items()}
                self._last_pre_gateway_metrics = pre_eval

            # --- Phase 3b: Gateway-side Knowledge Distillation (Tier 3) ---    
            # Chạy KD TRƯỚC evaluate để lưu được post-KD metrics vào history.
            # Adaptive Dropout Gate sẽ đọc history vòng trước để quyết định có chạy KD không.
            # Simulator2D (fedkdl) override → Teacher KD. Simulator1D → no-op.
            kd_metrics = {}
            if kd_active:
                kd_metrics = self._gateway_knowledge_distillation() or {}
            elif global_ft_active:
                kd_metrics = self._gateway_supervised_finetune(current_round=t, total_rounds=T_rounds) or {}

            msg = "POST-KD" if gateway_modifies_model else "(No KD)"
            cached_gateway_eval = getattr(self, '_last_gateway_eval_metrics', None)
            if cached_gateway_eval:
                print(f"\n[Simulator] Reusing Global Model {msg} metrics from gateway gate (Round {t})...")
                eval_metrics = cached_gateway_eval
                self._last_gateway_eval_metrics = None
            else:
                print(f"\n[Simulator] Evaluating Global Model {msg} (Round {t})...")
                eval_metrics = self.evaluate()

            if not gateway_modifies_model:
                # Nếu không có KD/FT, PRE_KD bằng POST_KD (tránh eval 2 lần)
                pre_kd_metrics = {f"pre_kd_{k}": v for k, v in eval_metrics.items()}

            eval_metrics.update(pre_kd_metrics)

            # --- Lưu post-KD metrics vào history cho Adaptive Dropout vòng sau ---
            if not hasattr(self, '_round_metrics_history'):
                self._round_metrics_history = []
            self._round_metrics_history.append(eval_metrics)

            # ── Eq. 22: Joint Optimisation Cost ──────────────────────────────────────
            # min λ_E · Σ E_round^t + λ_τ · Σ τ_round^t
            # joint_cost_round là đóng góp vật lý của round t vào tổng mục tiêu.
            # ─────────────────────────────────────────────────────────────────────────
            # Năng lượng vòng chính = Truyền thông + Tính toán AUV + Tính toán Relay.
            # Mobility energy is logged separately for velocity/re-clustering analysis,
            # but is intentionally excluded from e_total and joint cost for now.
            # Current objective: task loss + lambda_e * energy + lambda_tau * latency.
            # Physical-only cost is logged separately for operating-cost analysis.
            e_move_total = pending_mobility_metrics.get('e_move', 0.0)
            e_round_total = e_a2r_total + e_r2r_total + e_r2g_total + e_comp_total + e_svd_total
            round_loss    = float(np.mean(avg_losses)) if avg_losses else 0.0
            eval_loss_raw = eval_metrics.get('val_loss', None)
            try:
                eval_loss = float(eval_loss_raw) if eval_loss_raw is not None else round_loss
            except (TypeError, ValueError):
                eval_loss = round_loss
            task_objective_loss = eval_loss if np.isfinite(eval_loss) else round_loss
            lambda_e      = self.fed_cfg.LAMBDA_E
            lambda_tau    = self.fed_cfg.LAMBDA_TAU

            physical_joint_cost_round = physical_joint_cost(
                e_round_total,
                tau_round,
                lambda_e,
                lambda_tau,
            )
            joint_cost_round = task_objective_loss + physical_joint_cost_round
            cumulative_joint_cost += joint_cost_round
            cumulative_physical_joint_cost += physical_joint_cost_round

            metrics = {
                # ── Task Loss ─────────────────────────────────────────────────────
                'loss': round_loss,
                'alive': len(payloads),
                'min_battery': min([s.battery for s in self.auvs.values() if s.alive]) if any(s.alive for s in self.auvs.values()) else 0.0,

                # ── Latency (raw, seconds) — bóc tách từng chặng ─────────────────
                'tau_round_s': tau_round,
                'tau_status':  'OK' if tau_round <= self.fed_cfg.TAU_MAX else 'VIOLATED',
                'tau_a2r':     latency_info['tau_a2r'],   # max AUV→Relay bottleneck
                'tau_r2r':     latency_info['tau_r2r'],   # max Relay↔Relay cooperation
                'tau_r2g':     latency_info['tau_r2g'],   # max Relay→Gateway bottleneck
                'tau_comp':    latency_info['tau_comp'],  # max local computation
                'tau_svd':     latency_info['tau_svd'],
                'tau_cumul_s': self.latency_tracker.cumulative_latency,

                # ── Payload ───────────────────────────────────────────────────────
                'avg_payload_kb':    payload_kb,
                'payload_cumul_kb':  cumulative_payload,

                # ── Energy (raw, Joules) — bóc tách từng chặng ───────────────────
                'e_total': e_round_total,
                'e_a2r':   e_a2r_total,   # AUV → Relay TX + RX energy
                'e_r2r':   e_r2r_total,   # Relay ↔ Relay TX + RX energy
                'e_r2g':   e_r2g_total,   # Relay → Gateway TX + RX energy
                'e_a2r_rx': e_a2r_rx_total,
                'e_r2r_rx': e_r2r_rx_total,
                'e_r2g_rx': e_r2g_rx_total,
                'e_rx': e_a2r_rx_total + e_r2r_rx_total + e_r2g_rx_total,
                'e_comp':  e_comp_total,  # Local computation energy (all active auvs)
                'e_svd':   e_svd_total,   # Relay SVD computation energy
                'e_move':  e_move_total,  # Logged only; excluded from e_total/joint cost for now
                'e_move_cumul': pending_mobility_metrics.get('e_move_cumul', 0.0),
                'e_cumul': self.energy_tracker.cumulative_energy,

                # ── Mobility/re-clustering diagnostics ───────────────────────
                'avg_move_m': pending_mobility_metrics.get('avg_move_m', 0.0),
                'max_move_m': pending_mobility_metrics.get('max_move_m', 0.0),
                'avg_speed_mps': pending_mobility_metrics.get('avg_speed_mps', 0.0),
                'mobility_recluster_changes': pending_mobility_metrics.get('mobility_recluster_changes', 0),
                'mobility_recluster_rate': pending_mobility_metrics.get('mobility_recluster_rate', 0.0),
                'mobility_connected_after': pending_mobility_metrics.get('mobility_connected_after', len(self.association)),

                # ── Joint Cost — Eq. 22 (λ_E, λ_τ weighted) ─────────────────────
                # joint_cost_round  = λ_E·E_round^t + λ_τ·τ_round^t
                # joint_cost_cumul  = Σ_{s=1}^{t} joint_cost_round_s  (running sum)
                # joint_cost_round includes objective_loss plus the physical component.
                'lambda_e':           lambda_e,
                'lambda_tau':         lambda_tau,
                'objective_loss':      task_objective_loss,
                'physical_joint_cost_round': physical_joint_cost_round,
                'physical_joint_cost_cumul': cumulative_physical_joint_cost,
                'joint_cost_round':   joint_cost_round,
                'joint_cost_cumul':   cumulative_joint_cost,
                'server_mix_beta':    server_mix_beta,
                
                # ── Per-AUV Local Evaluation ──────────────────────────────────
                'auv_train_metrics': auv_local_metrics,
            }
            metrics.update(kd_metrics)
            metrics.update(eval_metrics)

            self.metrics_logger.log(t, metrics)
            try:
                self.metrics_logger.print_latest()
            except Exception as exc:
                print(f"[Warning] Failed to print round {t} metrics: {exc}")

            # =========================================================================
            # [NEW] Di chuyển AUV đáy biển và Tái phân cụm (Gauss-Markov + EMD Joint)
            # =========================================================================
            pending_mobility_metrics = {
                'e_move': 0.0,
                'e_move_cumul': movement_energy_cumul,
                'avg_move_m': 0.0,
                'max_move_m': 0.0,
                'avg_speed_mps': 0.0,
                'mobility_recluster_changes': 0,
                'mobility_recluster_rate': 0.0,
                'mobility_connected_after': len(self.association),
            }
            if getattr(self.net_cfg, 'MOBILITY_ENABLED', True) and hasattr(self.topology, 'step_mobile_auvs'):
                mobility_dt = getattr(self.net_cfg, 'MOBILITY_DT_PER_ROUND', 1.0)
                mobility_stats = self.topology.step_mobile_auvs(
                    mu=getattr(self.net_cfg, 'GM_ALPHA', 0.7),
                    mean_speed=getattr(self.net_cfg, 'GM_MEAN_SPEED', 1.5),
                    max_speed=getattr(self.net_cfg, 'GM_MAX_SPEED', 5.0),
                    mean_heading=getattr(self.net_cfg, 'GM_MEAN_HEADING', 0.0),
                    mean_pitch=getattr(self.net_cfg, 'GM_MEAN_PITCH', 0.0),
                    sigma_speed=getattr(self.net_cfg, 'GM_SIGMA_SPEED', 0.5),
                    sigma_heading=getattr(self.net_cfg, 'GM_SIGMA_HEADING', 0.3),
                    sigma_pitch=getattr(self.net_cfg, 'GM_SIGMA_PITCH', 0.1),
                    dt=mobility_dt,
                )
                e_move_total_next = 0.0
                if getattr(self.en_cfg, 'MOVE_ENERGY_ENABLED', False):
                    from physics_models.energy import e_move_yang_surge
                    speeds = mobility_stats.get('speed_mps', np.zeros(self.topology.N))
                    move_energy_by_auv = e_move_yang_surge(
                        speeds,
                        duration_s=mobility_dt,
                        rho_w=getattr(self.en_cfg, 'AUV_WATER_DENSITY', 1025.0),
                        thruster_radius=getattr(self.en_cfg, 'AUV_THRUSTER_RADIUS', 0.025),
                        surge_drag_coeff=getattr(self.en_cfg, 'AUV_SURGE_DRAG_COEFF', 48.17),
                        n_horizontal_thrusters=getattr(self.en_cfg, 'AUV_HORIZONTAL_THRUSTERS', 2),
                        hotel_power=getattr(self.en_cfg, 'AUV_HOTEL_POWER', 0.0),
                    )
                    for sid, auv in self.auvs.items():
                        if sid < len(move_energy_by_auv) and auv.alive:
                            e_i = float(move_energy_by_auv[sid])
                            e_move_total_next += e_i
                            if not np.isinf(auv.battery):
                                auv.battery = max(0.0, auv.battery - e_i)
                                if auv.battery < self.en_cfg.E_MIN:
                                    auv.alive = False
                movement_energy_cumul += e_move_total_next
                from physics_models.topology import build_feasibility_graph, nearest_feasible_association, build_clusters
                
                # Tính lại đồ thị khoảng cách vật lý
                self.G = build_feasibility_graph(self.topology, self.ac_cfg)
                
                # Tái phân cụm
                flat_baselines = ['fedavg', 'fedprox', 'fedkd', 'centralized']
                if self.baseline in flat_baselines:
                    from physics_models.topology import flat_topology_association
                    new_assoc = flat_topology_association(self.topology, self.G)
                elif hasattr(self, 'auv_label_hists') and hasattr(self, 'relay_label_hists'):
                    # Nếu là Simulator2D (có EMD), kết hợp EMD + Khoảng cách vật lý
                    from tasks.detection_2d.knowledge_compression.knowledge_association import knowledge_aware_association
                    class DummyTopo:
                        def __init__(self, n, m): self.N = n; self.M = m
                    
                    new_assoc = knowledge_aware_association(
                        topology=DummyTopo(self.topology.N, self.topology.M),
                        G=self.G,
                        auv_label_hists=self.auv_label_hists,
                        relay_label_hists=self.relay_label_hists,
                        beta=self.fed_cfg.BETA_EMD,
                    )
                else:
                    # Nếu chạy HFL Classic/1D, phân cụm theo SNR vật lý
                    from physics_models.topology import nearest_feasible_association
                    new_assoc = nearest_feasible_association(self.topology, self.G)

                # In ra chi tiết AUV nào đổi sang cụm nào
                old_assoc = dict(self.association)
                changes_log = []
                for s in sorted(set(old_assoc) | set(new_assoc)):
                    old_f = old_assoc.get(s, -1)
                    new_f = new_assoc.get(s, -1)
                    if old_f != new_f:
                        changes_log.append(f"    - AUV {s}: Relay {old_f} -> Relay {new_f}")
                        
                changed = len(changes_log)
                if changed > 0:
                    print(f"[*] [Round {t}] Re-clustering: {changed}/{self.topology.N} Mobile AUVs changed Relays:")
                    for log_msg in changes_log:
                        print(log_msg)

                pending_mobility_metrics = {
                    'e_move': e_move_total_next,
                    'e_move_cumul': movement_energy_cumul,
                    'avg_move_m': mobility_stats.get('avg_move_m', 0.0),
                    'max_move_m': mobility_stats.get('max_move_m', 0.0),
                    'avg_speed_mps': mobility_stats.get('avg_speed_mps', 0.0),
                    'mobility_recluster_changes': changed,
                    'mobility_recluster_rate': changed / max(1, self.topology.N),
                    'mobility_connected_after': len(new_assoc),
                }
                        
                self.association = new_assoc
                    
                if self.is_flat:
                    self.clusters = {0: sorted(self.association)}
                else:
                    self.clusters = build_clusters(self.association, self.topology.M)
                
                # Cập nhật danh sách quản lý vào đối tượng Relay và AUV
                for relay_id, relay in self.relays.items():
                    relay.cluster_members = self.clusters.get(relay_id, [])
                    relay.cluster_size = len(relay.cluster_members)
                for auv_id, auv in self.auvs.items():
                    if auv_id in self.association:
                        auv.associated_relay = self.association[auv_id]
            # =========================================================================


        return self.metrics_logger.logs
