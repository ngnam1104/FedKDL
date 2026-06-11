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

    def _gateway_knowledge_distillation(self):
        """
        Hook: Gateway-side Knowledge Distillation sau Global Aggregation.

        Mặc định: no-op (dùng cho Simulator1D hoặc baseline không dùng KD).
        Simulator2D (baseline='fedkdl') sẽ override để chạy Teacher KD tại Tier 3.
        """
        pass

    def _transport_relay_state(self, state):
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
        cumulative_joint_cost = 0.0  # Σ joint_cost^t  — tích lũy Eq.22 qua các round
        
        import math
        initial_lr = self.fed_cfg.LOCAL_LR

        if (
            self.task_key == "2D"
            and getattr(self.fed_cfg, 'PREWARM_YOLO_LABEL_CACHE', False)
            and hasattr(self, '_pre_warm_dataset_cache')
        ):
            self._pre_warm_dataset_cache()

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
                        print(f"    - Relay {relay_id} manages auvs: {relay.cluster_members}")
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
                    print(f"[!] BỎ QUA {len(missing_auvs)} AUVS (Out of Range / Mất kết nối): {missing_auvs}")
                if dead_auvs:
                    print(f"[!] BỎ QUA {len(dead_auvs)} AUVS (Đã chết / Hết pin): {dead_auvs}")
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

            # --- Phase 3: Global Aggregation ---
            relay_final = {
                m: self._transport_relay_state(relay.final_state_dict)
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
                ratio = len(payloads) / self.net_cfg.N_AUVS
                for k in self.global_c:
                    if k in delta_c_agg:
                        self.global_c[k] += delta_c_agg[k].to(self.global_c[k].device) * ratio

            # KD sẽ được gọi SAU evaluate() để Adaptive Dropout Gate có metrics của vòng này

            # --- Phase 4: Logging ---
            self.energy_tracker.add_round(
                t,
                e_a2r_total,
                e_r2r_total,
                e_r2g_total,
                e_comp_total,
                e_svd_total,
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

            # --- Đánh giá Global Model TRƯỚC khi KD để debug mAP/Prec/Rec ---
            pre_kd_metrics = {}
            # (Bỏ đánh giá pre-KD theo yêu cầu để giảm thiểu 40s mỗi vòng)

            # --- Phase 3b: Gateway-side Knowledge Distillation (Tier 3) ---
            # Chạy KD TRƯỚC evaluate để lưu được post-KD metrics vào history.
            # Adaptive Dropout Gate sẽ đọc history vòng trước để quyết định có chạy KD không.
            # Simulator2D (fedkdl) override → Teacher KD. Simulator1D → no-op.
            kd_metrics = {}
            if getattr(self.fed_cfg, 'KD_ACTIVE', False):
                kd_metrics = self._gateway_knowledge_distillation() or {}
            elif getattr(self.fed_cfg, 'GLOBAL_FT', False):
                self._gateway_supervised_finetune()

            print(f"\n[Simulator] Evaluating Global Model POST-KD (Round {t})...")
            eval_metrics = self.evaluate()
            eval_metrics.update(pre_kd_metrics)

            # --- Lưu post-KD metrics vào history cho Adaptive Dropout vòng sau ---
            if not hasattr(self, '_round_metrics_history'):
                self._round_metrics_history = []
            self._round_metrics_history.append(eval_metrics)

            # ── Eq. 22: Joint Optimisation Cost ──────────────────────────────────────
            # min λ_E · Σ E_round^t + λ_τ · Σ τ_round^t
            # joint_cost_round là đóng góp vật lý của round t vào tổng mục tiêu.
            # ─────────────────────────────────────────────────────────────────────────
            # Năng lượng vòng = Truyền thông + Tính toán AUV + Tính toán Relay
            e_round_total = e_a2r_total + e_r2r_total + e_r2g_total + e_comp_total + e_svd_total
            round_loss    = float(np.mean(avg_losses)) if avg_losses else 0.0
            lambda_e      = self.fed_cfg.LAMBDA_E
            lambda_tau    = self.fed_cfg.LAMBDA_TAU

            joint_cost_round = physical_joint_cost(
                e_round_total,
                tau_round,
                lambda_e,
                lambda_tau,
            )
            cumulative_joint_cost += joint_cost_round

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
                'e_cumul': self.energy_tracker.cumulative_energy,

                # ── Joint Cost — Eq. 22 (λ_E, λ_τ weighted) ─────────────────────
                # joint_cost_round  = λ_E·E_round^t + λ_τ·τ_round^t
                # joint_cost_cumul  = Σ_{s=1}^{t} joint_cost_round_s  (running sum)
                'lambda_e':           lambda_e,
                'lambda_tau':         lambda_tau,
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
            if hasattr(self.topology, 'step_mobile_auvs'):
                self.topology.step_mobile_auvs(max_speed=5.0)
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
                changes_log = []
                for s, new_f in new_assoc.items():
                    old_f = self.association.get(s, -1)
                    if old_f != new_f:
                        changes_log.append(f"    - AUV {s}: Relay {old_f} -> Relay {new_f}")
                        
                changed = len(changes_log)
                if changed > 0:
                    print(f"[*] [Round {t}] Re-clustering: {changed}/{self.topology.N} Mobile AUVs changed Relays:")
                    for log_msg in changes_log:
                        print(log_msg)
                        
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
