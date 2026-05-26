import gc
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from federated_core.metrics import EnergyTracker, LatencyTracker, MetricsLogger
from federated_core.hfl_rules import compute_mean_cluster_size, compute_q1_fog_distance

class BaseSimulator(ABC):
    """
    Khung mô phỏng chung cho 1D và 2D.
    Dùng Template Method Pattern: các class kế thừa override _process_sensor và evaluate.
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
        self.sensors = {}
        self.fogs = {}
        self.gateway = None
        self.task_key = "1D" # default, override in subclass

    def _load_environment(self):
        from utils.env_manager import EnvironmentManager
        topo = EnvironmentManager.load_topology(self.topo_path)
        self.N_actual = topo.N            # Số sensor thực tế trong topology (từ file pkl)
        self.sensor_positions = topo.sensor_positions
        self.fog_positions = topo.fog_positions
        self.gateway_position = topo.gateway_position
        self.G = EnvironmentManager.restore_graph(topo)
        flat_baselines = ['fedavg', 'fedprox', 'centralized', 'fedkd']
        self.is_flat = any(b in self.baseline for b in flat_baselines) and 'hfl' not in self.baseline and 'fedkdl' not in self.baseline
        self.association = topo.flat_association if self.is_flat else topo.hfl_association
        self.clusters = topo.clusters

    def get_flop_multiplier(self) -> float:
        """Cho phép các lớp con (ví dụ 2D Simulator) định nghĩa lại Flop Multiplier dựa theo logic của Local KD hay LoRA."""
        return self.fed_cfg.FLOP_MULTIPLIER[self.task_key]

    @abstractmethod
    def _process_sensor(self, s_id: int) -> Tuple[int, Any, float, int, float, float]:
        """
        Huấn luyện và nén tại một sensor.
        Returns:
            s_id, payload (hoặc state_dict), avg_loss, n_samples, e_tx, e_comp
        """
        pass

    @abstractmethod
    def _aggregate_intra_fog(self, m: int, fog, payloads, sensor_n_samples) -> float:
        """
        Nội cụm + tính e_f2f.
        Returns: e_f2f_cost
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
    def _compute_fog_model_bits(self) -> float:
        """Tính kích thước model fog (bits)."""
        pass

    def _gateway_knowledge_distillation(self):
        """
        Hook: Gateway-side Knowledge Distillation sau Global Aggregation.

        Mặc định: no-op (dùng cho Simulator1D hoặc baseline không dùng KD).
        Simulator2D (baseline='fedkdl') sẽ override để chạy Teacher KD tại Tier 3.
        """
        pass

    def run(self, T_rounds: int, baseline: str = None) -> list:
        if baseline is not None:
            self.baseline = baseline
        import concurrent.futures
        
        # In ra thông tin topology để biết sensor nào thuộc cụm nào
        print("\n" + "="*60)
        print("[*] CLUSTER TOPOLOGY INFO:")
        if self.is_flat:
            print("    (Flat Topology: Sensors connect directly or via relays to Gateway)")
        for fog_id, fog in self.fogs.items():
            print(f"    - Fog {fog_id} manages sensors: {fog.cluster_members}")
        print("="*60 + "\n")
        
        cumulative_payload = 0.0
        cumulative_joint_cost = 0.0  # Σ joint_cost^t  — tích lũy Eq.22 qua các round
        
        import math
        initial_lr = self.fed_cfg.LOCAL_LR

        for t in range(1, T_rounds + 1):
            # Tính toán Cosine Annealing Learning Rate cho toàn bộ quá trình FL
            progress = (t - 1) / max(1, T_rounds - 1)
            # LRF (Learning Rate Fraction) = 0.01 như mặc định của YOLO
            self.current_lr = initial_lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.task_key == "2D":
                print(f"\n{'='*60}")
                print(f"|  [Simulator] BẮT ĐẦU VÒNG {t}/{T_rounds}  |")
                print(f"{'='*60}\n")
                print(
                    f"[*] LR Schedule | round={t}/{T_rounds} | "
                    f"base_lr={initial_lr:.6f} -> current_lr={self.current_lr:.8f}"
                )
            else:
                print(
                    f"   -> [FL Simulator 1D] Round {t}/{T_rounds} | "
                    f"current_lr={self.current_lr:.8f}",
                    flush=True,
                )

            # --- Phase 1: Sensor Tier ---
            alive_sensors = [s.sensor_id for s in self.sensors.values() if s.alive]
            dead_sensors = [s.sensor_id for s in self.sensors.values() if not s.alive]
            missing_sensors = [i for i in range(self.N_actual) if i not in self.sensors]

            if self.task_key == "2D":
                if missing_sensors:
                    print(f"[!] BỎ QUA {len(missing_sensors)} SENSORS (Out of Range / Mất kết nối): {missing_sensors}")
                if dead_sensors:
                    print(f"[!] BỎ QUA {len(dead_sensors)} SENSORS (Đã chết / Hết pin): {dead_sensors}")
                print(f"[*] SENSORS ĐANG HOẠT ĐỘNG ({len(alive_sensors)}): {alive_sensors}\n")
            else:
                total = self.N_actual
                connected = len(alive_sensors)
                out_of_range = len(missing_sensors)
                dead = len(dead_sensors)
                participation = connected / total if total > 0 else 0
                print(f"   -> [1D Round {t}/{T_rounds}] Active={connected}/{total} ({participation:.0%}) | OOR={out_of_range} | Dead={dead}", flush=True)

            if not alive_sensors:
                print(f"[Simulator] All sensors depleted at round {t}. Stopping.")
                break

            payloads = {}
            sensor_n_samples = {}
            avg_losses = []
            sensor_local_metrics = {}

            
            e_s2f_total = 0.0
            e_comp_total = 0.0

            if self.task_key == "2D":
                # Chạy tuần tự cho YOLO 2D vì Ultralytics trainer gây deadlock/crash nếu chạy đa luồng CUDA
                for s_id in alive_sensors:
                    sid, payload, loss, n_samp, e_tx_cost, e_comp_cost, local_metrics = self._process_sensor(s_id)
                    if payload is not None:
                        payloads[sid] = payload
                        sensor_n_samples[sid] = n_samp
                        avg_losses.append(loss)
                        if local_metrics:
                            sensor_local_metrics[sid] = local_metrics
                        self.sensors[sid].deduct_battery(e_tx_cost + e_comp_cost, min_battery=self.energy_cfg.E_MIN)
                    e_s2f_total += e_tx_cost
                    e_comp_total += e_comp_cost
            else:
                import os
                torch.set_num_threads(1) # RẤT QUAN TRỌNG: Ngăn chặn PyTorch sinh thêm luồng ngầm gây chết CPU
                max_w = 4 if self.device == 'cuda' else (os.cpu_count() or 8)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as executor:
                    futures = {executor.submit(self._process_sensor, s_id): s_id for s_id in alive_sensors}
                    for future in concurrent.futures.as_completed(futures):
                        sid, payload, loss, n_samp, e_tx_cost, e_comp_cost, local_metrics = future.result()
                        
                        if payload is not None:
                            payloads[sid] = payload
                            sensor_n_samples[sid] = n_samp
                            avg_losses.append(loss)
                            if local_metrics:
                                sensor_local_metrics[sid] = local_metrics
                            
                            self.sensors[sid].deduct_battery(e_tx_cost + e_comp_cost, min_battery=self.energy_cfg.E_MIN)
                            
                        e_s2f_total += e_tx_cost
                        e_comp_total += e_comp_cost

            # --- Phase 2: Fog Tier ---
            e_f2f_total = 0.0
            e_f2g_total = 0.0
            cooperation_partners = {}

            cluster_sizes = {m: fog.cluster_size for m, fog in self.fogs.items()}
            mean_c = compute_mean_cluster_size(cluster_sizes)
            q1_dist = compute_q1_fog_distance(self.G)

            # Nội cụm (Intra-cluster)
            for m, fog in self.fogs.items():
                e_f2f_total += self._aggregate_intra_fog(m, fog, payloads, sensor_n_samples)

            # Liên cụm (Inter-cluster Cooperation)
            if 'selective' in self.baseline or 'fedkdl' in self.baseline:
                coop_rule = 'selective'
            elif 'nearest' in self.baseline:
                coop_rule = 'nearest'
            else:
                coop_rule = 'nocoop'
            
            all_intra = {m: fog.intra_state_dict for m, fog in self.fogs.items() if fog.intra_state_dict is not None}
            from physics_models.energy import e_tx
            
            for m, fog in self.fogs.items():
                if fog.intra_state_dict is None:
                    continue
                did_coop, partner_id = fog.cooperate(
                    rule=coop_rule,
                    mean_cluster_size=mean_c,
                    cluster_sizes=cluster_sizes,
                    feasibility_graph=self.G,
                    all_fogs_intra_states=all_intra,
                    q1_distance=q1_dist,
                )
                if did_coop and partner_id is not None:
                    cooperation_partners[m] = partner_id
                    link_key_fwd = ('fog', partner_id, 'fog', m)
                    link_key_bwd = ('fog', m, 'fog', partner_id)
                    key = link_key_fwd if link_key_fwd in self.G else (link_key_bwd if link_key_bwd in self.G else None)
                    if key:
                        link = self.G[key]
                        e_f2f_total += e_tx(
                            self._compute_fog_model_bits(), link.R_bps, link.SL_min,
                            self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                        )
            
            # Tính năng lượng gửi Fog -> Gateway
            from physics_models.energy import e_tx
            if not self.is_flat:
                for m, fog in self.fogs.items():
                    if fog.final_state_dict is not None:
                        link_key = ('fog', m, 'gateway', 0)
                        if link_key in self.G:
                            link = self.G[link_key]
                            e_f2g_total += e_tx(
                                self._compute_fog_model_bits(), link.R_bps, link.SL_min,
                                self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                            )

            # --- Phase 3: Global Aggregation ---
            fog_final = {m: fog.final_state_dict for m, fog in self.fogs.items() if fog.final_state_dict is not None}
            cluster_samples = {m: sum(sensor_n_samples.get(s_id, 0) for s_id in fog.cluster_members) for m, fog in self.fogs.items()}
            self.gateway.aggregate_global(fog_final, cluster_samples)

            # KD sẽ được gọi SAU evaluate() để Adaptive Dropout Gate có metrics của vòng này

            # --- Phase 4: Logging ---
            self.energy_tracker.add_round(t, e_s2f_total, e_f2f_total, e_f2g_total, e_comp_total)
            
            avg_payload_bits = self._compute_payload_bits(payloads)
            fog_model_bits = self._compute_fog_model_bits()
            
            payload_kb = avg_payload_bits / 8.0 / 1024.0
            cumulative_payload += payload_kb
            
            # Tính tau_comp trung bình của round
            from physics_models.latency import comp_delay_dynamic
            avg_n_samples = np.mean(list(sensor_n_samples.values())) if sensor_n_samples else 100
            tau_comp = comp_delay_dynamic(
                n_samples=int(avg_n_samples),
                n_local_epochs=self.fed_cfg.LOCAL_EPOCHS,
                flops_per_sample=self.fed_cfg.MODEL_FLOPS_PER_SAMPLE[self.task_key],
                flop_multiplier=self.get_flop_multiplier(),
                f_cpu=self.en_cfg.F_CPU
            )
            
            latency_info = self.latency_tracker.compute_round_latency(
                G=self.G,
                association={s: self.association[s] for s in alive_sensors if s in self.association},
                cooperation_partners=cooperation_partners,
                tau_comp=tau_comp,
                sensor_payload_bits=avg_payload_bits,
                fog_model_bits=fog_model_bits,
            )
            tau_round = latency_info['tau_round']
            self.latency_tracker.add_round(t, latency_info)

            # --- Phase 3b: Gateway-side Knowledge Distillation (Tier 3) ---
            # Chạy KD TRƯỚC evaluate để lưu được post-KD metrics vào history.
            # Adaptive Dropout Gate sẽ đọc history vòng trước để quyết định có chạy KD không.
            # Simulator2D (fedkdl) override → Teacher KD. Simulator1D → no-op.
            kd_metrics = self._gateway_knowledge_distillation() or {}

            eval_metrics = self.evaluate()

            # --- Lưu post-KD metrics vào history cho Adaptive Dropout vòng sau ---
            if not hasattr(self, '_round_metrics_history'):
                self._round_metrics_history = []
            self._round_metrics_history.append(eval_metrics)

            # ── Eq. 22: Joint Optimisation Cost ──────────────────────────────────────
            # min  F(θ^T) + λ_E · Σ E_round^t  +  λ_τ · Σ τ_round^t
            # joint_cost_round là đóng góp của round t vào tổng trên (chưa gộp F(θ^T)).
            # F(θ^T) = avg validation/training loss tại round T  —  đại diện bằng 'loss'.
            # ─────────────────────────────────────────────────────────────────────────
            e_round_total = e_s2f_total + e_f2f_total + e_f2g_total + e_comp_total
            round_loss    = float(np.mean(avg_losses)) if avg_losses else 0.0
            lambda_e      = self.fed_cfg.LAMBDA_E
            lambda_tau    = self.fed_cfg.LAMBDA_TAU

            joint_cost_round  = round_loss + lambda_e * e_round_total + lambda_tau * tau_round
            cumulative_joint_cost += joint_cost_round

            metrics = {
                # ── Task Loss ─────────────────────────────────────────────────────
                'loss': round_loss,
                'alive': len(alive_sensors),
                'min_battery': min([s.battery for s in self.sensors.values() if s.alive]) if any(s.alive for s in self.sensors.values()) else 0.0,

                # ── Latency (raw, seconds) — bóc tách từng chặng ─────────────────
                'tau_round_s': tau_round,
                'tau_status':  'OK' if tau_round <= self.fed_cfg.TAU_MAX else 'VIOLATED',
                'tau_s2f':     latency_info['tau_s2f'],   # max Sensor→Fog bottleneck
                'tau_f2f':     latency_info['tau_f2f'],   # max Fog↔Fog cooperation
                'tau_f2g':     latency_info['tau_f2g'],   # max Fog→Gateway bottleneck
                'tau_comp':    latency_info['tau_comp'],  # max local computation
                'tau_cumul_s': self.latency_tracker.cumulative_latency,

                # ── Payload ───────────────────────────────────────────────────────
                'avg_payload_kb':    payload_kb,
                'payload_cumul_kb':  cumulative_payload,

                # ── Energy (raw, Joules) — bóc tách từng chặng ───────────────────
                'e_total': e_round_total,
                'e_s2f':   e_s2f_total,   # Sensor → Fog TX energy
                'e_f2f':   e_f2f_total,   # Fog ↔ Fog cooperation TX energy
                'e_f2g':   e_f2g_total,   # Fog → Gateway TX energy
                'e_comp':  e_comp_total,  # Local computation energy (all active sensors)
                'e_cumul': self.energy_tracker.cumulative_energy,

                # ── Joint Cost — Eq. 22 (λ_E, λ_τ weighted) ─────────────────────
                # joint_cost_round  = loss_t + λ_E·E_round^t + λ_τ·τ_round^t
                # joint_cost_cumul  = Σ_{s=1}^{t} joint_cost_round_s  (running sum)
                'lambda_e':           lambda_e,
                'lambda_tau':         lambda_tau,
                'joint_cost_round':   joint_cost_round,
                'joint_cost_cumul':   cumulative_joint_cost,
                
                # ── Per-Sensor Local Evaluation ──────────────────────────────────
                'sensor_train_metrics': sensor_local_metrics,
            }
            metrics.update(kd_metrics)
            metrics.update(eval_metrics)

            self.metrics_logger.log(t, metrics)
            try:
                self.metrics_logger.print_latest()
            except Exception as exc:
                print(f"[Warning] Failed to print round {t} metrics: {exc}")

        return self.metrics_logger.logs
