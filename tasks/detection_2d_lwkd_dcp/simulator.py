"""
simulator.py — SOTA Baseline Simulator (Jiang et al., 2025)
Kế thừa BaseSimulator từ FedKDL core (KHÔNG sửa code gốc).

Điểm khác biệt so với Simulator2D (FedKDL):
  - Sensor chạy Local KD (Teacher YOLO12l tại chỗ) + DCP.
  - Sensor gửi TOÀN BỘ model Float32 (~5.4 MB) — không LoRA, không INT8.
  - Gateway KHÔNG chạy KD: chỉ làm FedAvg thuần túy.
  - Dùng StudentModel(full_param=True, use_lora=False).
"""

import os
import gc
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from federated_core.base_simulator import BaseSimulator
from federated_core.workers import BaseWorker, BaseFogNode, BaseGateway
from federated_core.aggregator import fedavg_intra_cluster, fedavg_global

from tasks.detection_2d.models.yolo_wrapper import StudentModel, TeacherModel
from tasks.detection_2d_sota.trainer import local_sgd_od_sota, evaluate_od_sota

from physics_models.energy import e_tx, e_comp_dynamic
from physics_models.latency import comm_delay, comp_delay_dynamic
from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg


# ─────────────────────────────────────────────────────────────────────────────
#  Payload size cho SOTA: toàn bộ YOLO11n Float32 (~5.4 MB = ~43.5 Mbits)
# ─────────────────────────────────────────────────────────────────────────────
SOTA_PAYLOAD_BITS = int(5.4 * 1024 * 1024 * 8)   # 5.4 MB × 8 bits
SOTA_PAYLOAD_KB   = 5.4 * 1024                     # 5529.6 KB


class SensorWorkerSOTA(BaseWorker):
    """
    Sensor (AUV) trong baseline SOTA:
      - Load Teacher YOLO12l cục bộ.
      - Train với DCP + Local KD.
      - Gửi TOÀN BỘ model Float32 lên Fog (rất nặng).
    """

    def __init__(self, sensor_id, client_yaml, battery_init, teacher_model):
        super().__init__(sensor_id, battery_init)
        self.client_yaml = client_yaml
        self.teacher_model = teacher_model   # Dùng chung (frozen, read-only)

        with open(self.client_yaml, 'r') as f:
            c_cfg = yaml.safe_load(f)
        with open(c_cfg['train'], 'r') as f:
            self.n_samples = sum(1 for _ in f)

    def train_and_get_payload(self, global_state: dict, epochs: int, lr: float,
                               device: str) -> tuple:
        """
        Returns:
            (full_state_dict, payload_kb, train_loss) hoặc (None, 0, 0) nếu dead.
        """
        if not self.alive or self.n_samples == 0:
            return None, 0.0, 0.0

        with open(self.client_yaml, 'r') as f:
            nc = yaml.safe_load(f).get('nc', 80)

        # Student: TOÀN BỘ tham số, KHÔNG LoRA, KHÔNG INT8
        local_student = StudentModel(
            "yolo11n.pt", rank=4, nc=nc,
            full_param=True, use_lora=False
        )
        local_student.load_trainable_state_dict(global_state)

        full_state, train_loss = local_sgd_od_sota(
            student_model=local_student,
            teacher_model=self.teacher_model,
            client_yaml=self.client_yaml,
            client_id=self.sensor_id,
            epochs=epochs,
            lr=lr,
            device=device,
            use_dcp=True,
        )
        return full_state, SOTA_PAYLOAD_KB, train_loss


class FogNodeSOTA(BaseFogNode):
    """Fog node nhận full model Float32, FedAvg, gửi lên Gateway."""

    def __init__(self, fog_id):
        super().__init__(fog_id)
        self.received: Dict[int, dict] = {}    # sensor_id → full_state_dict

    def receive(self, sensor_id: int, state_dict: dict):
        self.received[sensor_id] = state_dict

    def aggregate_intra_cluster(self) -> dict:
        """FedAvg thông thường trên toàn bộ key."""
        if not self.received:
            return {}
        keys = list(next(iter(self.received.values())).keys())
        avg = {}
        n = len(self.received)
        for k in keys:
            avg[k] = sum(sd[k].float() for sd in self.received.values()) / n
        return avg

    def reset(self):
        self.received.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  SimulatorSOTA
# ─────────────────────────────────────────────────────────────────────────────

class SimulatorSOTA(BaseSimulator):
    """
    Mô phỏng baseline Jiang et al. (2025) cho bài toán Object Detection IoUT.
    """

    def __init__(self, topo_path: str, data_path: str,
                 test_yaml: str = "datasets/URPC2020.yaml",
                 student_ckpt: str = "yolo11n.pt",
                 teacher_ckpt: str = "yolo12l.pt",
                 device: str = "cpu"):
        super().__init__(topo_path, data_path)
        self.test_yaml  = test_yaml
        self.device     = device

        # Tải Teacher MỘT LẦN — dùng chung cho tất cả Sensor (frozen)
        print("[SimulatorSOTA] Tải Teacher YOLO12l (frozen)...")
        self.teacher_model = TeacherModel(teacher_ckpt)
        self.teacher_model.yolo.model.to(device)
        print("[SimulatorSOTA] Teacher sẵn sàng.")

        # Khởi tạo Global Student (full_param)
        nc = self._get_nc()
        self.global_student = StudentModel(
            student_ckpt, rank=4, nc=nc,
            full_param=True, use_lora=False
        )
        self.global_state = self.global_student.trainable_state_dict()

        # Khởi tạo Sensors & Fogs
        self._init_workers(nc)

        # Cosine LR schedule
        self._lr_schedule = self._build_cosine_lr()

    def _get_nc(self) -> int:
        try:
            with open(self.test_yaml, 'r') as f:
                return yaml.safe_load(f).get('nc', 4)
        except Exception:
            return 4

    def _build_cosine_lr(self):
        """Giống với Simulator2D gốc."""
        T = fed_cfg.GLOBAL_ROUNDS.get("2D", 50)
        lr0 = fed_cfg.LOCAL_LR
        lrs = [lr0 * (1 + np.cos(np.pi * t / T)) / 2 for t in range(T + 1)]
        return lrs

    def _init_workers(self, nc: int):
        """Tạo SensorWorkerSOTA và FogNodeSOTA từ topology."""
        import pickle
        with open(self.topo_path, 'rb') as f:
            topo = pickle.load(f)
        with open(self.data_path, 'rb') as f:
            data_part = pickle.load(f)

        self.sensors: Dict[int, SensorWorkerSOTA] = {}
        self.fog_nodes: Dict[int, FogNodeSOTA]     = {}
        self.association: Dict[int, int]            = {}
        self.G = topo.G

        # Tạo Fogs
        fog_ids = list(set(v for _, v in topo.fog_sensor_pairs if v != -1))
        for fid in fog_ids:
            self.fog_nodes[fid] = FogNodeSOTA(fog_id=fid)

        # Tạo Sensors
        client_yamls = data_part.client_yamls
        for s_id, client_yaml in client_yamls.items():
            fog_id = topo.sensor_fog_map.get(s_id, -1)
            self.association[s_id] = fog_id
            self.sensors[s_id] = SensorWorkerSOTA(
                sensor_id=s_id,
                client_yaml=client_yaml,
                battery_init=energy_cfg.E_INIT,
                teacher_model=self.teacher_model,
            )

        self.gateway = BaseGateway(gateway_id=0)
        print(f"[SimulatorSOTA] {len(self.sensors)} sensors, {len(self.fog_nodes)} fogs.")

    def run(self, T_rounds: int = 50, **kwargs) -> dict:
        """Vòng lặp FL chính."""
        history = {
            'round': [], 'loss': [], 'mAP50-95': [], 'mAP50': [],
            'Prec': [], 'Rec': [], 'alive': [],
            'tau_round_s': [], 'tau_cumul_s': [], 'tau_s2f': [],
            'tau_f2f': [], 'tau_f2g': [], 'tau_comp': [],
            'avg_payload_kb': [], 'payload_cumul_kb': [],
            'e_total': [], 'e_s2f': [], 'e_f2f': [], 'e_f2g': [],
            'e_comp': [], 'e_cumul': [],
        }
        cumul_tau = 0.0
        cumul_payload_kb = 0.0
        cumul_energy = 0.0

        for rnd in range(1, T_rounds + 1):
            lr = self._lr_schedule[rnd - 1]
            epochs = fed_cfg.LOCAL_EPOCHS
            print(f"\n{'='*60}")
            print(f"[SimulatorSOTA] Round {rnd}/{T_rounds} | lr={lr:.6f}")

            # ── Phase 1: Local Train (Sensor) ──────────────────────────────
            payloads: Dict[int, dict] = {}
            train_losses = []
            e_comp_total = 0.0
            tau_comp_max = 0.0

            for s_id, sensor in self.sensors.items():
                if not sensor.alive:
                    continue
                full_state, payload_kb, loss = sensor.train_and_get_payload(
                    global_state=self.global_state,
                    epochs=epochs, lr=lr, device=self.device,
                )
                if full_state is None:
                    continue
                payloads[s_id] = full_state
                train_losses.append(loss)

                # Tính energy & latency cho Sensor (comp)
                tau_c = comp_delay_dynamic(
                    n_samples=sensor.n_samples,
                    n_local_epochs=epochs,
                    flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                    flop_multiplier=2.4,   # SOTA: full model + teacher forward
                    f_cpu=energy_cfg.F_CPU,
                )
                e_c = e_comp_dynamic(
                    n_samples=sensor.n_samples,
                    n_local_epochs=epochs,
                    flops_per_sample=fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"],
                    epsilon_op=energy_cfg.EPSILON_OP["2D"],
                    flop_multiplier=2.4,
                )
                tau_comp_max = max(tau_comp_max, tau_c)
                e_comp_total += e_c
                sensor.drain_battery(e_c)

            # ── Phase 2: Fog Aggregation ────────────────────────────────────
            fog_states: Dict[int, dict] = {}
            e_s2f_total, tau_s2f_max = 0.0, 0.0

            for s_id, full_state in payloads.items():
                fog_id = self.association.get(s_id, -1)
                if fog_id == -1:
                    continue
                # Năng lượng & latency truyền S→F (5.4 MB nặng!)
                key = ('sensor', s_id, 'fog', fog_id)
                if key in self.G:
                    link = self.G[key]
                    tau_link = comm_delay(SOTA_PAYLOAD_BITS, link.R_bps, link.distance)
                    e_link   = e_tx(SOTA_PAYLOAD_BITS, link.R_bps, link.SL_min,
                                    energy_cfg.ETA_EA, energy_cfg.P_C_TX)
                    tau_s2f_max = max(tau_s2f_max, tau_link)
                    e_s2f_total += e_link
                    self.sensors[s_id].drain_battery(e_link)

                fog = self.fog_nodes.get(fog_id)
                if fog:
                    fog.receive(s_id, full_state)

            for fog_id, fog in self.fog_nodes.items():
                agg = fog.aggregate_intra_cluster()
                if agg:
                    fog_states[fog_id] = agg
                fog.reset()

            # ── Phase 3: Global Aggregation (no KD at Gateway) ─────────────
            e_f2g_total, tau_f2g_max = 0.0, 0.0
            e_f2f_total, tau_f2f_max = 0.0, 0.0

            if fog_states:
                keys_all = list(next(iter(fog_states.values())).keys())
                n_fogs = len(fog_states)
                new_global = {}
                for k in keys_all:
                    new_global[k] = sum(
                        sd[k].float() for sd in fog_states.values()
                    ) / n_fogs
                self.global_state = new_global

                # Energy & latency F→G (5.4 MB!)
                for fog_id in fog_states:
                    key = ('fog', fog_id, 'gateway', 0)
                    if key in self.G:
                        link = self.G[key]
                        tau_l = comm_delay(SOTA_PAYLOAD_BITS, link.R_bps, link.distance)
                        e_l   = e_tx(SOTA_PAYLOAD_BITS, link.R_bps, link.SL_min,
                                     energy_cfg.ETA_EA, energy_cfg.P_C_TX)
                        tau_f2g_max = max(tau_f2g_max, tau_l)
                        e_f2g_total += e_l

            # ── Phase 4: Evaluate ───────────────────────────────────────────
            nc = self._get_nc()
            eval_student = StudentModel(
                "yolo11n.pt", rank=4, nc=nc,
                full_param=True, use_lora=False
            )
            eval_student.load_trainable_state_dict(self.global_state)
            metrics = evaluate_od_sota(eval_student, self.test_yaml, self.device)
            del eval_student
            gc.collect()
            torch.cuda.empty_cache()

            # ── Logging ─────────────────────────────────────────────────────
            tau_round = tau_comp_max + tau_s2f_max + tau_f2f_max + tau_f2g_max
            e_total   = e_comp_total + e_s2f_total + e_f2f_total + e_f2g_total
            alive_cnt = sum(1 for s in self.sensors.values() if s.alive)
            payload_kb_round = SOTA_PAYLOAD_KB * len(payloads)

            cumul_tau       += tau_round
            cumul_payload_kb += payload_kb_round
            cumul_energy    += e_total

            avg_loss = float(np.mean(train_losses)) if train_losses else 0.0

            tau_status = "OK" if tau_round <= fed_cfg.TAU_MAX else "VIOLATED"
            print(
                f"Round {rnd} | loss: {avg_loss:.4f} | alive: {alive_cnt} "
                f"| mAP50-95: {metrics['mAP50-95']:.4f} | mAP50: {metrics['mAP50']:.4f} "
                f"| tau_round_s: {tau_round:.1f}s ({tau_status}) | avg_payload_kb: {SOTA_PAYLOAD_KB:.1f} KB"
            )

            history['round'].append(rnd)
            history['loss'].append(avg_loss)
            history['mAP50-95'].append(metrics['mAP50-95'])
            history['mAP50'].append(metrics['mAP50'])
            history['Prec'].append(metrics['Prec'])
            history['Rec'].append(metrics['Rec'])
            history['alive'].append(alive_cnt)
            history['tau_round_s'].append(tau_round)
            history['tau_cumul_s'].append(cumul_tau)
            history['tau_s2f'].append(tau_s2f_max)
            history['tau_f2f'].append(tau_f2f_max)
            history['tau_f2g'].append(tau_f2g_max)
            history['tau_comp'].append(tau_comp_max)
            history['avg_payload_kb'].append(SOTA_PAYLOAD_KB)
            history['payload_cumul_kb'].append(cumul_payload_kb)
            history['e_total'].append(e_total)
            history['e_s2f'].append(e_s2f_total)
            history['e_f2f'].append(e_f2f_total)
            history['e_f2g'].append(e_f2g_total)
            history['e_comp'].append(e_comp_total)
            history['e_cumul'].append(cumul_energy)

        return history
