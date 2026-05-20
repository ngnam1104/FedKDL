"""
simulator_od.py
Simulator cho Kịch bản 2 (bottleneck) & Kịch bản 3 (FedKDL recovery).
Kế thừa hoàn toàn physics_models/ — không sửa bất kỳ file nào của physics_models.

Luồng mỗi round:
  1. Sensor: local_sgd_od → lazy_filter → pack_payload (INT8) → E_tx
  2. Fog:    unpack → FedAvg ({LoRA, Head}) → HFL-Selective check
  3. Gateway: FedAvg global → broadcast trainable state
"""
import copy
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

# ── Config (đã có sẵn trong settings.py) ──────────────────────────────
from config.settings import (
    NetworkConfig, AcousticChannelConfig, EnergyConfig, FedKDLConfig,
    network_cfg, acoustic_cfg, energy_cfg, fed_cfg,
)

# ── Physics Models (giữ nguyên, không sửa) ────────────────────────────
from physics_models.topology import (
    Topology3D, build_feasibility_graph,
    nearest_feasible_association, flat_topology_association, build_clusters
)
from kdl_core.algorithms.knowledge_association import knowledge_aware_association
from physics_models.energy import e_tx, e_comp_simple
from physics_models.communication import shannon_capacity

# ── FL Core ───────────────────────────────────────────────────────────
from kdl_core.models.yolo_wrapper import StudentModel, TeacherModel
from kdl_core.algorithms.local_trainer_od import local_sgd_od, evaluate_od
from kdl_core.knowledge_compression.int8_quantization import pack_payload, unpack_payload
from kdl_core.algorithms.hfl_rules import (
    should_cooperate, find_coop_partner, blend_state_dicts, compute_q1_fog_distance
)
from kdl_core.algorithms.concept_drift import ConceptDriftMonitor


class ODSimulator:
    """
    Simulator vòng lặp Federated Learning cho Object Detection.
    Hỗ trợ cả Kịch bản 2 (no compression) và Kịch bản 3 (KD-LoRA-INT8).
    """

    def __init__(
        self,
        client_yamls: List[str],
        test_yaml: str,
        student_ckpt: str = "yolo26n.pt",
        teacher_ckpt: str = "yolo12l.pt",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.client_yamls = client_yamls
        self.test_yaml = test_yaml
        self.device = device

        # ── Topology & Physics ─────────────────────────────────────────
        self.topology = Topology3D(network_cfg, acoustic_cfg, seed=seed)
        self.G = build_feasibility_graph(self.topology, acoustic_cfg)

        # ── Models ────────────────────────────────────────────────────
        print("[ODSimulator] Đang tải Teacher model...")
        self.teacher = TeacherModel(teacher_ckpt)

        print("[ODSimulator] Đang tải Student model (YOLO26n + LoRA)...")
        self.global_student = StudentModel(student_ckpt, rank=fed_cfg.LORA_RANK)
        self.global_state = self.global_student.trainable_state_dict()

        # ── Battery & Survival tracking ────────────────────────────────
        N = network_cfg.N_SENSORS
        self.battery = {i: energy_cfg.E_INIT for i in range(N)}
        self.alive = set(range(N))
        self.drift_monitors = {i: ConceptDriftMonitor() for i in range(N)}

    # ──────────────────────────────────────────────────────────────────
    def _drain_battery(self, sensor_id: int, cost: float) -> bool:
        """Trừ năng lượng. Trả False nếu AUV đã chết."""
        if self.battery[sensor_id] >= cost:
            self.battery[sensor_id] -= cost
            return True
        self.alive.discard(sensor_id)
        return False

    def _get_link_info(self, sensor_id: int, fog_id: int):
        """Lấy LinkInfo từ feasibility graph."""
        key = ('sensor', sensor_id, 'fog', fog_id)
        return self.G.get(key, None)

    # ──────────────────────────────────────────────────────────────────
    def run(
        self,
        baseline: str = "hfl_selective",
        use_kd_lora_int8: bool = True,
    ) -> dict:
        """
        Chạy toàn bộ vòng lặp FL.

        baseline: 'fedavg', 'hfl_nocoop', 'hfl_selective', 'hfl_nearest'
        use_kd_lora_int8: True = Kịch bản 3, False = Kịch bản 2
        """
        print(f"\n{'='*60}")
        print(f"Baseline: {baseline} | KD-LoRA-INT8: {use_kd_lora_int8}")
        print(f"{'='*60}")

        history = defaultdict(list)

        # Giả lập Label Histograms cho D_joint EMD
        np.random.seed(42)
        num_classes = 10
        sensor_hists = np.random.dirichlet(np.ones(num_classes) * fed_cfg.NON_IID_ALPHA, network_cfg.N_SENSORS)
        fog_hists = np.ones((network_cfg.M_FOGS, num_classes)) / num_classes

        # Tính association một lần (quasi-static topology)
        if baseline in ('hfl_selective', 'hfl_nearest', 'hfl_nocoop'):
            association = knowledge_aware_association(
                self.topology, self.G, sensor_hists, fog_hists, beta=fed_cfg.BETA_EMD
            )
        else:
            association = flat_topology_association(self.topology, self.G)

        clusters = build_clusters(association, network_cfg.M_FOGS)

        for rnd in range(fed_cfg.GLOBAL_ROUNDS):
            if not self.alive:
                print("Tất cả AUV đã hết pin! Dừng mô phỏng.")
                break

            print(f"\n── Round {rnd + 1}/{fed_cfg.GLOBAL_ROUNDS} | Alive: {len(self.alive)} ──")

            # ── Sensor Tier: Local training ───────────────────────────
            local_updates: Dict[int, dict] = {}
            total_payload_kb = 0.0

            for sid in list(self.alive):
                if sid >= len(self.client_yamls):
                    continue

                # Nạp global state vào student
                self.global_student.load_trainable_state_dict(self.global_state)

                # Local SGD (với hoặc không có KD)
                new_state, delta = local_sgd_od(
                    student_model=self.global_student,
                    teacher_model=self.teacher,
                    client_yaml=self.client_yamls[sid],
                    client_id=sid,
                    epochs=fed_cfg.LOCAL_EPOCHS,
                    device=self.device,
                    kd_lambda=1.0 if use_kd_lora_int8 else 0.0,
                    use_kd=use_kd_lora_int8,
                )

                if new_state is None:
                    # Bị Lazy Filter giữ lại — tiết kiệm E_tx
                    continue
                    
                # Update drift monitor (dùng delta norm làm proxy cho loss biến thiên ở kịch bản OD)
                delta_norm = float(torch.sum(delta ** 2))
                self.drift_monitors[sid].update(sid, delta_norm)

                # Tính payload size
                if use_kd_lora_int8:
                    payload_bytes, payload_kb = pack_payload(new_state)
                    S_bits = len(payload_bytes) * 8
                else:
                    # Gửi toàn bộ float32 — bottleneck simulation
                    S_bits = sum(t.numel() for t in new_state.values()) * 32
                    payload_kb = S_bits / 8 / 1024

                total_payload_kb += payload_kb

                # Tính E_tx (qua Fog gần nhất)
                fog_id = association.get(sid)
                if fog_id is None or fog_id < 0:
                    fog_id = 0
                link = self._get_link_info(sid, fog_id)

                if link is None:
                    # Không có link khả thi → AUV cô lập
                    self.alive.discard(sid)
                    continue

                E_communication = e_tx(
                    S_bits, link.R_bps, link.SL_min,
                    eta_ea=energy_cfg.ETA_EA,
                    P_c_tx=energy_cfg.P_C_TX,
                )
                E_computation = e_comp_simple(fed_cfg.LOCAL_EPOCHS,
                                              energy_cfg.E_COMP_EPOCH)

                if self._drain_battery(sid, E_communication + E_computation):
                    local_updates[sid] = new_state
                else:
                    print(f"  AUV {sid} hết pin "
                          f"(payload={payload_kb:.1f} KB, E_tx={E_communication:.2f} J)")

            # ── Fog/Gateway Aggregation ───────────────────────────────
            if local_updates:
                if baseline in ('fedavg', 'fedprox'):
                    # Flat aggregation
                    aggregated = {}
                    first_k = list(local_updates.values())[0]
                    for k in first_k:
                        stacked = torch.stack([u[k].float() for u in local_updates.values()])
                        aggregated[k] = stacked.mean(dim=0)
                    self.global_state = aggregated
                else:
                    # 1. Fog Intra-cluster Aggregation
                    fog_states = {}
                    cluster_sizes = {m: len(c) for m, c in clusters.items()}
                    for m, c in clusters.items():
                        c_updates = [local_updates[sid] for sid in c if sid in local_updates]
                        if c_updates:
                            fog_states[m] = {}
                            for k in c_updates[0]:
                                fog_states[m][k] = torch.stack([u[k].float() for u in c_updates]).mean(dim=0)
                    
                    # 2. HFL Cooperation
                    q1_dist = compute_q1_fog_distance(self.G)
                    mean_c = np.mean([s for s in cluster_sizes.values() if s > 0]) if any(s>0 for s in cluster_sizes.values()) else 1.0
                    coop_rule = 'selective' if baseline == 'hfl_selective' else ('nearest' if baseline == 'hfl_nearest' else 'nocoop')
                    
                    final_fog_states = {}
                    for m, state in fog_states.items():
                        final_fog_states[m] = copy.deepcopy(state)
                        if coop_rule == 'nocoop':
                            continue
                        if coop_rule == 'selective' and not should_cooperate(cluster_sizes[m], mean_c):
                            continue
                            
                        alpha = 0.8 if coop_rule == 'selective' else 0.7
                        dist_filter = q1_dist if coop_rule == 'selective' else None
                        partner_id = find_coop_partner(m, cluster_sizes, self.G, dist_filter)
                        
                        if partner_id is not None and partner_id in fog_states:
                            final_fog_states[m] = blend_state_dicts(state, fog_states[partner_id], alpha)
                            
                    # 3. Global Aggregation
                    if final_fog_states:
                        self.global_state = {}
                        total_samples = sum(cluster_sizes[m] for m in final_fog_states.keys())
                        for k in list(final_fog_states.values())[0]:
                            weighted_sum = sum(
                                final_fog_states[m][k].float() * (cluster_sizes[m] / max(1, total_samples))
                                for m in final_fog_states.keys()
                            )
                            self.global_state[k] = weighted_sum
            
            # --- Concept Drift Checking ---
            if baseline in ('hfl_selective', 'hfl_nearest'):
                drift_detected = any(self.drift_monitors[sid].check_drift(sid) for sid in self.alive)
                if drift_detected:
                    print(f"  ⚠️ Concept Drift detected! Reseting error buffers.")
                    for sid in self.alive:
                        self.drift_monitors[sid].clear()

            # ── Đánh giá mAP ──────────────────────────────────────────
            self.global_student.load_trainable_state_dict(self.global_state)
            map_score = evaluate_od(self.global_student, self.test_yaml, self.device)
            avg_payload = total_payload_kb / max(1, len(local_updates))
            E_cumul = sum(energy_cfg.E_INIT - b for b in self.battery.values())

            history['round'].append(rnd)
            history['map'].append(map_score)
            history['alive'].append(len(self.alive))
            history['avg_payload_kb'].append(avg_payload)
            history['energy_cumul_J'].append(E_cumul)

            print(f"  mAP: {map_score:.4f} | Avg Payload: {avg_payload:.1f} KB "
                  f"| E_cumul: {E_cumul:.0f} J")

        return dict(history)
