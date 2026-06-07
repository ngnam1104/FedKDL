import sys, os
import json
import numpy as np

sys.path.insert(0, os.getcwd())

from config.settings import fed_cfg, acoustic_cfg, energy_cfg
from federated_core.metrics import LatencyTracker
from physics_models.topology import LinkInfo
from physics_models.communication import shannon_capacity
from physics_models.energy import e_comp, e_tx
from physics_models.latency import comp_delay_dynamic, relay_comp_delay

def main():
    tracker = LatencyTracker(sound_speed=acoustic_cfg.SOUND_SPEED, time_per_epoch=0.1)

    # Băng thông và tốc độ
    R_bps = shannon_capacity(acoustic_cfg.BANDWIDTH, acoustic_cfg.TARGET_SNR)

    # Topology: 3 AUVs (straggler=1000 imgs), 2 Relays, 1 Gateway
    association = {1: 1, 2: 2, 3: 1}
    cooperation_partners = {2: 1}

    def make_link(dist):
        return LinkInfo(distance=dist, SL_min=100.0, TL=10.0, NL=50.0, R_bps=R_bps)

    G = {
        ('auv', 1, 'relay', 1): make_link(500.0),
        ('auv', 2, 'relay', 2): make_link(500.0),
        ('auv', 3, 'relay', 1): make_link(500.0),
        ('relay', 2, 'relay', 1): make_link(1000.0),
        ('relay', 1, 'gateway', 0): make_link(1000.0)
    }

    flops_per_sample = fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"]
    f_cpu = energy_cfg.F_CPU
    auv_samples = {1: 155, 2: 155, 3: 1000}
    max_samples = max(auv_samples.values())
    E_INIT = energy_cfg.E_INIT

    # ── Payload calculation ──────────────────────────────────────────────────────
    # YOLOv12n total params = 2,695,948  (từ dump_trainable.py)
    TOTAL_PARAMS = 2_695_948
    K = int(TOTAL_PARAMS * 0.05)          # 5% Top-K → 134,797 params

    # FedKDL / Naive LoRA: 297,708 × 1B INT8 = 290.73 KB (từ check_payload_final.py)
    fedkdl_kb = 290.73

    # Full Parameter: TOTAL_PARAMS × 4B FP32 (không nén, không lượng tử hóa)
    full_param_kb = TOTAL_PARAMS * 4 / 1024

    # Top-K (5%): K values (INT8, 1B) + K indices (INT32, 4B)
    # Bắt buộc phải gửi index để receiver biết vị trí tham số cần cộng vào
    topk_values_kb  = K * 1 / 1024   # ~131.6 KB
    topk_indices_kb = K * 4 / 1024   # ~526.2 KB
    topk_kb = topk_values_kb + topk_indices_kb

    print(f"[Payload] FedKDL     = {fedkdl_kb:.1f} KB (LoRA+Head INT8, structured)")
    print(f"[Payload] Full Param = {full_param_kb:.1f} KB ({TOTAL_PARAMS:,} × FP32)")
    print(f"[Payload] Top-K 5%   = {topk_kb:.1f} KB ({K:,} values + {K:,} INT32 indices)")
    print()

    scenarios = {
        'FedKDL (SVD-LoRA)':      {'payload_kb': fedkdl_kb,    'flop_mult': 1.5, 'has_svd': True},
        'Full Parameter (FedAvg)': {'payload_kb': full_param_kb, 'flop_mult': 3.0, 'has_svd': False},
        'Top-K (5%)':             {'payload_kb': topk_kb,       'flop_mult': 3.0, 'has_svd': False},
        'Naive LoRA':             {'payload_kb': fedkdl_kb,     'flop_mult': 1.5, 'has_svd': False},
    }

    results = {}
    eps_op = energy_cfg.EPSILON_OP.get("2D", 1e-28)

    for name, cfg in scenarios.items():
        auv_payload_bits = cfg['payload_kb'] * 1024 * 8
        relay_model_bits = cfg['payload_kb'] * 1024 * 8

        tau_comp = comp_delay_dynamic(
            n_samples=max_samples,
            n_local_epochs=fed_cfg.LOCAL_EPOCHS,
            flops_per_sample=flops_per_sample,
            flop_multiplier=cfg['flop_mult'],
            f_cpu=f_cpu,
            n_cores=energy_cfg.N_CORES,
            flops_per_cycle=energy_cfg.FLOPS_PER_CYCLE
        )

        tau_svd = relay_comp_delay(
            f_cpu=f_cpu, n_cores=energy_cfg.N_CORES,
            flops_per_cycle=energy_cfg.FLOPS_PER_CYCLE
        ) if cfg['has_svd'] else 0.0

        latency_info = tracker.compute_round_latency(
            G=G, association=association, cooperation_partners=cooperation_partners,
            tau_comp=tau_comp, tau_svd=tau_svd,
            auv_payload_bits=auv_payload_bits, relay_model_bits=relay_model_bits
        )

        tau_round = latency_info['tau_round']
        status = 'OK' if tau_round <= fed_cfg.TAU_MAX else 'VIOLATED (Max 1800s)'

        # Energy — dùng hàm chính thức từ physics_models.energy
        e_comp_cost = e_comp(
            n_samples=max_samples,
            local_epochs=fed_cfg.LOCAL_EPOCHS,
            flops_per_sample=flops_per_sample,
            epsilon_op=eps_op,
            flop_multiplier=cfg['flop_mult'],
            f_cpu=f_cpu
        )
        e_comm_cost = e_tx(
            S_bits=auv_payload_bits,
            R_bps=R_bps,
            SL_min_dB=100.0,
            eta_ea=energy_cfg.ETA_EA,
            P_c_tx=energy_cfg.P_C_TX
        )
        e_total = e_comp_cost + e_comm_cost

        results[name] = {
            'payload_kb':          round(cfg['payload_kb'], 2),
            'tau_comp_s':          round(tau_comp, 2),
            'tau_a2r_s':           round(latency_info['tau_a2r'], 2),
            'tau_r2r_s':           round(latency_info['tau_r2r'], 2),
            'tau_r2g_s':           round(latency_info['tau_r2g'], 2),
            'tau_round_s':         round(tau_round, 2),
            'latency_status':      status,
            'e_comp_J':            round(e_comp_cost, 2),
            'e_comm_J':            round(e_comm_cost, 2),
            'e_total_J':           round(e_total, 2),
            'battery_init_J':      E_INIT,
            'survival_rounds_est': round(E_INIT / e_total, 1) if e_total > 0 else 0
        }

    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
