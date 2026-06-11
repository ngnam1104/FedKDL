"""
compute_theoretical_metrics.py
Tính toán đầy đủ tất cả metrics lý thuyết cho Section 3 / 5.2–5.5 của luận văn.

Nguồn chính thức:
  - Topology: environments/2d/topo/N_30/topo_N30_seed1104.pkl
  - Physics:  physics_models.{communication, energy, latency}
  - Config:   config.settings
"""
import numpy as np
import pickle

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from physics_models.communication import min_source_level, shannon_capacity
from physics_models.energy import e_tx, e_comp, e_rx, e_svd, total_energy_round
from physics_models.latency import comm_delay, comp_delay_dynamic, relay_comp_delay
from utils.env_manager import EnvironmentManager

# ─────────────────────────────────────────────────────────────────────
# 1. Load Topology
# ─────────────────────────────────────────────────────────────────────
TOPO_PATH = "environments/2d/topo/N_30/topo_N30_seed1104.pkl"
DATA_PATH = "environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1104.pkl"
with open(TOPO_PATH, "rb") as f:
    topo = pickle.load(f)
if topo.N != network_cfg.N_AUVS or topo.M != network_cfg.M_RELAYS_2D:
    raise ValueError(
        f"Stale topology {TOPO_PATH}: N={topo.N}, M={topo.M}; "
        f"expected N={network_cfg.N_AUVS}, M={network_cfg.M_RELAYS_2D}. "
        "Regenerate it with utils/generate_all_envs.py before computing metrics."
    )
with open(DATA_PATH, "rb") as f:
    data_partition = pickle.load(f)
G = EnvironmentManager.restore_graph(topo)

# ─────────────────────────────────────────────────────────────────────
# 2. System / Physics Constants  (tất cả lấy từ settings, không hard-code)
# ─────────────────────────────────────────────────────────────────────
B_hz           = acoustic_cfg.BANDWIDTH
SNR_dB         = acoustic_cfg.TARGET_SNR
F_KHZ          = acoustic_cfg.CARRIER_FREQ
WIND           = acoustic_cfg.WIND_SPEED
SHIPPING       = acoustic_cfg.SHIPPING_FACTOR
IL             = acoustic_cfg.IL_LOSS
SPREADING      = acoustic_cfg.SPREADING_FACTOR
C_S            = acoustic_cfg.SOUND_SPEED
ETA_EA         = energy_cfg.ETA_EA
P_C_TX         = energy_cfg.P_C_TX
P_C_RX         = energy_cfg.P_C_RX
# Giả lập AUV dùng chip nhúng (Edge TPU/NPU) thay vì Jetson to:
# Giữ nguyên F_CPU, chỉ giảm Epsilon cực thấp để E_comp < E_comm
F_CPU          = energy_cfg.F_CPU
N_CORES        = energy_cfg.N_CORES
FPC            = energy_cfg.FLOPS_PER_CYCLE          # flops_per_cycle
EPS_OP         = energy_cfg.EPSILON_OP["2D"]
E_INIT_REF     = energy_cfg.E_INIT_REF

R_BPS          = shannon_capacity(B_hz, SNR_dB)

FLOPS_SAMPLE   = fed_cfg.MODEL_FLOPS_PER_SAMPLE["2D"]   # 2.175e9
LOCAL_EPOCHS   = fed_cfg.LOCAL_EPOCHS                    # 3
TAU_MAX        = fed_cfg.TAU_MAX                         # 1800 s

# Cân bằng đóng góp của tau và E trong Joint Cost:
#   tau_round  ~ 300 – 1000 s
#   E_total    ~ 5,000 – 20,000 J   (tỷ lệ E/tau ≈ 20x)
# Để F = lambda_tau * tau + lambda_E * E có hai hạng tử cùng bậc đại lượng:
#   lambda_tau = 1e-3  (s^-1)
#   lambda_E   = 1e-4  (J^-1)
LAMBDA_TAU     = fed_cfg.LAMBDA_TAU
LAMBDA_E       = fed_cfg.LAMBDA_E

# Model params (YOLOv12-N student)
NUM_PARAMS_FULL  = 2_695_948   # tổng params
NUM_PARAMS_LORA  = 389_772     # trainable (Full backbone r=4, Neck r=8 + Head)

SAMPLE_COUNTS = {
    auv_id: len(indices)
    for auv_id, indices in data_partition.auv_data_indices.items()
}
MAX_SAMPLES = max(SAMPLE_COUNTS.values())

# ─────────────────────────────────────────────────────────────────────
# 3. Helper Functions
# ─────────────────────────────────────────────────────────────────────

def dist3(p1, p2) -> float:
    return float(np.linalg.norm(p1 - p2))

def payload_kb(n_params: int, bits: int) -> float:
    """Payload của n_params tham số được lượng tử hoá thành 'bits' bit."""
    return n_params * bits / 8 / 1024

def topk_payload_kb(k_ratio: float) -> float:
    """
    Top-K Sparsification:
      - K values  → INT8  (1 byte each)
      - K indices → INT32 (4 bytes each)  ← bắt buộc để receiver biết vị trí
    """
    K = int(NUM_PARAMS_FULL * k_ratio)
    return (K * 1 + K * 4) / 1024   # KB

def sl_min(d: float) -> float:
    return min_source_level(d, F_KHZ, B_hz, SNR_dB, IL, SPREADING, WIND, SHIPPING)

def tx_energy(payload_kb_val: float, d: float) -> float:
    S_bits = payload_kb_val * 1024 * 8
    return e_tx(S_bits, R_BPS, sl_min(d), ETA_EA, P_C_TX, 1025.0, C_S)

def tx_latency(payload_kb_val: float, d: float) -> float:
    S_bits = payload_kb_val * 1024 * 8
    return comm_delay(S_bits, R_BPS, d, C_S)

def link_physics(payload_kb_val: float, d: float, link):
    """Return latency, TX energy, and RX energy using a restored graph link."""
    s_bits = payload_kb_val * 1024 * 8
    latency = comm_delay(s_bits, link.R_bps, d, C_S)
    tx = e_tx(s_bits, link.R_bps, link.SL_min, ETA_EA, P_C_TX, 1025.0, C_S)
    rx = e_rx(s_bits, link.R_bps, P_C_RX)
    return latency, tx, rx

def nearest_relay_partner(relay_id: int):
    candidates = []
    for other_id in range(topo.M):
        if other_id == relay_id:
            continue
        key_fwd = ('relay', relay_id, 'relay', other_id)
        key_bwd = ('relay', other_id, 'relay', relay_id)
        if key_fwd in G or key_bwd in G:
            d = dist3(topo.relay_positions[relay_id], topo.relay_positions[other_id])
            candidates.append((d, other_id, key_fwd if key_fwd in G else key_bwd))
    return min(candidates) if candidates else None

def local_comp_delay(flop_mult: float, n_samples: int = MAX_SAMPLES) -> float:
    """tau_comp cho một AUV."""
    return comp_delay_dynamic(n_samples, LOCAL_EPOCHS, FLOPS_SAMPLE,
                               flop_mult, F_CPU, N_CORES, FPC)

def local_comp_energy(flop_mult: float, n_samples: int = MAX_SAMPLES) -> float:
    """e_comp cho một AUV."""
    return e_comp(n_samples, LOCAL_EPOCHS, FLOPS_SAMPLE, EPS_OP, flop_mult, F_CPU)

def svd_delay() -> float:
    """tau_svd tại Relay (2 lần SVD mỗi vòng)."""
    return relay_comp_delay(f_cpu=F_CPU, n_cores=N_CORES, flops_per_cycle=FPC)

# ─────────────────────────────────────────────────────────────────────
# 4. Core Topology-based Computation
# ─────────────────────────────────────────────────────────────────────

def compute_flat_fl(pload_kb: float, flop_mult: float):
    tau_comp = local_comp_delay(flop_mult)
    total_e_comm = 0.0
    total_e_comp = 0.0
    max_tau_comm = 0.0
    max_e_auv = 0.0

    for i in range(topo.N):
        d = dist3(topo.auv_positions[i], topo.gateway_position)
        key = ('auv', i, 'gateway', 0)
        if key not in G:
            continue
        t, e, e_recv = link_physics(pload_kb, d, G[key])
        e_comp_i = local_comp_energy(flop_mult, SAMPLE_COUNTS[i])
        
        total_e_comm += e + e_recv
        total_e_comp += e_comp_i
        
        # Max energy for a single AUV (chỉ tính tx và comp của nó)
        auv_e = e_comp_i + e
        if auv_e > max_e_auv:
            max_e_auv = auv_e
            
        if t > max_tau_comm:
            max_tau_comm = t

    tau_round    = tau_comp + max_tau_comm
    total_energy = total_e_comp + total_e_comm
    return tau_round, total_energy, max_e_auv


def compute_hfl(pload_kb_auv: float, pload_kb_relay: float,
                flop_mult: float, has_svd: bool = False,
                has_coop: bool = False):
    tau_comp  = local_comp_delay(flop_mult)
    svd_calls = 2 if has_coop else 1
    tau_svd_v = (
        relay_comp_delay(
            n_svd_calls=svd_calls,
            f_cpu=F_CPU,
            n_cores=N_CORES,
            flops_per_cycle=FPC,
        )
        if has_svd else 0.0
    )

    relay_max_tau_comm = {m: 0.0 for m in range(topo.M)}
    total_e_a2r        = 0.0
    total_e_comp       = 0.0
    max_e_auv          = 0.0

    relay_recv_energy = {m: 0.0 for m in range(topo.M)}

    for i in range(topo.N):
        m = topo.hfl_association.get(i, 0)
        d = dist3(topo.auv_positions[i], topo.relay_positions[m])
        key = ('auv', i, 'relay', m)
        if key not in G:
            raise KeyError(f"Missing topology link: {key}")
        t, e, e_recv = link_physics(pload_kb_auv, d, G[key])
        e_comp_i = local_comp_energy(flop_mult, SAMPLE_COUNTS[i])
        
        total_e_a2r += e + e_recv
        total_e_comp += e_comp_i
        relay_recv_energy[m] += e_recv
        
        auv_e = e_comp_i + e
        if auv_e > max_e_auv:
            max_e_auv = auv_e
            
        if t > relay_max_tau_comm[m]:
            relay_max_tau_comm[m] = t

    total_e_r2r      = 0.0
    relay_r2r_latency = {m: 0.0 for m in range(topo.M)}
    relay_r2r_energy = {m: 0.0 for m in range(topo.M)}
    if has_coop:
        for m in range(topo.M):
            partner = nearest_relay_partner(m)
            if partner is None:
                continue
            d, partner_id, key = partner
            t, e_tx_cost, e_rx_cost = link_physics(pload_kb_relay, d, G[key])
            relay_r2r_latency[m] = t
            relay_r2r_energy[m] = e_tx_cost + e_rx_cost
            total_e_r2r += e_tx_cost + e_rx_cost

    total_e_r2g      = 0.0
    round_max_latency = 0.0

    for m in range(topo.M):
        d   = dist3(topo.relay_positions[m], topo.gateway_position)
        key = ('relay', m, 'gateway', 0)
        t = e = e_recv = 0.0
        if key in G:
            t, e, e_recv = link_physics(pload_kb_relay, d, G[key])
            total_e_r2g += e + e_recv

        relay_total_e = relay_recv_energy[m] + relay_r2r_energy[m] + e
        if relay_total_e > max_e_auv:
            max_e_auv = relay_total_e

        relay_total = (
            tau_comp + relay_max_tau_comm[m] + tau_svd_v
            + relay_r2r_latency[m] + t
        )
        if relay_total > round_max_latency:
            round_max_latency = relay_total

    total_e_svd = (
        topo.M * e_svd(256, 128, EPS_OP, svd_calls, F_CPU)
        if has_svd else 0.0
    )
    total_energy = total_energy_round(
        total_e_a2r, total_e_r2r, total_e_r2g, total_e_comp, total_e_svd
    )
    return round_max_latency, total_energy, max_e_auv


def joint_cost(tau: float, energy: float) -> float:
    return LAMBDA_TAU * tau + LAMBDA_E * energy


def survival_rounds(max_e_auv: float) -> float:
    return E_INIT_REF / max_e_auv if max_e_auv > 0 else float('inf')

def print_stage_physics(payload_auv_kb: float, payload_relay_kb: float):
    """Print every physical term for one FedKDL communication round."""
    link_rows = []
    print("\n[Physics breakdown]")
    print(
        f"{'Stage':7} {'Link':15} {'d(m)':>8} {'R(bps)':>10} {'SL(dB)':>9} "
        f"{'tx(s)':>10} {'prop(s)':>9} {'total(s)':>10} "
        f"{'E_tx(J)':>10} {'E_rx(J)':>10}"
    )

    def emit(stage, label, payload_kb_val, d, link):
        total_delay, tx_energy, rx_energy = link_physics(payload_kb_val, d, link)
        bits = payload_kb_val * 1024 * 8
        tx_delay = bits / link.R_bps
        prop_delay = d / C_S
        link_rows.append({
            'stage': stage,
            'link': label,
            'distance': d,
            'rate': link.R_bps,
            'source_level': link.SL_min,
            'tx_delay': tx_delay,
            'prop_delay': prop_delay,
            'total_delay': total_delay,
            'tx_energy': tx_energy,
            'rx_energy': rx_energy,
            'status': 'feasible',
        })
        print(
            f"{stage:7} {label:15} {d:8.2f} {link.R_bps:10.2f} "
            f"{link.SL_min:9.2f} {tx_delay:10.4f} {prop_delay:9.4f} "
            f"{total_delay:10.4f} {tx_energy:10.4f} {rx_energy:10.4f}"
        )
        return total_delay, tx_energy, rx_energy

    totals = {
        stage: {'latency': 0.0, 'tx': 0.0, 'rx': 0.0}
        for stage in ('A2R', 'R2R', 'R2G')
    }
    for i in range(topo.N):
        m = topo.hfl_association.get(i, 0)
        key = ('auv', i, 'relay', m)
        d = dist3(topo.auv_positions[i], topo.relay_positions[m])
        delay, tx, rx = emit('A2R', f'AUV{i}->R{m}', payload_auv_kb, d, G[key])
        totals['A2R']['latency'] = max(totals['A2R']['latency'], delay)
        totals['A2R']['tx'] += tx
        totals['A2R']['rx'] += rx

    for m in range(topo.M):
        partner = nearest_relay_partner(m)
        if partner is None:
            continue
        d, partner_id, key = partner
        delay, tx, rx = emit(
            'R2R', f'R{partner_id}->R{m}', payload_relay_kb, d, G[key]
        )
        totals['R2R']['latency'] = max(totals['R2R']['latency'], delay)
        totals['R2R']['tx'] += tx
        totals['R2R']['rx'] += rx

    for m in range(topo.M):
        key = ('relay', m, 'gateway', 0)
        d = dist3(topo.relay_positions[m], topo.gateway_position)
        if key not in G:
            link_rows.append({
                'stage': 'R2G',
                'link': f'R{m}->GW',
                'distance': d,
                'status': 'infeasible',
            })
            print(f"{'R2G':7} {f'R{m}->GW':15} {d:8.2f} {'N/A':>10} "
                  f"{'N/A':>9} {'SKIPPED: infeasible link':>41}")
            continue
        delay, tx, rx = emit('R2G', f'R{m}->GW', payload_relay_kb, d, G[key])
        totals['R2G']['latency'] = max(totals['R2G']['latency'], delay)
        totals['R2G']['tx'] += tx
        totals['R2G']['rx'] += rx

    print("\n[Stage totals: bottleneck latency, accumulated energy]")
    for stage, values in totals.items():
        print(
            f"{stage}: tau={values['latency']:.4f}s | "
            f"E_tx={values['tx']:.4f}J | E_rx={values['rx']:.4f}J | "
            f"E_link={values['tx'] + values['rx']:.4f}J"
        )
    return totals, link_rows

# ─────────────────────────────────────────────────────────────────────
# 5. Pre-compute payload sizes
# ─────────────────────────────────────────────────────────────────────
FULL_KB    = payload_kb(NUM_PARAMS_FULL, 32)     # 10531.0 KB
LORA_32_KB = payload_kb(NUM_PARAMS_LORA, 32)    # 1162.9 KB
LORA_INT8  = payload_kb(NUM_PARAMS_LORA, 8)     # 290.7  KB
TOPK5_KB   = topk_payload_kb(0.05)              # INT8 values + INT32 indices
TOPK1_KB   = topk_payload_kb(0.01)

print(f"[Payload] Full FP32    = {FULL_KB:.1f} KB")
print(f"[Payload] LoRA FP32    = {LORA_32_KB:.1f} KB")
print(f"[Payload] LoRA INT8    = {LORA_INT8:.1f} KB  <- FedKDL target")
print(f"[Payload] Top-K  5%    = {TOPK5_KB:.1f} KB  (INT8 values + INT32 idx)")
print(f"[Payload] Top-K  1%    = {TOPK1_KB:.1f} KB  (INT8 values + INT32 idx)")
print(f"[Shannon] R_bps        = {R_BPS:.1f} bps")
print(f"[Comp]    tau_comp(Full, flop×3.0) = {local_comp_delay(3.0):.2f} s")
print(f"[Comp]    tau_comp(LoRA, flop×1.5) = {local_comp_delay(1.5):.2f} s")
print(f"[Comp]    tau_svd                  = {svd_delay():.4f} s")
print()
PHYSICS_BREAKDOWN, PHYSICS_LINK_ROWS = print_stage_physics(
    LORA_INT8,
    LORA_INT8,
)
PHYSICS_TAU_COMP = local_comp_delay(1.5)
PHYSICS_E_COMP = sum(
    local_comp_energy(1.5, sample_count)
    for sample_count in SAMPLE_COUNTS.values()
)
PHYSICS_TAU_SVD = relay_comp_delay(
    n_svd_calls=2,
    f_cpu=F_CPU,
    n_cores=N_CORES,
    flops_per_cycle=FPC,
)
PHYSICS_E_SVD = topo.M * e_svd(256, 128, EPS_OP, 2, F_CPU)
PHYSICS_TAU_ROUND, PHYSICS_E_ROUND, _ = compute_hfl(
    LORA_INT8,
    LORA_INT8,
    1.5,
    has_svd=True,
    has_coop=True,
)

# ─────────────────────────────────────────────────────────────────────
# 6. Write Markdown
# ─────────────────────────────────────────────────────────────────────
with open("theoretical_metrics.md", "w", encoding="utf-8") as f:

    f.write("# Metrics Tính Toán Từ Mô Hình Hệ Thống (Section 3 / 5.2–5.5)\n\n")
    f.write(f"- **Model**: YOLOv12-N (Student) — {NUM_PARAMS_FULL:,} total params\n")
    f.write(f"- **Topology**: N={topo.N} AUVs, M={topo.M} Relays (from `{TOPO_PATH.split('/')[-1]}`)\n")
    f.write(f"- **Shannon Capacity** R = {R_BPS:.0f} bps\n")
    f.write(f"- **Max samples/AUV** = {MAX_SAMPLES} (from `{DATA_PATH.split('/')[-1]}`)\n\n")

    # ── 5.2 Dual Compression ─────────────────────────────────────────
    f.write("## 5.2 Dual Compression\n\n")
    f.write("Payload được tính dựa trên cấu trúc nén thực tế.\n"
            "Top-K truyền INT8 values **và** INT32 indices.\n\n")
    f.write("| Method | Payload (KB) | vs Full |\n")
    f.write("|---|---|---|\n")

    rows_52 = [
        ("Full Parameter FL (FP32)",    FULL_KB),
        ("Top-K Compression (5%, INT8 + idx)", TOPK5_KB),
        ("Top-K Compression (1%, INT8 + idx)", TOPK1_KB),
        ("LoRA Only (FP32)",            LORA_32_KB),
        ("LoRA + INT8 (FedKDL Tier-1)", LORA_INT8),
    ]
    for name, kb in rows_52:
        cr = FULL_KB / kb
        f.write(f"| {name} | {kb:.1f} | {cr:.1f}× |\n")

    # ── 5.3 Relay Aggregation ────────────────────────────────────────
    f.write("\n## 5.3 Relay Aggregation\n\n")
    f.write("Tất cả baselines đều dùng LoRA+INT8 payload để so sánh thuần tuý chiến lược tập hợp.\n\n")
    f.write("| Method | Payload A→R (KB) | Relay Op | tau_svd (s) | tau_round (s) | E_total (J) | Joint Cost |\n")
    f.write("|---|---|---|---|---|---|---|\n")

    rows_53 = [
        ("FedAvg-LoRA (no relay)",    LORA_INT8, LORA_INT8, "Average",        False, False, 3.0),
        ("Naive SVD-LoRA",            LORA_INT8, LORA_INT8, "SVD (no coop)",  True,  False, 1.5),
        ("FedKDL Relay (SVD+Coop)",   LORA_INT8, LORA_INT8, "SVD + Coop",     True,  True,  1.5),
    ]
    for name, p_auv, p_relay, relay_op, has_svd, has_coop, fm in rows_53:
        tau, eng, _ = compute_hfl(p_auv, p_relay, fm, has_svd, has_coop)
        tau_s = svd_delay() if has_svd else 0.0
        f.write(f"| {name} | {p_auv:.1f} | {relay_op} | {tau_s:.4f} | {tau:.1f} | {eng:.1f} | {joint_cost(tau, eng):.4f} |\n")

    # ── 5.4 Knowledge Distillation ───────────────────────────────────
    f.write("\n## 5.4 Knowledge Distillation\n\n")
    f.write("Payload truyền không đổi; overhead KD nằm ở phía Gateway (không ảnh hưởng comm energy).\n\n")
    f.write("| Method | Payload (KB) | Comm overhead | tau_round (s) | E_total (J) | Joint Cost |\n")
    f.write("|---|---|---|---|---|---|\n")

    rows_54 = [
        ("No KD",        LORA_INT8, "None",              1.5),
        ("Logit KD",     LORA_INT8, "+ Output logits",   1.5),
        ("Feature KD",   LORA_INT8, "+ Dense features",  1.5),
        ("LoRA-Proj KD", LORA_INT8, "None (proj only)",  1.5),
    ]
    for name, pload, overhead, fm in rows_54:
        tau, eng, _ = compute_hfl(
            pload, pload, fm, has_svd=True, has_coop=True
        )
        f.write(f"| {name} | {pload:.1f} | {overhead} | {tau:.1f} | {eng:.1f} | {joint_cost(tau, eng):.4f} |\n")

    # ── 5.5 Latency-Energy-Accuracy Tradeoff ─────────────────────────
    f.write("\n## 5.5 Latency-Energy-Accuracy Tradeoff\n\n")
    f.write(f"- **tau_comp** in tau_round: YES (local training bottleneck modeled).\n")
    f.write(f"- **Lambda_tau** = {LAMBDA_TAU} (s⁻¹), **Lambda_E** = {LAMBDA_E} (J⁻¹) — scaled to balance contributions.\n\n")
    f.write("| Method | Payload (KB) | tau_round (s) | E_total (J) | Joint Cost | Survival (rounds) |\n")
    f.write("|---|---|---|---|---|---|\n")

    rows_55 = [
        # (name, pload_auv, pload_relay, flop_mult, has_svd, is_hfl)
        ("Flat FL (Full FP32)",          FULL_KB,    None,       3.0, False, False),
        ("HFL (Full FP32, no compress)", FULL_KB,    FULL_KB,    3.0, False, True),
        ("LoRA FL (FP32, flat)",         LORA_32_KB, None,       1.5, False, False),
        ("HFL LoRA (FP32, no SVD)",      LORA_32_KB, LORA_32_KB, 1.5, False, True),
        ("HFL Top-K 5%",                 TOPK5_KB,   TOPK5_KB,   3.0, False, True),
        ("FedKDL (LoRA+INT8+SVD+KD)",   LORA_INT8,  LORA_INT8,  1.5, True,  True),
    ]
    for name, p_auv, p_relay, fm, has_svd, is_hfl in rows_55:
        if is_hfl:
            tau, eng, max_e = compute_hfl(
                p_auv, p_relay, fm, has_svd, has_coop=has_svd
            )
        else:
            tau, eng, max_e = compute_flat_fl(p_auv, fm)
        jc  = joint_cost(tau, eng)
        sur = survival_rounds(max_e)
        f.write(f"| {name} | {p_auv:.1f} | {tau:.1f} | {eng:.1f} | {jc:.4f} | {sur:.1f} |\n")

    # Bổ sung section 5.6: Phân tích chi tiết Năng lượng & Trễ từng chặng (Per-Device Breakdown)
    f.write("\n## 5.6 Per-Device Energy & Latency Breakdown (FedKDL)\n\n")
    f.write("| Chặng | Độ trễ bottleneck (s) | E_tx (J) | E_rx (J) | E_link (J) |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for stage, values in PHYSICS_BREAKDOWN.items():
        f.write(
            f"| {stage} | {values['latency']:.4f} | {values['tx']:.4f} | "
            f"{values['rx']:.4f} | {values['tx'] + values['rx']:.4f} |\n"
        )
    f.write(
        f"| Local computation | {PHYSICS_TAU_COMP:.4f} | N/A | N/A | "
        f"{PHYSICS_E_COMP:.4f} |\n"
    )
    f.write(
        f"| Relay SVD | {PHYSICS_TAU_SVD:.6f} | N/A | N/A | "
        f"{PHYSICS_E_SVD:.6f} |\n"
    )
    f.write(
        f"| **Round total** | **{PHYSICS_TAU_ROUND:.4f}** | N/A | N/A | "
        f"**{PHYSICS_E_ROUND:.4f}** |\n"
    )
    f.write("\n")
    f.write("### Breakdown tính toán theo AUV\n\n")
    f.write("| AUV | Số mẫu | tau_comp (s) | E_comp (J) |\n")
    f.write("|---:|---:|---:|---:|\n")
    for auv_id in sorted(SAMPLE_COUNTS):
        sample_count = SAMPLE_COUNTS[auv_id]
        f.write(
            f"| {auv_id} | {sample_count} | "
            f"{local_comp_delay(1.5, sample_count):.4f} | "
            f"{local_comp_energy(1.5, sample_count):.4f} |\n"
        )
    f.write(
        f"| **Tổng / bottleneck** | **{sum(SAMPLE_COUNTS.values())}** | "
        f"**{PHYSICS_TAU_COMP:.4f}** | **{PHYSICS_E_COMP:.4f}** |\n"
    )
    f.write("\n")
    f.write("### Chi tiết vật lý từng liên kết\n\n")
    f.write(
        "| Chặng | Liên kết | d (m) | R (bps) | SL (dB) | "
        "tau_tx (s) | tau_prop (s) | tau_total (s) | E_tx (J) | E_rx (J) |\n"
    )
    f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in PHYSICS_LINK_ROWS:
        if row['status'] == 'infeasible':
            f.write(
                f"| {row['stage']} | {row['link']} | {row['distance']:.2f} | "
                "N/A | N/A | N/A | N/A | N/A | N/A | N/A "
                "(liên kết không khả thi) |\n"
            )
            continue
        f.write(
            f"| {row['stage']} | {row['link']} | {row['distance']:.2f} | "
            f"{row['rate']:.2f} | {row['source_level']:.2f} | "
            f"{row['tx_delay']:.4f} | {row['prop_delay']:.4f} | "
            f"{row['total_delay']:.4f} | {row['tx_energy']:.4f} | "
            f"{row['rx_energy']:.4f} |\n"
        )
    f.write("\n")
    f.write("Tính toán chi tiết cho AUV 0 và Relay phụ trách (Sử dụng đúng các hàm vật lý gốc).\n\n")
    
    auv_id = 0
    relay_id = topo.hfl_association.get(auv_id, 0)
    d_a2r = dist3(topo.auv_positions[auv_id], topo.relay_positions[relay_id])
    d_r2g = dist3(topo.relay_positions[relay_id], topo.gateway_position)
    
    pload_auv = LORA_INT8
    S_bits_auv = pload_auv * 1024 * 8
    
    # 1. Tính toán tại AUV
    tau_comp_auv = comp_delay_dynamic(SAMPLE_COUNTS[auv_id], LOCAL_EPOCHS, FLOPS_SAMPLE, 1.5, F_CPU, N_CORES, FPC)
    e_comp_auv_val = e_comp(SAMPLE_COUNTS[auv_id], LOCAL_EPOCHS, FLOPS_SAMPLE, EPS_OP, 1.5, F_CPU)
    
    # 2. Truyền AUV -> Relay
    link_a2r = G[('auv', auv_id, 'relay', relay_id)]
    tau_comm_a2r, e_tx_auv, e_rx_relay = link_physics(
        pload_auv, d_a2r, link_a2r
    )
    
    # 3. Tính toán tại Relay (SVD)
    tau_svd_relay = relay_comp_delay(256, 128, 1, F_CPU, N_CORES, FPC)
    e_svd_relay = e_svd(256, 128, EPS_OP, 1, F_CPU)
    
    # 4. Truyền Relay -> Gateway
    S_bits_relay = S_bits_auv
    r2g_key = ('relay', relay_id, 'gateway', 0)
    if r2g_key in G:
        tau_comm_r2g, e_tx_relay, e_rx_gateway = link_physics(
            pload_auv, d_r2g, G[r2g_key]
        )
        r2g_detail = f"Khoảng cách: {d_r2g:.0f}m"
    else:
        tau_comm_r2g = e_tx_relay = e_rx_gateway = 0.0
        r2g_detail = "Liên kết không khả thi; không truyền"
    
    f.write("| Thiết bị | Chặng | Trễ (s) | Năng lượng (J) | Chi tiết |\n")
    f.write("|---|---|---|---|---|\n")
    f.write(f"| **AUV 0** | Huấn luyện cục bộ (LoRA) | {tau_comp_auv:.2f} | {e_comp_auv_val:.2f} | Local training |\n")
    f.write(f"| **AUV 0** | Truyền AUV -> Relay | {tau_comm_a2r:.2f} | {e_tx_auv:.2f} | Khoảng cách: {d_a2r:.0f}m |\n")
    f.write(f"| **Relay {relay_id}** | Nhận từ AUV 0 | - | {e_rx_relay:.2f} | Mạch thu: {P_C_RX}W |\n")
    f.write(f"| **Relay {relay_id}** | Tổng hợp SVD | {tau_svd_relay:.4f} | {e_svd_relay:.6f} | D_out=256, D_in=128 |\n")
    f.write(f"| **Relay {relay_id}** | Truyền Relay -> Gateway | {tau_comm_r2g:.2f} | {e_tx_relay:.2f} | {r2g_detail} |\n")
    f.write(f"| **Gateway** | Nhận từ Relay {relay_id} | - | {e_rx_gateway:.2f} | Mạch thu: {P_C_RX}W |\n")
    f.write("\n")

print("Done -> theoretical_metrics.md")

# ─────────────────────────────────────────────────────────────────────
# 7. Generate Mock Metrics for LaTeX Demo (Grouped by RQ)
# ─────────────────────────────────────────────────────────────────────
import os
import random
import pandas as pd
from tasks.detection_2d.baselines import parse_baseline_config
from utils.env_manager import EnvironmentManager

def get_centralized_metrics():
    csv_path = os.path.join("results", "lora_vs_nolora", "results_yolo12n_lora.csv")
    if not os.path.exists(csv_path):
        return 0.70, 0.65, 0.75, 0.60
    df = pd.read_csv(csv_path)
    return df['metrics/mAP50-95(B)'].max(), df['metrics/mAP50(B)'].max(), df['metrics/precision(B)'].max(), df['metrics/recall(B)'].max()

def calc_flat_physics(baseline):
    # Dù baseline là HFL hay Flat, yêu cầu là tính Flat cho TẤT CẢ AUVs nối thẳng lên Gateway
    cfg = parse_baseline_config(baseline)
    
    if cfg.full_param:
        auv_kb = 11.0 * 1024
    else:
        if cfg.use_int8:
            auv_kb = 300
        elif cfg.topk_grad:
            auv_kb = 100
        else:
            auv_kb = 1200
            
    total_energy = 0.0
    max_latency = 0.0
    total_payload_mb = 0.0
    
    flop_mult = 3.0 if cfg.full_param else 1.5
    
    for s_id in range(topo.N):
        n_samples = SAMPLE_COUNTS[s_id]
        S_bits = auv_kb * 1024 * 8
        link_key = ('auv', s_id, 'gateway', 0)
        if link_key not in G:
            continue
        total_payload_mb += auv_kb / 1024.0
        dist = dist3(topo.auv_positions[s_id], topo.gateway_position)
        t_comm, e_tx_cost, e_rx_cost = link_physics(auv_kb, dist, G[link_key])
        
        e_comp_cost = local_comp_energy(flop_mult, n_samples)
        t_comp = local_comp_delay(flop_mult, n_samples)
        
        total_energy += e_tx_cost + e_rx_cost + e_comp_cost
        auv_latency = t_comp + t_comm
        if auv_latency > max_latency:
            max_latency = auv_latency
            
    return total_payload_mb, total_energy, max_latency

def generate_mock_latex():
    random.seed(42)
    max_map50_95, max_map50, max_prec, max_rec = get_centralized_metrics()
    
    # Định nghĩa mức giảm theo sức mạnh thuật toán (Base: Centralized)
    tiers = {
        "centralized": 0.000,
        "fedkdl_nolora": 0.012,
        "fedkd": 0.015,
        "topk_grad": 0.035,
        "fedkdl": 0.055,
        "fedkdl_selective": 0.075,
        "scaffold": 0.095,
        "fedkdl_nocoop": 0.110,
        "logit_kd": 0.115,
        "fedkdl_proxy_ft": 0.120,
        "fedprox_hfl": 0.124,
        "fedavg_hfl": 0.127,
        "flora": 0.130,
        "fedkdl_nokd": 0.133,
        "fedprox_kdl": 0.145,
        "fedprox": 0.163,
        "fedavg": 0.197,
        "naive_lora": 0.233
    }
    
    rq_groups = {
        "RQ1 (Connection/Stability)": ["fedkdl", "fedavg", "fedprox"],
        "RQ2 (Compression)": ["topk_grad", "fedkdl", "flora", "fedavg_hfl"],
        "RQ3 (Non-IID & Relay)": ["fedkdl_selective", "fedkdl", "fedkdl_nocoop", "scaffold", "flora", "fedavg_hfl"],
        "RQ4 (Gateway KD Ablation)": ["centralized", "fedkdl", "logit_kd", "fedkdl_proxy_ft", "fedkdl_nokd"],
        "Reference & Ablation": ["fedkd", "fedkdl_nolora", "fedprox_kdl", "fedprox_hfl", "naive_lora"]
    }
    
    results = []
    
    for rq_name, baselines in rq_groups.items():
        for baseline in baselines:
            base_drop = tiers.get(baseline, 0.10)
            noise_range = base_drop * 0.5
            drop = base_drop + random.uniform(-noise_range, noise_range)
            
            p_mb, eng_j, lat_s = calc_flat_physics(baseline)
            
            results.append({
                "RQ_Group": rq_name,
                "Baseline": baseline,
                "mAP50-95": max(0.1, round(max_map50_95 - drop, 4)),
                "mAP50": max(0.2, round(max_map50 - drop, 4)),
                "Precision": max(0.2, round(max_prec - drop, 4)),
                "Recall": max(0.2, round(max_rec - drop, 4)),
                "Payload_MB": round(p_mb, 2),
                "Energy_J": round(eng_j, 2),
                "Latency_s": round(lat_s, 2)
            })
            
    df = pd.DataFrame(results)
    out_csv = "latex_demo_metrics_grouped.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Mock] Đã xuất {out_csv} gom nhóm theo RQ.")

generate_mock_latex()
