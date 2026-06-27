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
from pathlib import Path

from config.settings import network_cfg, acoustic_cfg, energy_cfg, fed_cfg
from physics_models.communication import min_source_level, shannon_capacity
from physics_models.energy import e_tx, e_comp, e_rx, e_svd, total_energy_round
from physics_models.latency import comm_delay, comp_delay_dynamic, relay_comp_delay
from utils.env_manager import EnvironmentManager

# ─────────────────────────────────────────────────────────────────────
# 1. Load Topology
# ─────────────────────────────────────────────────────────────────────
TOPO_PATH = "environments/2d/topo/N_30/topo_N30_seed1107.pkl"
DATA_PATH = "environments/2d/data/URPC/N_30/data_N30_URPC_a1p0_seed1107.pkl"
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

# Model structure follows the current adaptive-rank LoRA payload configuration.
NUM_PARAMS_FULL = fed_cfg.MODEL_TOTAL_PARAMS_2D
NUM_PARAMS_LORA = fed_cfg.LORA_TRAINABLE_PARAMS_2D

SAMPLE_COUNTS = {
    auv_id: len(indices)
    for auv_id, indices in data_partition.auv_data_indices.items()
}
MAX_SAMPLES = max(SAMPLE_COUNTS.values())


def _reference_image_sizes_kb():
    """Return actual raw-image bytes owned by each AUV in the N=30 partition."""
    dataset_root = Path("datasets/URPC2020/URPC2020")
    image_paths = sorted(
        path
        for suffix in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        for path in (dataset_root / "train" / "images").glob(suffix)
    )
    if not image_paths:
        raise FileNotFoundError(
            "No URPC training images found for centralized payload accounting."
        )
    image_sizes = [path.stat().st_size / 1024.0 for path in image_paths]
    return {
        auv_id: sum(image_sizes[index] for index in indices)
        for auv_id, indices in data_partition.auv_data_indices.items()
    }


RAW_IMAGE_KB_BY_AUV = _reference_image_sizes_kb()

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
        if i not in topo.hfl_association:
            continue
        m = topo.hfl_association[i]
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
        if i not in topo.hfl_association:
            link_rows.append({
                'stage': 'A2R',
                'link': f'AUV{i}->?',
                'distance': 0.0,
                'status': 'infeasible',
            })
            print(f"{'A2R':7} {f'AUV{i}->?':15} {'N/A':>8} {'N/A':>10} "
                  f"{'N/A':>9} {'SKIPPED: no relay assigned':>41}")
            continue
        m = topo.hfl_association[i]
        key = ('auv', i, 'relay', m)
        d = dist3(topo.auv_positions[i], topo.relay_positions[m])
        if key not in G:
            link_rows.append({
                'stage': 'A2R',
                'link': f'AUV{i}->R{m}',
                'distance': d,
                'status': 'infeasible',
            })
            print(f"{'A2R':7} {f'AUV{i}->R{m}':15} {d:8.2f} {'N/A':>10} "
                  f"{'N/A':>9} {'SKIPPED: infeasible link':>41}")
            continue
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
LORA_INT8  = fed_cfg.LORA_INT8_PAYLOAD_BYTES_2D / 1024.0
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
        ("No KD",              LORA_INT8, "None",              1.5),
        ("Logit KD",           LORA_INT8, "+ Logit (KL)",      1.5),
        ("Logit + Box KD",     LORA_INT8, "+ Logit + Box",     1.5),
        ("Logit + Proj KD",    LORA_INT8, "+ Logit + Proj",    1.5),
        ("FedKDL (full KD)",   LORA_INT8, "+ Logit+Box+Proj",  1.5),
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
from tasks.detection_2d.baselines import (
    STANDARD_BASELINES,
    parse_baseline_config,
)
from utils.env_manager import EnvironmentManager

def get_centralized_metrics():
    csv_path = os.path.join("results", "lora_vs_nolora", "results_yolo12n_lora.csv")
    if not os.path.exists(csv_path):
        return 0.70, 0.65, 0.75, 0.60
    df = pd.read_csv(csv_path)
    return df['metrics/mAP50-95(B)'].max(), df['metrics/mAP50(B)'].max(), df['metrics/precision(B)'].max(), df['metrics/recall(B)'].max()

def forced_link_physics(payload_kb_val: float, d: float):
    """Physical cost of a transmission attempt, including infeasible flat links."""
    if d <= 0:
        return 0.0, 0.0, 0.0
    sl_required = min_source_level(
        d, F_KHZ, B_hz, SNR_dB, IL, SPREADING, WIND, SHIPPING
    )
    # An infeasible node still transmits, but a real modem cannot exceed SL_MAX.
    sl_used = min(sl_required, acoustic_cfg.SL_MAX)
    s_bits = payload_kb_val * 1024 * 8
    latency = comm_delay(s_bits, R_BPS, d, C_S)
    tx = e_tx(s_bits, R_BPS, sl_used, ETA_EA, P_C_TX, 1025.0, C_S)
    rx = e_rx(s_bits, R_BPS, P_C_RX)
    return latency, tx, rx

def calc_physics_for_baseline(baseline):
    cfg = parse_baseline_config(baseline)
    
    is_centralized = (baseline == 'centralized')
    if is_centralized:
        auv_kb = 0.0
        relay_kb = 0.0
    elif cfg.topk_grad:
        auv_kb = TOPK5_KB
        relay_kb = TOPK5_KB
    elif cfg.full_param:
        auv_kb = FULL_KB
        relay_kb = FULL_KB
    elif cfg.use_int8:
        auv_kb = LORA_INT8
        relay_kb = LORA_INT8
    else:
        auv_kb = LORA_32_KB
        relay_kb = LORA_32_KB
        
    flop_mult = 3.0 if cfg.full_param else 1.5
    
    total_energy = 0.0
    max_latency = 0.0
    total_payload_mb = 0.0
    
    if not cfg.hfl or is_centralized:
        # Flat FL hoặc Centralized: AUV -> Gateway (ép truyền dù nhiễu)
        for i in range(topo.N):
            n_samples = SAMPLE_COUNTS[i]
            if is_centralized:
                current_auv_kb = RAW_IMAGE_KB_BY_AUV[i]
            else:
                current_auv_kb = auv_kb
                
            total_payload_mb += current_auv_kb / 1024.0
            dist = dist3(topo.auv_positions[i], topo.gateway_position)
            
            t_comm, e_tx_cost, e_rx_cost = forced_link_physics(current_auv_kb, dist)
            e_comp_cost = local_comp_energy(flop_mult, n_samples)
            t_comp = local_comp_delay(flop_mult, n_samples)
            
            total_energy += e_tx_cost + e_rx_cost + e_comp_cost
            auv_latency = t_comp + t_comm
            if auv_latency > max_latency:
                max_latency = auv_latency
                
    else:
        # HFL: AUV -> Relay -> Relay -> Gateway
        relay_recv_energy = {m: 0.0 for m in range(topo.M)}
        relay_max_tau_comm = {m: 0.0 for m in range(topo.M)}
        
        # 1. AUV -> Relay
        for i in range(topo.N):
            if i not in topo.hfl_association:
                continue
            m = topo.hfl_association[i]
            n_samples = SAMPLE_COUNTS[i]
            
            total_payload_mb += auv_kb / 1024.0
            dist = dist3(topo.auv_positions[i], topo.relay_positions[m])
            
            t_comm, e_tx_cost, e_rx_cost = forced_link_physics(auv_kb, dist)
            e_comp_cost = local_comp_energy(flop_mult, n_samples)
            t_comp = local_comp_delay(flop_mult, n_samples)
            
            total_energy += e_tx_cost + e_rx_cost + e_comp_cost
            relay_recv_energy[m] += e_rx_cost
            
            if t_comp + t_comm > relay_max_tau_comm[m]:
                relay_max_tau_comm[m] = t_comp + t_comm
                
        # 2. Relay SVD computation
        has_svd = getattr(cfg, 'lora_aggregation', '') == 'svd' and not cfg.full_param
        svd_calls = 2 if cfg.coop_rule != 'nocoop' else 1
        t_svd = relay_comp_delay(n_svd_calls=svd_calls, f_cpu=F_CPU, n_cores=N_CORES, flops_per_cycle=FPC) if has_svd else 0.0
        e_svd_val = e_svd(256, 128, EPS_OP, svd_calls, F_CPU) if has_svd else 0.0
        total_energy += topo.M * e_svd_val
        
        # 3. Relay -> Relay
        relay_r2r_latency = {m: 0.0 for m in range(topo.M)}
        if cfg.coop_rule != 'nocoop':
            from federated_core.hfl_rules import should_cooperate, compute_mean_cluster_size
            import numpy as np
            
            cluster_sizes = {m_id: 0 for m_id in range(topo.M)}
            for auv_id, m_id in getattr(topo, 'hfl_association', {}).items():
                cluster_sizes[m_id] += 1
            mean_c = compute_mean_cluster_size(cluster_sizes)
            
            all_r2r_dists = []
            for m1 in range(topo.M):
                for m2 in range(m1 + 1, topo.M):
                    all_r2r_dists.append(dist3(topo.relay_positions[m1], topo.relay_positions[m2]))
            q1 = float(np.percentile(all_r2r_dists, 25)) if all_r2r_dists else float('inf')

            for m in range(topo.M):
                if cfg.coop_rule == 'selective':
                    if not should_cooperate(cluster_sizes[m], mean_c):
                        continue
                        
                my_size = cluster_sizes[m]
                candidates = []
                for other_id in range(topo.M):
                    if other_id == m:
                        continue
                    if cfg.coop_rule == 'selective' and cluster_sizes[other_id] <= my_size:
                        continue
                        
                    d = dist3(topo.relay_positions[m], topo.relay_positions[other_id])
                    if cfg.coop_rule == 'selective' and d > q1:
                        continue
                        
                    candidates.append((other_id, d))
                    
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    partner_id, d_partner = candidates[0]
                    t_comm, e_tx_cost, e_rx_cost = forced_link_physics(relay_kb, d_partner)
                    
                    relay_r2r_latency[m] = t_comm
                    total_energy += e_tx_cost + e_rx_cost
                
        # 4. Relay -> Gateway
        for m in range(topo.M):
            dist = dist3(topo.relay_positions[m], topo.gateway_position)
            t_comm, e_tx_cost, e_rx_cost = forced_link_physics(relay_kb, dist)
            total_energy += e_tx_cost + e_rx_cost
            
            total_path_latency = relay_max_tau_comm[m] + t_svd + relay_r2r_latency[m] + t_comm
            if total_path_latency > max_latency:
                max_latency = total_path_latency

    return total_payload_mb, total_energy, max_latency


# ---------------------------------------------------------------------------
# 8. Three-seed physics-only scalability evaluation
# ---------------------------------------------------------------------------

SCALABILITY_N = (30, 40, 50, 60, 70, 80, 90, 100)
SCALABILITY_SEEDS = (220, 1252, 2419)
SCALABILITY_M_RELAYS = 8
SCALABILITY_BASELINES = STANDARD_BASELINES


def _scaled_sample_counts(n_auvs: int):
    """Reuse the empirical N=30 workload distribution without extrapolating mAP."""
    reference = [SAMPLE_COUNTS[i] for i in sorted(SAMPLE_COUNTS)]
    if not reference:
        raise ValueError("The reference N=30 data partition has no AUV samples.")
    return {i: reference[i % len(reference)] for i in range(n_auvs)}


def _scaled_raw_image_kb(n_auvs: int):
    """Reuse measured per-AUV raw-image ownership for physics-only scaling."""
    reference = [
        RAW_IMAGE_KB_BY_AUV[i] for i in sorted(RAW_IMAGE_KB_BY_AUV)
    ]
    return {i: reference[i % len(reference)] for i in range(n_auvs)}


def _baseline_physics_profile(baseline: str):
    cfg = parse_baseline_config(baseline)
    if cfg.topk_grad:
        auv_kb = relay_kb = TOPK5_KB
    elif cfg.full_param:
        auv_kb = relay_kb = FULL_KB
    elif cfg.use_int8:
        auv_kb = relay_kb = LORA_INT8
    else:
        auv_kb = relay_kb = LORA_32_KB
    # SCAFFOLD uploads the model delta together with a same-shaped delta_c.
    # The relay/global state likewise carries both tensors.
    if cfg.scaffold:
        auv_kb *= 2.0
        relay_kb *= 2.0
    return cfg, auv_kb, relay_kb, 3.0 if cfg.full_param else 1.5


def _nearest_feasible_partner(snapshot, graph, relay_id: int):
    candidates = []
    for other_id in range(snapshot.M):
        if other_id == relay_id:
            continue
        key = ("relay", relay_id, "relay", other_id)
        reverse_key = ("relay", other_id, "relay", relay_id)
        if key in graph:
            candidates.append((graph[key].distance, other_id, key))
        elif reverse_key in graph:
            candidates.append((graph[reverse_key].distance, other_id, reverse_key))
    return min(candidates) if candidates else None


def _cooperation_partners(snapshot, graph, coop_rule: str):
    """Apply the same nearest/selective R2R rule as the training simulator."""
    if coop_rule == "nocoop":
        return {}

    cluster_sizes = {relay_id: 0 for relay_id in range(snapshot.M)}
    for relay_id in snapshot.hfl_association.values():
        cluster_sizes[relay_id] += 1
    nonempty_sizes = [size for size in cluster_sizes.values() if size > 0]
    mean_cluster_size = (
        float(sum(nonempty_sizes) / len(nonempty_sizes))
        if nonempty_sizes else 1.0
    )
    feasible_r2r_distances = [
        link.distance
        for (type_u, _, type_v, _), link in graph.items()
        if type_u == "relay" and type_v == "relay"
    ]
    q1_distance = (
        float(np.percentile(feasible_r2r_distances, 25))
        if coop_rule == "selective" and feasible_r2r_distances
        else None
    )

    partners = {}
    for relay_id in range(snapshot.M):
        if coop_rule == "selective":
            threshold = max(
                2,
                fed_cfg.COOP_THRESHOLD_MULTIPLIER * mean_cluster_size,
            )
            if cluster_sizes[relay_id] > threshold:
                continue

        candidates = []
        for other_id in range(snapshot.M):
            if other_id == relay_id:
                continue
            if (
                coop_rule == "selective"
                and cluster_sizes[other_id] <= cluster_sizes[relay_id]
            ):
                continue
            key = ("relay", relay_id, "relay", other_id)
            reverse_key = ("relay", other_id, "relay", relay_id)
            graph_key = key if key in graph else reverse_key
            if graph_key not in graph:
                continue
            distance = graph[graph_key].distance
            if q1_distance is not None and distance > q1_distance:
                continue
            candidates.append((distance, other_id))

        if not candidates:
            continue
        partners[relay_id] = min(candidates)[1]
    return partners


def _compute_scalability_case(snapshot, graph, baseline: str):
    """Compute one-round physical metrics for one topology and baseline."""
    cfg, auv_kb, relay_kb, flop_mult = _baseline_physics_profile(baseline)
    sample_counts = _scaled_sample_counts(snapshot.N)
    raw_image_kb = _scaled_raw_image_kb(snapshot.N)
    is_centralized = baseline == "centralized"
    is_hfl = bool(cfg.hfl)
    feasible_association = (
        snapshot.hfl_association if is_hfl else snapshot.flat_association
    )
    attempted_auvs = snapshot.N
    delivered_auvs = len(feasible_association)
    participants = delivered_auvs

    e_a2r_tx = e_a2r_rx = e_r2r_tx = e_r2r_rx = 0.0
    e_a2g_tx = e_a2g_rx = 0.0
    e_r2g_tx = e_r2g_rx = e_comp_total = e_svd_total = 0.0
    tau_a2g = tau_a2r = tau_r2r = tau_r2g = tau_comp = tau_svd = 0.0
    payload_auv_total_kb = (
        sum(raw_image_kb.values())
        if is_centralized else attempted_auvs * auv_kb
    )
    payload_r2r_total_kb = 0.0
    payload_r2g_total_kb = 0.0

    if not is_hfl:
        # Flat methods attempt direct AUV->gateway transmission for every AUV.
        # Feasibility is reported separately as delivery rate.
        for auv_id in range(snapshot.N):
            key = ("auv", auv_id, "gateway", 0)
            payload_kb = raw_image_kb[auv_id] if is_centralized else auv_kb
            distance = dist3(
                snapshot.auv_positions[auv_id],
                snapshot.gateway_position,
            )
            if key in graph:
                delay, tx, rx = link_physics(
                    payload_kb, distance, graph[key]
                )
            else:
                delay, tx, rx = forced_link_physics(payload_kb, distance)

            if is_centralized:
                comp_t = comp_e = 0.0
            else:
                comp_t = local_comp_delay(
                    flop_mult, sample_counts[auv_id]
                )
                comp_e = local_comp_energy(
                    flop_mult, sample_counts[auv_id]
                )
            tau_a2g = max(tau_a2g, delay)
            tau_comp = max(tau_comp, comp_t)
            e_a2g_tx += tx
            e_a2g_rx += rx
            e_comp_total += comp_e

        if is_centralized:
            total_samples = sum(sample_counts.values())
            gateway_cpu = F_CPU * 5.0
            tau_comp = comp_delay_dynamic(
                total_samples,
                1,
                FLOPS_SAMPLE,
                fed_cfg.FLOP_MULTIPLIER["2D"],
                gateway_cpu,
                N_CORES,
                FPC,
            )
            e_comp_total = e_comp(
                total_samples,
                1,
                FLOPS_SAMPLE,
                EPS_OP,
                fed_cfg.FLOP_MULTIPLIER["2D"],
                gateway_cpu,
            )
        tau_round = tau_comp + tau_a2g
        cluster_sizes = []
        coop_links = 0
    else:
        per_relay_a2r = {m: 0.0 for m in range(snapshot.M)}
        per_relay_r2r = {m: 0.0 for m in range(snapshot.M)}
        per_relay_r2g = {m: 0.0 for m in range(snapshot.M)}
        cluster_sizes = [0 for _ in range(snapshot.M)]

        for auv_id, relay_id in feasible_association.items():
            cluster_sizes[relay_id] += 1
            key = ("auv", auv_id, "relay", relay_id)
            if key not in graph:
                continue
            delay, tx, rx = link_physics(
                auv_kb, graph[key].distance, graph[key]
            )
            comp_t = local_comp_delay(flop_mult, sample_counts[auv_id])
            comp_e = local_comp_energy(flop_mult, sample_counts[auv_id])
            per_relay_a2r[relay_id] = max(
                per_relay_a2r[relay_id], delay
            )
            tau_comp = max(tau_comp, comp_t)
            e_a2r_tx += tx
            e_a2r_rx += rx
            e_comp_total += comp_e

        has_svd = (
            getattr(cfg, "lora_aggregation", "") == "svd"
            and not cfg.full_param
        )
        cooperation_partners = _cooperation_partners(
            snapshot, graph, cfg.coop_rule
        )
        has_coop = bool(cooperation_partners)
        svd_calls = 2 if has_coop else 1
        if has_svd:
            tau_svd = relay_comp_delay(
                n_svd_calls=svd_calls,
                f_cpu=F_CPU,
                n_cores=N_CORES,
                flops_per_cycle=FPC,
            )
            e_svd_total = snapshot.M * e_svd(
                256, 128, EPS_OP, svd_calls, F_CPU
            )

        coop_links = 0
        if has_coop:
            for relay_id, partner_id in cooperation_partners.items():
                key = ("relay", partner_id, "relay", relay_id)
                if key not in graph:
                    key = ("relay", relay_id, "relay", partner_id)
                if key not in graph:
                    continue
                distance = graph[key].distance
                delay, tx, rx = link_physics(
                    relay_kb, distance, graph[key]
                )
                per_relay_r2r[relay_id] = delay
                e_r2r_tx += tx
                e_r2r_rx += rx
                payload_r2r_total_kb += relay_kb
                coop_links += 1

        for relay_id in range(snapshot.M):
            key = ("relay", relay_id, "gateway", 0)
            if key not in graph:
                continue
            delay, tx, rx = link_physics(
                relay_kb, graph[key].distance, graph[key]
            )
            per_relay_r2g[relay_id] = delay
            e_r2g_tx += tx
            e_r2g_rx += rx
            payload_r2g_total_kb += relay_kb

        tau_a2r = max(per_relay_a2r.values(), default=0.0)
        tau_r2r = max(per_relay_r2r.values(), default=0.0)
        tau_r2g = max(per_relay_r2g.values(), default=0.0)
        tau_round = max(
            (
                tau_comp
                + per_relay_a2r[m]
                + tau_svd
                + per_relay_r2r[m]
                + per_relay_r2g[m]
            )
            for m in range(snapshot.M)
        )

    e_comm = (
        e_a2g_tx + e_a2g_rx + e_a2r_tx + e_a2r_rx
        + e_r2r_tx + e_r2r_rx
        + e_r2g_tx + e_r2g_rx
    )
    e_total = e_comm + e_comp_total + e_svd_total
    payload_total_kb = (
        payload_auv_total_kb + payload_r2r_total_kb
        + payload_r2g_total_kb
    )

    return {
        "baseline": baseline,
        "topology": "hfl" if is_hfl else "flat",
        "N_AUV": snapshot.N,
        "M_Relay": snapshot.M,
        "seed": snapshot.seed,
        "attempted_auvs": attempted_auvs,
        "delivered_auvs": delivered_auvs,
        "participants": participants,
        "isolated_auvs": snapshot.N - delivered_auvs,
        "delivery_rate": delivered_auvs / snapshot.N,
        "participation_rate": delivered_auvs / snapshot.N,
        "communication_frequency": (
            "once" if is_centralized else "per_round"
        ),
        "mean_cluster_size": (
            float(np.mean(cluster_sizes)) if cluster_sizes else 0.0
        ),
        "max_cluster_size": max(cluster_sizes, default=0),
        "std_cluster_size": (
            float(np.std(cluster_sizes, ddof=0)) if cluster_sizes else 0.0
        ),
        "coop_links": coop_links,
        "payload_per_auv_kb": (
            float(np.mean(list(raw_image_kb.values())))
            if is_centralized else auv_kb
        ),
        "payload_total_kb": payload_total_kb,
        "tau_a2g_s": tau_a2g,
        "tau_a2r_s": tau_a2r,
        "tau_r2r_s": tau_r2r,
        "tau_r2g_s": tau_r2g,
        "tau_comp_s": tau_comp,
        "tau_svd_s": tau_svd,
        "tau_round_s": tau_round,
        "initial_collection_latency_s": (
            tau_a2g if is_centralized else 0.0
        ),
        "gateway_training_latency_s": (
            tau_comp if is_centralized else 0.0
        ),
        "e_a2g_tx_j": e_a2g_tx,
        "e_a2g_rx_j": e_a2g_rx,
        "e_a2r_tx_j": e_a2r_tx,
        "e_a2r_rx_j": e_a2r_rx,
        "e_r2r_tx_j": e_r2r_tx,
        "e_r2r_rx_j": e_r2r_rx,
        "e_r2g_tx_j": e_r2g_tx,
        "e_r2g_rx_j": e_r2g_rx,
        "e_comm_j": e_comm,
        "e_comp_j": e_comp_total,
        "e_svd_j": e_svd_total,
        "e_total_j": e_total,
        "initial_collection_energy_j": (
            e_a2g_tx + e_a2g_rx if is_centralized else 0.0
        ),
        "gateway_training_energy_j": (
            e_comp_total if is_centralized else 0.0
        ),
        # Centralized is an offline full-data upper bound. Its one-time raw
        # data collection cost is not comparable with a recurring FL round.
        "joint_cost": (
            np.nan if is_centralized else joint_cost(tau_round, e_total)
        ),
    }


def generate_scalability_metrics():
    """Generate physics-only scalability results; no mAP is extrapolated."""
    rows = []
    for n_auvs in SCALABILITY_N:
        for seed in SCALABILITY_SEEDS:
            snapshots = {}
            graphs = {}
            for view in ("flat", "hfl"):
                path = EnvironmentManager.topo_path(
                    "2d", n_auvs, seed, view
                )
                if not path.exists():
                    raise FileNotFoundError(
                        f"Missing scalability topology: {path}"
                    )
                snapshot = EnvironmentManager.load_topology(path)
                if snapshot.M != SCALABILITY_M_RELAYS:
                    raise ValueError(
                        f"{path} has M={snapshot.M}; "
                        f"expected M={SCALABILITY_M_RELAYS}"
                    )
                snapshots[view] = snapshot
                graphs[view] = EnvironmentManager.restore_graph(snapshot)

            for baseline in SCALABILITY_BASELINES:
                cfg = parse_baseline_config(baseline)
                view = "hfl" if cfg.hfl else "flat"
                rows.append(
                    _compute_scalability_case(
                        snapshots[view], graphs[view], baseline
                    )
                )

    raw = pd.DataFrame(rows)
    metric_cols = [
        column for column in raw.select_dtypes(include=[np.number]).columns
        if column not in {"N_AUV", "M_Relay", "seed"}
    ]
    summary = (
        raw.groupby(["baseline", "topology", "N_AUV", "M_Relay"])
        [metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]

    # Round all float columns to 3 decimal places
    for col in summary.columns:
        if summary[col].dtype == np.float64:
            summary[col] = summary[col].round(3)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "scalability_physics_raw.csv")
    summary_path = os.path.join(
        output_dir, "scalability_physics_summary.csv"
    )
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    with open("theoretical_metrics.md", "a", encoding="utf-8") as report:
        report.write("\n## Physics-only Scalability Evaluation\n\n")
        report.write(
            "Learning metrics are measured only for the trained N=30 run. "
            "The following scalability results contain no extrapolated mAP.\n\n"
        )
        report.write(
            f"- N = {list(SCALABILITY_N)}\n"
            f"- M = {SCALABILITY_M_RELAYS}\n"
            f"- Seeds = {list(SCALABILITY_SEEDS)}\n"
            "- Reported statistics: mean and sample standard deviation.\n\n"
        )
        report.write(
            "| Baseline | N | Delivery rate | Round latency (s) | "
            "Total energy (J) | Joint cost |\n"
        )
        report.write("|---|---:|---:|---:|---:|---:|\n")
        for _, row in summary.iterrows():
            if pd.isna(row["joint_cost_mean"]):
                joint_cost_text = "N/A (offline upper bound)"
            else:
                joint_cost_text = (
                    f"{row['joint_cost_mean']:.4f} +/- "
                    f"{row['joint_cost_std']:.4f}"
                )
            report.write(
                f"| {row['baseline']} | {int(row['N_AUV'])} | "
                f"{row['delivery_rate_mean']:.4f} +/- "
                f"{row['delivery_rate_std']:.4f} | "
                f"{row['tau_round_s_mean']:.2f} +/- "
                f"{row['tau_round_s_std']:.2f} | "
                f"{row['e_total_j_mean']:.2f} +/- "
                f"{row['e_total_j_std']:.2f} | "
                f"{joint_cost_text} |\n"
            )

    print(f"Done -> {raw_path}")
    print(f"Done -> {summary_path}")
    return raw, summary


if __name__ == "__main__":
    generate_scalability_metrics()