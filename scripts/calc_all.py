import sys
from pathlib import Path

# Thêm root_dir vào sys.path để import physics_models
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from physics_models.energy import acoustic_power_watts, e_tx, e_comp, e_svd
from physics_models.latency import comm_delay, comp_delay_dynamic, relay_comp_delay

# -----------------
# 1. Các tham số
# -----------------
epsilon_op = 1.0e-28       # Standard CMOS value (yang2021energy)
flops_2d = 2.175e9         
kappa = 1.2                
f_cpu = 2.0e9              

rho_w = 1025.0             
c_s = 1500.0               
eta_ea = 0.2               
P_c_tx = 0.1               
SL_min_dB = 105.0          
R_bps = 50000.0            

n_samples = 16             
local_epochs = 3           

auv_payload_bits = 4_000_000   
relay_model_bits = 16_000_000  

d_a2r = 1000.0             
d_r2r = 2000.0             
d_r2g = 3000.0             

# -----------------
# 2. Gọi các hàm vật lý
# -----------------

# [1] AUV Computation
E_comp_auv = e_comp(
    n_samples=n_samples,
    local_epochs=local_epochs,
    flops_per_sample=flops_2d,
    epsilon_op=epsilon_op,
    flop_multiplier=kappa,
    f_cpu=f_cpu
)
tau_comp_auv = comp_delay_dynamic(
    n_samples=n_samples,
    n_local_epochs=local_epochs,
    flops_per_sample=flops_2d,
    flop_multiplier=kappa,
    f_cpu=f_cpu
)
# (Để in FLOPs AUV ra cho giống log cũ)
phi_auv = n_samples * local_epochs * flops_2d * kappa

# [2] Relay Computation (SVD)
d_out, d_in = 256, 128
E_svd_relay = e_svd(
    d_out=d_out,
    d_in=d_in,
    epsilon_op=epsilon_op,
    n_svd_calls=2,
    f_cpu=f_cpu
)
tau_svd_relay = relay_comp_delay(
    d_out=d_out,
    d_in=d_in,
    n_svd_calls=2,
    f_cpu=f_cpu
)
# FLOPs SVD
phi_svd = 2 * 6 * d_out * d_in * min(d_out, d_in)

# [3] Acoustic
P_ac = acoustic_power_watts(SL_min_dB, rho_w, c_s)

# [4] AUV -> Relay TX
tau_a2r = comm_delay(auv_payload_bits, R_bps, d_a2r, c_s)
E_tx_a2r = e_tx(auv_payload_bits, R_bps, SL_min_dB, eta_ea, P_c_tx, rho_w, c_s)

# [5] Relay -> Gateway TX
tau_r2g = comm_delay(relay_model_bits, R_bps, d_r2g, c_s)
E_tx_r2g = e_tx(relay_model_bits, R_bps, SL_min_dB, eta_ea, P_c_tx, rho_w, c_s)

# [6] Total latency round
tau_round = tau_comp_auv + tau_a2r + tau_svd_relay + tau_r2g

# -----------------
# 3. In kết quả
# -----------------
print("="*50)
print(" PHYSICAL MODEL TEST RESULTS (IMPORTED FROM PHYSICS_MODELS)")
print("="*50)
print(f"[1] AUV Computation (Batch={n_samples}, Epochs={local_epochs}):")
print(f"    - Workload Phi_i: {phi_auv:.2e} FLOPs")
print(f"    - Energy E_comp: {E_comp_auv:.4f} Joules")
print(f"    - Latency tau_comp: {tau_comp_auv:.4f} s")
print(f"\n[2] Relay Computation (SVD):")
print(f"    - Workload Phi_svd: {phi_svd:.2e} FLOPs")
print(f"    - Energy E_svd: {E_svd_relay:.8f} Joules")
print(f"    - Latency tau_svd: {tau_svd_relay:.4f} s")
print(f"\n[3] Acoustic Communication Physics:")
print(f"    - Acoustic Power P_ac: {P_ac:.8f} W")
print(f"\n[4] AUV -> Relay Transmission (Payload: {auv_payload_bits/8e3} KB, Dist: {d_a2r} m):")
print(f"    - Delay (Tx+Prop): {tau_a2r:.2f} s")
print(f"    - Tx Energy E_tx: {E_tx_a2r:.4f} Joules")
print(f"\n[5] Relay -> Gateway Transmission (Model: {relay_model_bits/8e6} MB, Dist: {d_r2g} m):")
print(f"    - Delay (Tx+Prop): {tau_r2g:.2f} s")
print(f"    - Tx Energy E_tx: {E_tx_r2g:.4f} Joules")
print(f"\n[6] Total Round Latency:")
print(f"    - tau_round: {tau_round:.2f} s")
print("="*50)
