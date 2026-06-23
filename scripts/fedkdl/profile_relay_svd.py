"""
profile_relay_svd.py
Standalone SVD runtime profiler for relay aggregation.

Measures ACTUAL wall-clock time of the LoRA SVD aggregation pipeline
with proper torch.cuda.synchronize() barriers. Output: mean ± std ms
over N_ITER iterations.

This script does NOT modify aggregator.py. It imports and calls
svd_lora_aggregate() directly.

Usage:
    python scripts/fedkdl/profile_relay_svd.py [--device cuda] [--n-iter 100] [--n-clients 5]
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np

# ── Ensure project root is importable ──
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def make_fake_client_states(
    reference_sd: Dict[str, torch.Tensor],
    n_clients: int,
    noise_scale: float = 0.01,
    device: str = "cpu",
) -> List[Dict[str, torch.Tensor]]:
    """Create n_clients perturbed copies of the reference state dict."""
    clients = []
    for i in range(n_clients):
        sd = {}
        for k, v in reference_sd.items():
            v_dev = v.to(device).float()
            noise = torch.randn_like(v_dev) * noise_scale * v_dev.abs().mean().clamp(min=1e-7)
            sd[k] = v_dev + noise
        clients.append(sd)
    return clients


def profile_svd_aggregate(
    client_sds: List[Dict[str, torch.Tensor]],
    weights: List[float],
    device: str,
    n_iter: int = 100,
    warmup: int = 5,
):
    """
    Profile svd_lora_aggregate with proper GPU synchronization.
    
    Returns dict with timing results in milliseconds.
    """
    from federated_core.aggregator import svd_lora_aggregate
    
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    # Count LoRA layers for reporting
    all_keys = set().union(*(sd.keys() for sd in client_sds))
    lora_B_keys = sorted(k for k in all_keys if 'lora_B' in k)
    non_lora_keys = sorted(k for k in all_keys if 'lora_A' not in k and 'lora_B' not in k)
    
    print(f"\n[SVD Profiler] Configuration:")
    print(f"  Device:       {device} (CUDA sync: {use_cuda})")
    print(f"  Clients:      {len(client_sds)}")
    print(f"  LoRA layers:  {len(lora_B_keys)} pairs")
    print(f"  Non-LoRA keys: {len(non_lora_keys)}")
    print(f"  Iterations:   {n_iter} (+ {warmup} warmup)")
    
    # ── Warmup ──
    print(f"\n  Warming up ({warmup} iterations)...", end=" ", flush=True)
    for _ in range(warmup):
        if use_cuda:
            torch.cuda.synchronize()
        _ = svd_lora_aggregate(client_sds, weights)
        if use_cuda:
            torch.cuda.synchronize()
    print("done.")
    
    # ── Measure peak memory ──
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    
    # ── Timed iterations ──
    total_times_ms = []
    
    for i in range(n_iter):
        if use_cuda:
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        _ = svd_lora_aggregate(client_sds, weights)
        
        if use_cuda:
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_times_ms.append(elapsed_ms)
    
    total_arr = np.array(total_times_ms)
    
    peak_mem_mb = 0.0
    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    results = {
        'total_aggregation': {
            'mean_ms': float(np.mean(total_arr)),
            'std_ms': float(np.std(total_arr)),
            'min_ms': float(np.min(total_arr)),
            'max_ms': float(np.max(total_arr)),
            'median_ms': float(np.median(total_arr)),
            'p95_ms': float(np.percentile(total_arr, 95)),
        },
        'peak_memory_mb': peak_mem_mb,
        'n_lora_layers': len(lora_B_keys),
        'n_clients': len(client_sds),
        'n_iter': n_iter,
        'device': device,
    }
    
    # ── Report ──
    print(f"\n{'=' * 65}")
    print(f"  SVD RELAY AGGREGATION PROFILING RESULTS")
    print(f"{'=' * 65}")
    print(f"  Hardware:     {device}")
    if use_cuda:
        print(f"  GPU:          {torch.cuda.get_device_name()}")
    print(f"  LoRA layers:  {len(lora_B_keys)}")
    print(f"  Clients:      {len(client_sds)}")
    print()
    
    # Markdown table
    print(f"| {'Operation':<35} | {'Mean ms':>10} | {'Std ms':>10} | {'P95 ms':>10} |")
    print(f"|{'-' * 37}|{'-' * 12}|{'-' * 12}|{'-' * 12}|")
    t = results['total_aggregation']
    print(f"| {'Total relay aggregation':<35} | {t['mean_ms']:>10.2f} | {t['std_ms']:>10.2f} | {t['p95_ms']:>10.2f} |")
    
    if peak_mem_mb > 0:
        print(f"\n  Peak GPU memory: {peak_mem_mb:.1f} MB")
    
    print()
    
    # ── Now profile the two sub-operations separately ──
    print(f"  Profiling sub-operations (reconstruction vs SVD)...")
    
    recon_times = []
    svd_times = []
    
    for i in range(n_iter):
        # --- Phase 1: Effective-weight reconstruction (B @ A) ---
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # Replicate the reconstruction loop from svd_lora_aggregate
        for b_key in lora_B_keys:
            a_key = b_key.replace('lora_B', 'lora_A')
            W_avg = None
            for sd, w in zip(client_sds, weights):
                if b_key not in sd or a_key not in sd:
                    continue
                B_i = sd[b_key].float().double()
                A_i = sd[a_key].float().double()
                W_i = torch.matmul(B_i, A_i)
                W_avg = w * W_i if W_avg is None else W_avg + w * W_i
        
        if use_cuda:
            torch.cuda.synchronize()
        recon_ms = (time.perf_counter() - t0) * 1000.0
        recon_times.append(recon_ms)
        
        # --- Phase 2: Truncated SVD extraction ---
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        for b_key in lora_B_keys:
            a_key = b_key.replace('lora_B', 'lora_A')
            # Reconstruct W_avg (needed for SVD input)
            W_avg = None
            for sd, w in zip(client_sds, weights):
                if b_key not in sd or a_key not in sd:
                    continue
                B_i = sd[b_key].float().double()
                A_i = sd[a_key].float().double()
                W_i = torch.matmul(B_i, A_i)
                W_avg = w * W_i if W_avg is None else W_avg + w * W_i
            
            if W_avg is not None:
                U, S, Vh = torch.linalg.svd(W_avg, full_matrices=False)
                rank = client_sds[0][b_key].shape[1]
                keep = min(rank, S.numel())
                sqrt_S = torch.sqrt(S[:keep]).float()
                U_r = U[:, :keep].float()
                Vh_r = Vh[:keep, :].float()
        
        if use_cuda:
            torch.cuda.synchronize()
        svd_ms = (time.perf_counter() - t0) * 1000.0
        svd_times.append(svd_ms)
    
    recon_arr = np.array(recon_times)
    svd_arr = np.array(svd_times)
    
    results['reconstruction'] = {
        'mean_ms': float(np.mean(recon_arr)),
        'std_ms': float(np.std(recon_arr)),
    }
    results['truncated_svd'] = {
        'mean_ms': float(np.mean(svd_arr)),
        'std_ms': float(np.std(svd_arr)),
    }
    
    print(f"\n| {'Operation':<35} | {'Mean ms':>10} | {'Std ms':>10} |")
    print(f"|{'-' * 37}|{'-' * 12}|{'-' * 12}|")
    print(f"| {'Effective-weight reconstruction':<35} | {results['reconstruction']['mean_ms']:>10.2f} | {results['reconstruction']['std_ms']:>10.2f} |")
    print(f"| {'Truncated-SVD (recon + SVD)':<35} | {results['truncated_svd']['mean_ms']:>10.2f} | {results['truncated_svd']['std_ms']:>10.2f} |")
    print(f"| {'Total relay aggregation':<35} | {results['total_aggregation']['mean_ms']:>10.2f} | {results['total_aggregation']['std_ms']:>10.2f} |")
    
    return results


def main():
    parser = argparse.ArgumentParser("FedKDL SVD Relay Profiler")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-iter", type=int, default=100, help="Number of timed iterations")
    parser.add_argument("--n-clients", type=int, default=5, help="Number of simulated clients per relay")
    parser.add_argument("--student-ckpt", type=str, default="yolo12n_warmup.pt")
    parser.add_argument("--output", type=str, default="results/svd_profiling.json")
    args = parser.parse_args()
    
    ckpt_path = Path(args.student_ckpt)
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)
    
    from config.settings import fed_cfg
    from tasks.detection_2d.models.yolo_wrapper import StudentModel
    
    print(f"[Loading] {ckpt_path}")
    student = StudentModel(
        str(ckpt_path),
        rank=fed_cfg.LORA_RANK,
        nc=4,
        full_param=False,
        use_lora=True,
    )
    reference_sd = student.trainable_state_dict()
    
    # Only keep LoRA + non-BN keys for profiling (BN goes through FedAvg, not SVD)
    # Actually, svd_lora_aggregate handles both LoRA (SVD) and non-LoRA (FedAvg),
    # so we pass the full state dict.
    print(f"[State dict] {len(reference_sd)} keys")
    
    # Create fake clients
    print(f"[Creating] {args.n_clients} simulated client state dicts on {args.device}")
    client_sds = make_fake_client_states(
        reference_sd, args.n_clients, device=args.device
    )
    weights = [1.0 / args.n_clients] * args.n_clients
    
    # Profile
    results = profile_svd_aggregate(
        client_sds, weights, args.device,
        n_iter=args.n_iter,
    )
    
    # Compare with theoretical model
    from physics_models.latency import relay_comp_delay
    from config.settings import energy_cfg
    
    theoretical_ms = relay_comp_delay(
        n_svd_calls=1,
        f_cpu=energy_cfg.F_CPU,
        n_cores=energy_cfg.N_CORES,
        flops_per_cycle=energy_cfg.FLOPS_PER_CYCLE,
    ) * 1000.0
    
    results['theoretical_ms'] = theoretical_ms
    
    print(f"\n[Comparison with theoretical model]")
    print(f"  Theoretical relay_comp_delay = {theoretical_ms:.4f} ms")
    print(f"  Measured total (mean)        = {results['total_aggregation']['mean_ms']:.2f} ms")
    ratio = results['total_aggregation']['mean_ms'] / theoretical_ms if theoretical_ms > 0 else float('inf')
    print(f"  Ratio (measured / theoretical) = {ratio:.1f}x")
    
    # Export
    import json
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Exported] {output_path}")


if __name__ == "__main__":
    main()
