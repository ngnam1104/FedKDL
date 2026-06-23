"""
summarize_multiseed.py
Aggregate multi-seed experiment results into mean ± std tables.

Reads JSON log files from results/logs_kdl/ and groups them by
(baseline, alpha) across seeds to compute statistical summaries.

This script is training-independent — it only reads the output JSON logs.

Usage:
    python scripts/fedkdl/summarize_multiseed.py [--log-dir results/logs_kdl]
    python scripts/fedkdl/summarize_multiseed.py --baselines fedkdl topk_grad fedkdl_nocoop logit_kd
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_log_filename(filename: str) -> dict:
    """
    Parse experiment metadata from log filename.
    Expected format: log_N30_URPC_a1p0_fedkdl_seed1104.json
    """
    stem = Path(filename).stem  # log_N30_URPC_a1p0_fedkdl_seed1104
    parts = stem.split('_')
    
    # Find the seed part (last element starting with 'seed')
    seed = None
    seed_idx = None
    for i, p in enumerate(parts):
        if p.startswith('seed'):
            seed = int(p[4:])
            seed_idx = i
            break
    
    if seed is None:
        return None
    
    # Find N
    N = None
    for p in parts:
        if p.startswith('N') and p[1:].isdigit():
            N = int(p[1:])
            break
    
    # Find alpha (a1p0 -> 1.0)
    alpha_str = None
    alpha_val = None
    for p in parts:
        if p.startswith('a') and 'p' in p[1:]:
            alpha_str = p[1:]  # e.g., "1p0"
            alpha_val = float(alpha_str.replace('p', '.'))
            break
    
    # Dataset is between N and alpha
    # Baseline is between alpha and seed
    # This is fragile but matches the actual naming convention
    # log_N30_URPC_a1p0_fedkdl_seed1104
    # parts: ['log', 'N30', 'URPC', 'a1p0', 'fedkdl', 'seed1104']
    # or: log_N30_URPC_a1p0_fedkdl_nocoop_seed1104
    
    # Find indices
    n_idx = next(i for i, p in enumerate(parts) if p.startswith('N') and p[1:].isdigit())
    a_idx = next(i for i, p in enumerate(parts) if p.startswith('a') and 'p' in p[1:])
    
    dataset = '_'.join(parts[n_idx + 1:a_idx])
    baseline = '_'.join(parts[a_idx + 1:seed_idx])
    
    return {
        'N': N,
        'dataset': dataset,
        'alpha_str': alpha_str,
        'alpha': alpha_val,
        'baseline': baseline,
        'seed': seed,
        'filename': filename,
    }


def extract_metrics(log_data: dict) -> dict:
    """Extract key metrics from a single experiment log JSON."""
    metrics = log_data.get('metrics', {})
    metadata = log_data.get('metadata', {})
    
    # mAP series
    map50_95 = metrics.get('mAP50-95', [])
    map50 = metrics.get('mAP50', [])
    prec = metrics.get('Prec', [])
    rec = metrics.get('Rec', [])
    val_loss = metrics.get('val_loss', [])
    train_loss = metrics.get('loss', [])
    
    # Payload and cost
    avg_payload_kb = metrics.get('avg_payload_kb', [])
    e_cumul = metrics.get('e_cumul', [])
    tau_cumul_s = metrics.get('tau_cumul_s', [])
    
    result = {}
    
    # Best metrics
    if map50_95:
        result['best_mAP50_95'] = max(map50_95)
        result['final_mAP50_95'] = map50_95[-1]
    if map50:
        result['best_mAP50'] = max(map50)
        result['final_mAP50'] = map50[-1]
    if prec:
        result['best_Prec'] = max(prec)
    if rec:
        result['best_Rec'] = max(rec)
    
    # Final losses
    if val_loss:
        result['final_val_loss'] = val_loss[-1]
    if train_loss:
        result['final_train_loss'] = train_loss[-1]
    
    # Communication metrics (from last round)
    if avg_payload_kb:
        # Filter out 0s (centralized has 0 after round 1)
        non_zero = [x for x in avg_payload_kb if x > 0]
        result['payload_kb'] = non_zero[-1] if non_zero else 0.0
    
    # Cumulative costs
    if e_cumul:
        result['e_cumul_final'] = e_cumul[-1]
    if tau_cumul_s:
        result['tau_cumul_final'] = tau_cumul_s[-1]
    
    # Joint cost (if available)
    from config.settings import fed_cfg
    if 'e_cumul_final' in result and 'tau_cumul_final' in result:
        result['joint_cost'] = (
            fed_cfg.LAMBDA_TAU * result['tau_cumul_final']
            + fed_cfg.LAMBDA_E * result['e_cumul_final']
        )
    
    result['n_rounds'] = len(map50_95) if map50_95 else 0
    
    return result


def main():
    parser = argparse.ArgumentParser("FedKDL Multi-seed Summary")
    parser.add_argument("--log-dir", type=str, default="results/logs_kdl",
                        help="Directory containing JSON log files")
    parser.add_argument("--baselines", nargs='+', default=None,
                        help="Filter to specific baselines (default: all found)")
    parser.add_argument("--output", type=str, default="results/multiseed_summary.csv",
                        help="Output CSV path")
    parser.add_argument("--output-json", type=str, default="results/multiseed_summary.json",
                        help="Output JSON path")
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[Error] Log directory not found: {log_dir}")
        sys.exit(1)
    
    # Discover all log files
    log_files = sorted(log_dir.glob("log_*.json"))
    print(f"[Found] {len(log_files)} log files in {log_dir}")
    
    # Parse and group
    groups = defaultdict(list)  # (baseline, alpha_str) -> [(seed, metrics)]
    
    for log_file in log_files:
        meta = parse_log_filename(log_file.name)
        if meta is None:
            print(f"  [Skip] Cannot parse: {log_file.name}")
            continue
        
        if args.baselines and meta['baseline'] not in args.baselines:
            continue
        
        # Load JSON
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [Skip] Cannot read {log_file.name}: {e}")
            continue
        
        metrics = extract_metrics(log_data)
        if metrics.get('n_rounds', 0) == 0:
            print(f"  [Skip] Empty metrics: {log_file.name}")
            continue
        
        key = (meta['baseline'], meta['alpha_str'])
        groups[key].append({
            'seed': meta['seed'],
            'N': meta['N'],
            'dataset': meta['dataset'],
            **metrics,
        })
    
    if not groups:
        print("[Error] No valid log files found matching criteria.")
        sys.exit(1)
    
    # Compute summary
    metric_names = [
        'best_mAP50_95', 'best_mAP50', 'final_mAP50_95', 'final_mAP50',
        'best_Prec', 'best_Rec', 'final_val_loss', 'final_train_loss',
        'payload_kb', 'e_cumul_final', 'tau_cumul_final', 'joint_cost',
    ]
    
    summary_rows = []
    json_summary = {}
    
    print(f"\n{'=' * 90}")
    print(f"  MULTI-SEED SUMMARY")
    print(f"{'=' * 90}")
    
    for (baseline, alpha_str), entries in sorted(groups.items()):
        n_seeds = len(entries)
        seeds = sorted(e['seed'] for e in entries)
        
        row = {
            'baseline': baseline,
            'alpha': alpha_str,
            'n_seeds': n_seeds,
            'seeds': str(seeds),
        }
        
        print(f"\n  {baseline} (alpha={alpha_str}, seeds={seeds})")
        
        for metric_name in metric_names:
            values = [e[metric_name] for e in entries if metric_name in e]
            if not values:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            
            row[f'{metric_name}_mean'] = round(float(mean_val), 4)
            row[f'{metric_name}_std'] = round(float(std_val), 4)
            
            if 'mAP' in metric_name or 'Prec' in metric_name or 'Rec' in metric_name:
                print(f"    {metric_name:25s}: {mean_val:.4f} ± {std_val:.4f}")
            elif 'loss' in metric_name:
                print(f"    {metric_name:25s}: {mean_val:.4f} ± {std_val:.4f}")
            elif 'payload' in metric_name:
                print(f"    {metric_name:25s}: {mean_val:.1f} ± {std_val:.1f} KB")
            else:
                print(f"    {metric_name:25s}: {mean_val:.2f} ± {std_val:.2f}")
        
        summary_rows.append(row)
        json_summary[f"{baseline}_a{alpha_str}"] = row
    
    # Export CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if summary_rows:
        # Get all columns
        all_cols = []
        for row in summary_rows:
            for k in row.keys():
                if k not in all_cols:
                    all_cols.append(k)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(','.join(all_cols) + '\n')
            for row in summary_rows:
                values = [str(row.get(c, '')) for c in all_cols]
                f.write(','.join(values) + '\n')
        
        print(f"\n[Exported CSV] {output_path}")
    
    # Export JSON
    json_path = Path(args.output_json)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    print(f"[Exported JSON] {json_path}")
    
    # Print LaTeX-ready table
    print(f"\n{'=' * 90}")
    print(f"  LATEX-READY TABLE (best_mAP50-95)")
    print(f"{'=' * 90}")
    print()
    print(f"| {'Baseline':<25} | {'Seeds':>6} | {'mAP@50-95':>20} | {'mAP@50':>20} |")
    print(f"|{'-' * 27}|{'-' * 8}|{'-' * 22}|{'-' * 22}|")
    
    for row in summary_rows:
        m1 = f"{row.get('best_mAP50_95_mean', 0):.4f} ± {row.get('best_mAP50_95_std', 0):.4f}"
        m2 = f"{row.get('best_mAP50_mean', 0):.4f} ± {row.get('best_mAP50_std', 0):.4f}"
        print(f"| {row['baseline']:<25} | {row['n_seeds']:>6} | {m1:>20} | {m2:>20} |")
    
    print()


if __name__ == "__main__":
    main()
