import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import torch
from federated_core.workers import BaseGateway
from detection_2d.simulator import RelayNode2D

def parse_baseline_config(baseline: str) -> dict:
    cfg_map = {
        'fedavg':           (True,  False, False, False, False),
        'fedprox':          (True,  False, False, False, False),
        'fedavg_hfl':       (True,  False, False, False, False),
        'fedprox_hfl':      (True,  False, False, False, False),
        'flora':            (False, True,  False, False, False),
        'scaffold':         (True,  False, False, False, False),
        'fedkdl':           (False, True,  True,  True,  False),
        'fedkdl_nocoop':    (False, True,  True,  True,  False),
        'logit_kd':         (False, True,  True,  True,  False),
        'fedprox_kdl':      (False, True,  True,  True,  False),
        'fedkd':            (True,  False, False, True,  False),
        'topk_grad':        (True,  False, False, False, False),
        'centralized':      (False, True,  False, False, False),
        'fedkdl_nokd':      (False, True,  True,  False, False),
        'fedkdl_nolora':    (True,  False, False, True,  False),
        'fedkdl_proxy_ft':  (False, True,  True,  False, True), 
    }
    f_p, u_l, u_i, u_kd, u_ft = cfg_map.get(baseline, (False, True, True, True, False))
    return {
        'full_param': f_p,
        'use_lora': u_l,
        'use_int8': u_i,
        'use_gateway_kd': u_kd,
        'use_gateway_proxy_ft': u_ft,
        'is_hfl': 'hfl' in baseline or baseline in ['flora', 'scaffold', 'fedkdl', 'fedkdl_nocoop', 'logit_kd', 'fedprox_kdl', 'topk_grad', 'fedkdl_nokd', 'fedkdl_nolora', 'fedkdl_proxy_ft']
    }

def print_tensor_dict(title, d):
    print(f"{title}:")
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}:\n{v.tolist()}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                 print(f"    {sub_k}:\n{sub_v.tolist()}")

def run_scenario(baseline):
    cfg = parse_baseline_config(baseline)
    print("\n" + "="*80)
    print(f"SCENARIO: {baseline.upper()}")
    print(f"Config: {cfg}")
    print("="*80)
    
    global_state = {}
    if cfg['full_param'] or not cfg['use_lora']:
        global_state['layer1'] = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    if cfg['use_lora']:
        global_state['model.0.lora_A'] = torch.tensor([[0.0, 0.0]])
        global_state['model.0.lora_B'] = torch.tensor([[0.0], [0.0]])
        
    auv0_payload = {}
    auv1_payload = {}
    
    if cfg['full_param'] or not cfg['use_lora']:
        auv0_payload['layer1'] = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        auv1_payload['layer1'] = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
        
    if cfg['use_lora']:
        auv0_payload['model.0.lora_A'] = torch.tensor([[1.0, 2.0]])
        auv0_payload['model.0.lora_B'] = torch.tensor([[1.0], [1.0]])
        auv1_payload['model.0.lora_A'] = torch.tensor([[2.0, 4.0]])
        auv1_payload['model.0.lora_B'] = torch.tensor([[2.0], [2.0]])
        
    if baseline == 'scaffold':
        auv0_payload['__scaffold_delta_c__'] = {'layer1': torch.tensor([[0.5, 0.5], [0.5, 0.5]])}
        auv1_payload['__scaffold_delta_c__'] = {'layer1': torch.tensor([[1.5, 1.5], [1.5, 1.5]])}
        
    print_tensor_dict("Input Payload AUV 0", auv0_payload)
    print("-" * 40)
    print_tensor_dict("Input Payload AUV 1", auv1_payload)
    print("-" * 40)
    
    if cfg['is_hfl']:
        # Relay Aggregation
        relay = RelayNode2D(relay_id=0, cluster_members=[0, 1])
        payloads = {0: auv0_payload, 1: auv1_payload}
        auv_n_samples = {0: 100, 1: 100}
        
        # Test Relay Intra-Cluster Aggregation
        # bypass int8 unpacking for dummy test by passing use_kd_lora_int8=False 
        # (svd_lora_aggregate inside RelayNode2D handles dicts directly)
        relay.aggregate_intra_cluster(global_state, payloads, auv_n_samples, use_kd_lora_int8=False)
        print_tensor_dict("Relay Output (Intra-Cluster Aggregation)", relay.final_state_dict)
        print("-" * 40)
        
        # Gateway Aggregation
        gateway = BaseGateway(global_state)
        gateway.aggregate_global({0: relay.final_state_dict}, {0: 200})
        print_tensor_dict("Gateway Output (Global Aggregation)", gateway.global_state_dict)
    else:
        # Flat Topology - Direct Gateway Aggregation
        gateway = BaseGateway(global_state)
        payloads = {0: auv0_payload, 1: auv1_payload}
        cluster_samples = {0: 100, 1: 100}
        gateway.aggregate_global(payloads, cluster_samples)
        print_tensor_dict("Gateway Output (Flat Global Aggregation)", gateway.global_state_dict)

if __name__ == '__main__':
    baselines = [
        'fedavg', 'fedprox', 'fedavg_hfl', 'fedprox_hfl', 'flora', 'scaffold', 
        'fedkdl', 'fedkdl_nocoop', 'logit_kd', 'fedprox_kdl', 'fedkd', 
        'topk_grad', 'centralized', 'fedkdl_nokd', 'fedkdl_nolora', 'fedkdl_proxy_ft'
    ]
    for b in baselines:
        run_scenario(b)
