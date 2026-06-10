import sys
from pathlib import Path
from unittest.mock import patch
sys.path.append(str(Path(__file__).resolve().parent))

from tasks.detection_2d.simulator import Simulator2D
import torch

def mock_local_sgd_od(student_model, auv_yaml, auv_id, epochs, batch_size, lr, device,
                      fedprox_mu, global_weights, local_teacher, cached_optimizer_state,
                      global_c, local_c):
    # Dummy training: just return the current trainable state dict, delta_norm=0.1, loss=1.0, optim=None
    state = student_model.trainable_state_dict()
    # Apply a tiny random noise to simulate training
    for k in state:
        state[k] = state[k] + torch.randn_like(state[k]) * 0.001
        
    if global_c is not None and local_c is not None:
        state['__scaffold_delta_c__'] = {k: torch.zeros_like(v) for k, v in local_c.items()}
        
    return state, 0.1, 1.0, None

def mock_evaluate_od(student_model, test_yaml, device):
    return {'mAP50': 0.5, 'mAP50-95': 0.3, 'Prec': 0.6, 'Rec': 0.6}

def mock_evaluate_od_on_auv_train(student_model, auv_yaml, device):
    return {'mAP50': 0.5, 'mAP50-95': 0.3, 'Prec': 0.6, 'Rec': 0.6}

def mock_KDDetectionTrainer_train(self):
    print("[Mock] KDDetectionTrainer.train() called")
    pass

@patch('tasks.detection_2d.simulator.local_sgd_od', side_effect=mock_local_sgd_od)
@patch('tasks.detection_2d.simulator.evaluate_od', side_effect=mock_evaluate_od)
@patch('tasks.detection_2d.simulator.evaluate_od_on_auv_train', side_effect=mock_evaluate_od_on_auv_train)
@patch('tasks.detection_2d.knowledge_compression.knowledge_distillation.KDDetectionTrainer.train', side_effect=mock_KDDetectionTrainer_train)
def test_baseline(baseline, m1, m2, m3, m4):
    print(f"\n{'='*50}\nTesting Baseline: {baseline}\n{'='*50}")
    try:
        sim = Simulator2D(
            baseline=baseline,
            yaml_config="config/urpc2020_env.yaml",
            dataset_yaml="datasets/urpc_data.yaml",
            student_ckpt="yolo12n.pt",
            teacher_ckpt="yolo12l.pt" if 'kd' in baseline or baseline in ['flora', 'centralized'] else None,
            experiment_name=f"test_{baseline}"
        )
        # Patch simulator parameters to make it fast
        sim.fed_cfg.GLOBAL_ROUNDS['2D'] = 1
        sim.net_cfg.N_AUVS = 4
        # Rebuild cluster topology with 4 AUVs
        sim._build_topology()
        
        sim.simulate()
        print(f"[{baseline}] Passed!")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{baseline}] Failed: {e}")
        return False

if __name__ == "__main__":
    baselines = [
        'fedavg', 'fedprox', 'fedavg_hfl', 'fedprox_hfl', 'flora', 'scaffold', 
        'fedkdl', 'fedkdl_nocoop', 'logit_kd', 'fedprox_kdl', 'fedkd', 
        'topk_grad', 'centralized', 'fedkdl_nokd', 'fedkdl_nolora', 'fedkdl_proxy_ft'
    ]
    results = {}
    for b in baselines:
        results[b] = test_baseline(b)
        
    print("\n\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for b, res in results.items():
        status = "✅ PASS" if res else "❌ FAIL"
        print(f"{b:<20}: {status}")
