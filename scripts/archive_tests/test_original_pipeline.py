import torch
import copy
from detection_2d.simulator import RelayNode2D
from federated_core.workers import BaseGateway

def main():
    print("=== Khởi tạo Global Server ===")
    global_weights = {'A': torch.tensor([1.0, 1.0]), 'B': torch.tensor([2.0, 2.0])}
    print(f"Global Weights ban đầu: {global_weights}")

    # Tạo Topology giả lập: 3 AUVs, 2 Relays
    clusters = {
        0: [1, 2],
        1: [3]
    }
    
    # Khởi tạo Relay Node
    relays = {
        0: RelayNode2D(relay_id=0, cluster_members=[1, 2]),
        1: RelayNode2D(relay_id=1, cluster_members=[3])
    }
    
    # Khởi tạo Gateway Node
    gateway = BaseGateway(initial_state=global_weights)

    print("\n=== [Tầng 1] AUV Local Training (Tạo Payload gửi lên Relay) ===")
    payloads = {}
    auv_n_samples = {1: 100, 2: 100, 3: 200} # Giả sử AUV 1,2 có 100 ảnh, AUV 3 có 200 ảnh
    
    for auv_id in [1, 2, 3]:
        # Giả lập kết quả train của từng AUV
        local_w = copy.deepcopy(global_weights)
        local_w['A'] += auv_id * 0.1  # AUV 1: +0.1, AUV 2: +0.2, AUV 3: +0.3
        local_w['B'] += auv_id * 0.1
        
        # Payload ở dạng Float32 (Dictionary Tensor)
        payloads[auv_id] = local_w
        print(f"AUV {auv_id} gửi Payload -> A: {local_w['A'].tolist()}, B: {local_w['B'].tolist()} (Samples: {auv_n_samples[auv_id]})")

    print("\n=== [Tầng 2] Relay Aggregation (Nội cụm) ===")
    # Relay 0 tổng hợp AUV 1 và 2
    relays[0].aggregate_intra_cluster(global_weights, payloads, auv_n_samples, use_kd_lora_int8=False)
    print(f"Relay 0 tổng hợp (AUV 1,2) -> A: {relays[0].final_state_dict['A'].tolist()}, B: {relays[0].final_state_dict['B'].tolist()}")

    # Relay 1 tổng hợp AUV 3
    relays[1].aggregate_intra_cluster(global_weights, payloads, auv_n_samples, use_kd_lora_int8=False)
    print(f"Relay 1 tổng hợp (AUV 3)   -> A: {relays[1].final_state_dict['A'].tolist()}, B: {relays[1].final_state_dict['B'].tolist()}")

    print("\n=== [Tầng 3] Gateway Global Update (SVD/FedAvg toàn cục) ===")
    # Chuẩn bị tham số cho Gateway
    relay_final_states = {
        0: relays[0].final_state_dict,
        1: relays[1].final_state_dict
    }
    cluster_total_samples = {
        0: sum(auv_n_samples[sid] for sid in clusters[0]), # 200
        1: sum(auv_n_samples[sid] for sid in clusters[1])  # 200
    }
    
    gateway.aggregate_global(relay_final_states, cluster_total_samples)
    final_w = gateway.global_state_dict
    
    print(f"Global Weights mới: A: {final_w['A'].tolist()}, B: {final_w['B'].tolist()}")
    
    # Kiểm tra toán học
    assert torch.allclose(final_w['A'], torch.tensor([1.225, 1.225])), "Toán học sai lệch!"
    print("\n✅ Thành công! Các Class gốc (RelayNode2D, BaseGateway) hoạt động hoàn hảo và giao tiếp chuẩn xác qua 3 tier.")

if __name__ == '__main__':
    main()
