import torch
from federated_core.workers import BaseRelayNode, BaseGateway

# --- Mock imports ---
class MockSimulator:
    pass

class MockG:
    pass

class RelayNode2D(BaseRelayNode):
    def aggregate_intra_cluster(self, global_state_dict, payloads, auv_n_samples, use_kd_lora_int8=True):
        import copy
        from federated_core.aggregator import svd_lora_aggregate
        
        c_updates = []
        delta_c_updates = []
        valid_sids = []
        for sid in self.cluster_members:
            if sid not in payloads:
                continue
            payload = payloads[sid]
            
            # --- Extract SCAFFOLD delta_c if present ---
            delta_c = None
            if isinstance(payload, dict) and '__scaffold_delta_c__' in payload:
                delta_c = payload.pop('__scaffold_delta_c__')
            
            # Mock INT8 or full payload
            state = payload
            c_updates.append(state)
            if delta_c is not None:
                delta_c_updates.append(delta_c)
            valid_sids.append(sid)

        if not c_updates:
            self.intra_state_dict = copy.deepcopy(global_state_dict)
            self.final_state_dict = copy.deepcopy(global_state_dict)
            return

        total_samples = sum(auv_n_samples.get(sid, 0) for sid in valid_sids)
        if total_samples == 0:
            total_samples = 1

        weights = [auv_n_samples.get(sid, 0) / total_samples for sid in valid_sids]
        
        # Fake SVD or FEDAVG for testing
        agg_state = {}
        for k in c_updates[0].keys():
            agg_state[k] = sum(c[k] * w for c, w in zip(c_updates, weights))
        
        self.intra_state_dict = agg_state
        
        if len(delta_c_updates) == len(c_updates):
            delta_c_agg = {}
            for k in delta_c_updates[0].keys():
                delta_c_agg[k] = sum(d[k] * w for d, w in zip(delta_c_updates, weights))
            self.intra_state_dict['__scaffold_delta_c__'] = delta_c_agg
            
        self.final_state_dict = copy.deepcopy(self.intra_state_dict)

def test_flat_topology():
    print("--- Test Flat Topology ---")
    gateway = BaseGateway({"layer1": torch.ones(2, 2)})
    relay_finals = {
        0: {"layer1": torch.ones(2, 2) * 2},
        1: {"layer1": torch.ones(2, 2) * 4}
    }
    cluster_samples = {0: 100, 1: 100}
    gateway.aggregate_global(relay_finals, cluster_samples)
    print("Global State (layer1):", gateway.global_state_dict["layer1"])
    assert torch.allclose(gateway.global_state_dict["layer1"], torch.ones(2, 2) * 3)
    print("Flat Topology Passed!")

def test_hfl_topology_scaffold():
    print("--- Test HFL Topology (SCAFFOLD) ---")
    global_state = {"layer1": torch.ones(2, 2)}
    
    relay = RelayNode2D(relay_id=0, cluster_members=[0, 1])
    
    payloads = {
        0: {"layer1": torch.ones(2, 2) * 2, "__scaffold_delta_c__": {"layer1": torch.ones(2, 2) * 0.5}},
        1: {"layer1": torch.ones(2, 2) * 4, "__scaffold_delta_c__": {"layer1": torch.ones(2, 2) * 1.5}},
    }
    auv_n_samples = {0: 100, 1: 100}
    
    relay.aggregate_intra_cluster(global_state, payloads, auv_n_samples, use_kd_lora_int8=False)
    
    print("Relay Intra State (layer1):", relay.intra_state_dict["layer1"])
    print("Relay Intra Delta C:", relay.intra_state_dict["__scaffold_delta_c__"]["layer1"])
    
    assert torch.allclose(relay.intra_state_dict["layer1"], torch.ones(2, 2) * 3)
    assert torch.allclose(relay.intra_state_dict["__scaffold_delta_c__"]["layer1"], torch.ones(2, 2) * 1.0)
    
    gateway = BaseGateway(global_state)
    gateway.aggregate_global({0: relay.final_state_dict}, {0: 200})
    
    print("Gateway Global State:", gateway.global_state_dict["layer1"])
    print("Gateway Global Delta C:", gateway.global_state_dict["__scaffold_delta_c__"]["layer1"])
    
    assert torch.allclose(gateway.global_state_dict["__scaffold_delta_c__"]["layer1"], torch.ones(2, 2) * 1.0)
    print("HFL Topology SCAFFOLD Passed!")

if __name__ == '__main__':
    test_flat_topology()
    test_hfl_topology_scaffold()
