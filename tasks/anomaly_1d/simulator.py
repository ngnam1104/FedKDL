import torch
import numpy as np
from typing import Dict, Any, Tuple

from federated_core.base_simulator import BaseSimulator
from federated_core.workers import BaseWorker, BaseFogNode, BaseGateway
# from tasks.anomaly_1d.dataloader import get_dataloaders # Removed since we build it locally
from tasks.anomaly_1d.autoencoder import SmallAutoencoder, get_model_state_dict_copy
from federated_core.metrics import anomaly_threshold, point_adjusted_f1


class SensorWorker1D(BaseWorker):
    def __init__(self, sensor_id, dataloader, model_template, battery_init, rho_s):
        super().__init__(sensor_id, battery_init)
        self.dataloader = dataloader
        self.n_samples = len(dataloader.dataset)
        
        import copy
        self.model = copy.deepcopy(model_template)
        
        from tasks.anomaly_1d.knowledge_compression.topk_sparsification import TopKCompressor
        total_params = sum(p.numel() for p in model_template.parameters())
        self.compressor = TopKCompressor(total_params=total_params, rho_s=rho_s)

    def train_and_get_payload(self, global_state, epochs, lr, mu, device):
        if not self.alive:
            return None, 0.0

        self.model.load_state_dict(global_state)
        
        from tasks.anomaly_1d.trainer import local_sgd
        delta_theta, avg_loss = local_sgd(
            model=self.model,
            dataloader=self.dataloader,
            epochs=epochs,
            lr=lr,
            global_model=None, # pass if needed
            mu=mu,
            device=device,
        )

        topk_indices, topk_values = self.compressor.compress(delta_theta)
        
        from tasks.anomaly_1d.knowledge_compression.int8_quantization import SparseINT8Payload
        payload = SparseINT8Payload(
            topk_indices=topk_indices,
            topk_values=topk_values,
            total_params=self.compressor.total_params,
        )
        return payload, avg_loss


class FogNode1D(BaseFogNode):
    def __init__(self, fog_id, cluster_members, model_template):
        super().__init__(fog_id, cluster_members)
        self.model_template = model_template

    def aggregate_intra_cluster(self, global_state_dict, payloads, sensor_n_samples, **kwargs):
        from federated_core.aggregator import fedavg_intra_cluster
        import copy
        client_deltas = []
        for sensor_id, payload in payloads.items():
            if sensor_id in self.cluster_members:
                dense_delta = payload.decompress()
                n_i = sensor_n_samples[sensor_id]
                client_deltas.append((dense_delta, n_i))

        self.intra_state_dict = fedavg_intra_cluster(
            global_state_dict=global_state_dict,
            client_deltas=client_deltas,
            model_template=self.model_template,
        )
        self.final_state_dict = copy.deepcopy(self.intra_state_dict)


class Simulator1D(BaseSimulator):
    def __init__(self, topo_path: str, data_path: str, baseline: str, device: str = "cpu"):
        super().__init__(topo_path=topo_path, baseline=baseline, device=device)
        self.task_key = "1D"
        
        from utils.env_manager import EnvironmentManager
        from tasks.anomaly_1d.dataloader import load_dataset, SlidingWindowDataset, make_client_loaders, make_val_loader
        
        data_part = EnvironmentManager.load_data_partition(data_path)
        self.dataset_name = data_part.dataset_name
        
        # Build dataloaders
        train_data, train_labels, val_parts, val_labels_parts, test_parts, test_labels_parts = load_dataset(
            self.dataset_name, seed=data_part.seed, per_channel_eval=True
        )
        
        train_ds = SlidingWindowDataset(train_data, train_labels, window_size=10)
        self.train_loaders = make_client_loaders(train_ds, data_part.client_data_indices, batch_size=64)
        
        self.val_loaders_per_channel = []
        self.test_loaders_per_channel = []
        
        for v_d, v_l in zip(val_parts, val_labels_parts):
            if len(v_d) >= 10:
                ds = SlidingWindowDataset(v_d, v_l, window_size=10)
                self.val_loaders_per_channel.append(make_val_loader(ds, batch_size=256))
            else:
                self.val_loaders_per_channel.append(None)
                
        for t_d, t_l in zip(test_parts, test_labels_parts):
            if len(t_d) >= 10:
                ds = SlidingWindowDataset(t_d, t_l, window_size=10)
                self.test_loaders_per_channel.append(make_val_loader(ds, batch_size=256))
            else:
                self.test_loaders_per_channel.append(None)

        
        # Initialize model
        first_loader = next(iter(self.train_loaders.values()))
        sample_batch, _ = next(iter(first_loader))
        self.input_dim = sample_batch.shape[1]
        self.model_template = SmallAutoencoder(input_dim=self.input_dim).to(self.device)
        self.fog_model_bits = self.model_template.count_parameters() * 32

        self.gateway = BaseGateway(initial_state=self.model_template.state_dict())
        
        # Initialize workers
        self._init_network()

    def _init_network(self):
        for s_id in range(self.net_cfg.N_SENSORS):
            if s_id in self.train_loaders and s_id in self.association:
                self.sensors[s_id] = SensorWorker1D(
                    sensor_id=s_id,
                    dataloader=self.train_loaders[s_id],
                    model_template=self.model_template,
                    battery_init=self.en_cfg.E_INIT,
                    rho_s=self.fed_cfg.RHO_S,
                )

        for m, members in self.clusters.items():
            self.fogs[m] = FogNode1D(
                fog_id=m,
                cluster_members=members,
                model_template=self.model_template,
            )

    def _process_sensor(self, s_id: int) -> Tuple[int, Any, float, int, float, float]:
        sensor = self.sensors[s_id]
        
        payload, avg_loss = sensor.train_and_get_payload(
            global_state=self.gateway.global_state_dict,
            epochs=self.fed_cfg.LOCAL_EPOCHS,
            lr=self.fed_cfg.LOCAL_LR,
            mu=0.0, # fedavg
            device=self.device,
        )

        from physics_models.energy import e_tx, e_comp_dynamic
        if payload is not None:
            # tx cost
            fog_id = self.association.get(s_id, -1)
            if fog_id == -1:
                link_key = ('sensor', s_id, 'gateway', 0)
            else:
                link_key = ('sensor', s_id, 'fog', fog_id)
                
            if link_key in self.G:
                link = self.G[link_key]
                e_tx_cost = e_tx(
                    payload.payload_bits, link.R_bps, link.SL_min,
                    self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                )
                # comp cost
                e_comp_cost = e_comp_dynamic(
                    sensor.n_samples, self.fed_cfg.LOCAL_EPOCHS,
                    self.fed_cfg.MODEL_FLOPS_PER_SAMPLE[self.task_key],
                    self.fed_cfg.FLOP_MULTIPLIER[self.task_key],
                    self.en_cfg.EPSILON_OP[self.task_key]
                )
                return s_id, payload, avg_loss, sensor.n_samples, e_tx_cost, e_comp_cost
        
        return s_id, None, 0.0, 0, 0.0, 0.0

    def _aggregate_intra_fog(self, m: int, fog, payloads, sensor_n_samples) -> float:
        fog.aggregate_intra_cluster(
            global_state_dict=self.gateway.global_state_dict,
            payloads=payloads,
            sensor_n_samples=sensor_n_samples,
        )
        return 0.0 # No inter-fog tx cost for intra-aggregation
        
    def _compute_payload_bits(self, payloads: Dict) -> float:
        return np.mean([p.payload_bits for p in payloads.values()]) if payloads else self.fog_model_bits

    def _compute_fog_model_bits(self) -> float:
        return self.fog_model_bits

    def evaluate(self) -> Dict[str, float]:
        self.model_template.load_state_dict(self.gateway.global_state_dict)
        self.model_template.eval()
        
        from federated_core.metrics import anomaly_threshold, point_adjusted_f1_components
        
        total_tp_pa = 0
        total_fp_pa = 0
        total_fn_pa = 0
        total_tp_std = 0
        total_fp_std = 0
        total_fn_std = 0
        val_loss_list, test_loss_list = [], []
        
        for v_loader, t_loader in zip(self.val_loaders_per_channel, self.test_loaders_per_channel):
            if v_loader is None or t_loader is None:
                continue
                
            val_errors = []
            with torch.no_grad():
                for x_val, y_val in v_loader:
                    x_val = x_val.to(self.device)
                    errs = self.model_template.reconstruction_error(x_val).cpu().numpy()
                    normal_errs = errs[y_val.numpy() == 0]
                    val_errors.extend(normal_errs)
            
            if len(val_errors) == 0:
                continue
                
            tau_A = anomaly_threshold(np.array(val_errors), percentile=99.0)
            
            test_errors = []
            test_labels = []
            with torch.no_grad():
                for x_test, y_test in t_loader:
                    x_test = x_test.to(self.device)
                    errs = self.model_template.reconstruction_error(x_test).cpu().numpy()
                    test_errors.extend(errs)
                    test_labels.extend(y_test.numpy())
                    
            if len(test_errors) == 0:
                continue
                
            tp_pa, fp_pa, fn_pa, tp_std, fp_std, fn_std = point_adjusted_f1_components(np.array(test_labels), np.array(test_errors), tau_A)
            total_tp_pa += tp_pa
            total_fp_pa += fp_pa
            total_fn_pa += fn_pa
            total_tp_std += tp_std
            total_fp_std += fp_std
            total_fn_std += fn_std
            
            val_loss_list.append(float(np.mean(val_errors)))
            test_loss_list.append(float(np.mean(test_errors)))
            
        prec = total_tp_pa / (total_tp_pa + total_fp_pa) if (total_tp_pa + total_fp_pa) > 0 else 0.0
        rec = total_tp_pa / (total_tp_pa + total_fn_pa) if (total_tp_pa + total_fn_pa) > 0 else 0.0
        pa_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        prec_std = total_tp_std / (total_tp_std + total_fp_std) if (total_tp_std + total_fp_std) > 0 else 0.0
        rec_std = total_tp_std / (total_tp_std + total_fn_std) if (total_tp_std + total_fn_std) > 0 else 0.0
        f1_std = 2 * prec_std * rec_std / (prec_std + rec_std) if (prec_std + rec_std) > 0 else 0.0
            
        def safe_mean(l): return float(np.mean(l)) if len(l) > 0 else 0.0
        
        return {
            'PA-F1': float(pa_f1), 'F1-Score': float(f1_std),
            'Prec': float(prec), 'Rec': float(rec),
            'Prec-Std': float(prec_std), 'Rec-Std': float(rec_std),
            'val_loss': safe_mean(val_loss_list), 'test_loss': safe_mean(test_loss_list)
        }
