import os
import gc
import copy
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from federated_core.base_simulator import BaseSimulator
from federated_core.workers import BaseWorker, BaseRelayNode, BaseGateway
from detection_2d.baselines import parse_baseline_config
from detection_2d.models.yolo_wrapper import StudentModel, TeacherModel
from detection_2d.trainer import local_sgd_od, evaluate_od, evaluate_od_on_auv_train
from detection_2d.knowledge_compression.int8_quantization import (
    pack_delta_payload,
    pack_payload,
    SparseINT8Payload,
    unpack_delta_payload,
)
from utils.image_payload import list_unique_image_files


class AUVWorker2D(BaseWorker):
    def __init__(self, auv_id, auv_yaml, battery_init):
        super().__init__(auv_id, battery_init)
        self.auv_yaml = auv_yaml
        # Cache optimizer state (exp_avg / exp_avg_sq) giữa các FL round.
        # None = chưa có (round đầu tiên, optimizer khởi đầu lạnh).
        self._optimizer_state: dict = None
        self._train_loader = None
        self._val_loader = None
        # TopK Compressor - lazy init khi biết số params (sau lần train đầu tiên)
        self._topk_compressor = None

        with open(self.auv_yaml, 'r') as f:
            c_cfg = yaml.safe_load(f)
        with open(c_cfg['train'], 'r') as f:
            self.n_samples = sum(1 for _ in f)

    def train_and_get_payload(
        self, global_state, epochs: int, lr: float, device: str,
        baseline: str, global_weights: dict = None, fedprox_mu_override: float = 0.0,
        nc: int = 4, student_ckpt: str = "yolo12n_warmup.pt",
        global_c: dict = None, local_c: dict = None,
    ):
        """
        Train local SGD (Tier 1, KHÔNG dùng KD) và đóng gói payload INT8.
        KD chỉ chạy tại Gateway (Tier 3) sau global aggregation.

        Returns:
            (payload_bytes, payload_kb, delta_norm, train_loss, local_metrics)
            payload_bytes : bytes đã nén INT8 để gửi qua kênh âm thanh
            payload_kb    : kích thước payload tính bằng KB
            delta_norm    : L2 norm của sự thay đổi trọng số (cho Lazy Filter)
            train_loss    : tổng box+cls+dfl loss vòng cuối
            local_metrics : dict mAP50-95/mAP50/Prec/Rec đánh giá trên tập train của chính auv
        """
        if not self.alive or getattr(self, 'n_samples', 0) == 0:
            if getattr(self, 'n_samples', 0) == 0:
                print(f"\n[{'='*40}]\n[AUV {self.auv_id}] BỎ QUA VÌ KHÔNG CÓ DỮ LIỆU (n_samples = 0)\n[{'='*40}]\n")
            return None, 0.0, 0.0, 0.0, {}

        import yaml
        with open(self.auv_yaml, 'r') as f:
            c_cfg = yaml.safe_load(f)
        nc = c_cfg.get('nc', 80)

        cfg = parse_baseline_config(baseline)
        full_param = cfg.full_param
        use_lora = cfg.use_lora
        use_int8 = cfg.use_int8
        
        from config.settings import fed_cfg
        fedprox_mu = fed_cfg.FEDPROX_MU if cfg.fedprox else 0.0
        # Nếu KD bị Adaptive Dropout tắt, fedprox_mu_override sẽ được truyền xuống từ Simulator
        fedprox_mu = max(fedprox_mu, fedprox_mu_override)
        rank = fed_cfg.LORA_R4_RANK if 'r4' in baseline else fed_cfg.LORA_RANK

        # [CRITICAL FIX] Use the exact SAME baseline student_ckpt (warmup_model) for local 
        # students so freezing non-payload layers uses the correct warmup backbone/head.
        local_student = StudentModel(
            student_ckpt,
            rank=rank,
            nc=nc,
            full_param=full_param,
            use_lora=use_lora,
            lora_targets=list(getattr(fed_cfg, 'LORA_TARGETS', ('Conv',))),
            lora_strategy=getattr(fed_cfg, 'LORA_STRATEGY', 'adaptive'),
        )
        local_student.load_trainable_state_dict(global_state)

        # Cấp phát Teacher cục bộ nếu chạy thuật toán FedKD (Local KD)
        local_teacher = None
        if cfg.local_kd:
            if not hasattr(self, 'local_teacher'):
                from detection_2d.models.yolo_wrapper import TeacherModel
                print(f"[Simulator2D] Khởi tạo Local Teacher (YOLO12l) dùng chung cho thuật toán FedKD...")
                # Teacher load sẵn weights pretrained
                self.local_teacher = TeacherModel(
                    "yolo12l_pretrained.pt",
                    rank=rank,
                    lora_targets=list(getattr(fed_cfg, 'LORA_TARGETS', ('Conv',))),
                    lora_strategy=getattr(fed_cfg, 'LORA_STRATEGY', 'adaptive'),
                )
                self.local_teacher.yolo.to(device)
            local_teacher = self.local_teacher

        # [TWEAK] ĐÃ GỠ BỎ: Trước đây giảm cực mạnh LR để chống Gradient Explosion khi AdamW bị cold-start
        # Nhưng nay đã chuyển sang SGD, SGD cần giữ nguyên LR gốc (0.001) để hội tụ tốt.
        # lr = lr * (0.1 if use_lora else 0.5)
        optimizer_state_for_round = self._optimizer_state
        if cfg.topk_grad:
            # Sparse Top-K sends only selected coordinates; stale AdamW moments
            # on unselected coordinates can accumulate drift and corrupt YOLO.
            optimizer_state_for_round = None
        if (
            optimizer_state_for_round
            and use_lora
            and getattr(fed_cfg, 'RESET_LORA_OPTIMIZER_STATE', True)
        ):
            # FL aggregation changes LoRA coordinates between rounds. This is
            # explicit for SVD-FLoRA and still disruptive for naive A/B averaging
            # under non-IID clients. Keep Head/BN moments only.
            optimizer_state_for_round = {
                key: value
                for key, value in optimizer_state_for_round.items()
                if 'lora_' not in key
            }

        new_state, delta_norm, train_loss, new_opt_state = local_sgd_od(
            student_model=local_student,
            auv_yaml=self.auv_yaml,
            auv_id=self.auv_id,
            epochs=epochs,
            batch_size=getattr(fed_cfg, 'LOCAL_BATCH_SIZE', 16),
            lr=lr,           # Truyền Cosine LR liên tục (không bị reset)
            device=device,
            fedprox_mu=fedprox_mu,
            global_weights=global_weights,
            local_teacher=local_teacher,
            cached_optimizer_state=optimizer_state_for_round,
            global_c=global_c,
            local_c=local_c,
            optimizer_name=getattr(fed_cfg, 'SCAFFOLD_OPTIMIZER', 'AdamW') if cfg.scaffold else None,
            cached_train_loader=self._train_loader,
            cached_val_loader=self._val_loader,
        )

        # Keep Ultralytics' dataset, label metadata, decoded/resized RAM images,
        # and InfiniteDataLoader alive for this AUV across subsequent rounds.
        if self._train_loader is None:
            self._train_loader = getattr(local_student, "_cached_train_loader", None)
            self._val_loader = getattr(local_student, "_cached_val_loader", None)
            if self._train_loader is not None:
                print(f"[AUV {self.auv_id}] Persistent RAM dataloader cache initialized.")

        if use_int8 and isinstance(new_state, dict) and '__scaffold_delta_c__' in new_state:
            # INT8 model payloads are shape-driven byte streams. Control variates
            # are metadata, not model tensors, so they must not enter pack_payload.
            # SCAFFOLD itself is currently a Float32 baseline in parse_baseline_config.
            new_state.pop('__scaffold_delta_c__')
        
        # Lưu optimizer state cho vòng sau
        self._optimizer_state = None if cfg.topk_grad else new_opt_state

        # [TỐI ƯU HÓA] Bỏ qua đánh giá local model trên tập train của auv
        # Việc này tiết kiệm 30% tổng thời gian huấn luyện mà không ảnh hưởng kết quả Global
        local_metrics = {}
        print(f"[AUV {self.auv_id}] Local train metrics skipped to save time.")

        if getattr(fed_cfg, 'LAZY_FILTER_ENABLED', False) and delta_norm < fed_cfg.DELTA_SKIP:
            print(f"[AUV {self.auv_id}] 💤 Lazy Filter Activated (delta={delta_norm:.4f} < {fed_cfg.DELTA_SKIP}). Node is resting (No TX).")
            payload_bytes = None
            payload_kb = 0.0
        elif cfg.topk_grad:
            # === TOP-K SPARSE FULL-MODEL UPDATE TRANSPORT ===
            from detection_2d.knowledge_compression.topk_sparsification import (
                TopKCompressor, SparseFloatPayload, flatten_state_dict
            )
            # Top-K is a full-parameter baseline, but the transmitted object is
            # the sparse update theta_local - theta_global. This is separate
            # from LoRA delta payloads used by FedKDL INT8.
            delta_state = {}
            for k in new_state:
                if (
                    k in global_state
                    and torch.is_tensor(new_state[k])
                    and torch.is_tensor(global_state[k])
                    and torch.is_floating_point(new_state[k])
                    and torch.is_floating_point(global_state[k])
                ):
                    delta_state[k] = new_state[k].cpu() - global_state[k].cpu()
            
            delta_flat, shapes = flatten_state_dict(delta_state)
            total_params = len(delta_flat)
            
            if self._topk_compressor is None or self._topk_compressor.total_params != total_params:
                rho_s = cfg.topk_ratio if cfg.topk_ratio is not None else getattr(fed_cfg, 'RHO_S', 0.05)
                self._topk_compressor = TopKCompressor(total_params=total_params, rho_s=rho_s)
                print(f"[AUV {self.auv_id}] [Top-K] Init sparse update compressor: total_params={total_params}, K={self._topk_compressor.K} (rho={rho_s:.2f})")

            topk_indices, topk_values = self._topk_compressor.compress(delta_flat)
            payload_bytes = SparseFloatPayload(
                topk_indices=topk_indices,
                topk_values=topk_values,
                total_params=total_params,
                shapes=shapes,
            )
            payload_kb = payload_bytes.payload_bytes / 1024.0
            print(f"[AUV {self.auv_id}] [Top-K] Sparse Float32 Payload: {payload_kb:.1f} KB (K={self._topk_compressor.K}/{total_params} params)")
        elif use_int8:
            # Quantize the local update rather than absolute weights. Payload
            # size is unchanged, while the smaller per-tensor range sharply
            # reduces repeated quantization error across FL rounds.
            use_delta_payload = getattr(fed_cfg, 'INT8_DELTA_PAYLOAD', True)
            if use_delta_payload:
                payload_bytes, payload_kb = pack_delta_payload(new_state, global_state)
            else:
                payload_bytes, payload_kb = pack_payload(new_state)
            payload_mode = "Delta" if use_delta_payload else "Raw"
            print(f"[AUV {self.auv_id}] Payload: {payload_kb:.1f} KB INT8 {payload_mode} "
                  f"(target ≤ {fed_cfg.TARGET_PAYLOAD_KB:.0f} KB)")
        else:
            # Fake packing for simulation (Float32 payload)
            payload_bytes = new_state
            # Calculate bytes based on float32 (4 bytes per param)
            total_params = 0
            for k, v in new_state.items():
                if k == '__scaffold_delta_c__':
                    total_params += sum(t.numel() for t in v.values())
                else:
                    total_params += v.numel()
            payload_kb = (total_params * 4) / 1024.0
            print(f"[AUV {self.auv_id}] Payload: {payload_kb:.1f} KB Float32")

        del local_student
        if getattr(fed_cfg, 'CLEAR_CUDA_CACHE_PER_AUV', False):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return payload_bytes, payload_kb, delta_norm, train_loss, local_metrics



class RelayNode2D(BaseRelayNode):
    def aggregate_intra_cluster(
        self,
        global_state_dict,
        payloads,
        auv_n_samples,
        use_kd_lora_int8=True,
        lora_aggregation="svd",
        global_student=None,
    ):
        import torch
        import copy
        from detection_2d.knowledge_compression.int8_quantization import unpack_payload
        from federated_core.aggregator import svd_lora_aggregate, weighted_state_dict_average
        
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
                delta_c = payload['__scaffold_delta_c__']
                payload = {key: value for key, value in payload.items() if key != '__scaffold_delta_c__'}
            
            # --- Phân biệt loại payload ---
            from detection_2d.knowledge_compression.topk_sparsification import (
                SparseFloatPayload, unflatten_state_dict
            )
            if isinstance(payload, (SparseFloatPayload, SparseINT8Payload)):
                # Top-K payload carries selected full-model update values.
                dense_delta = payload.decompress()
                delta_state = unflatten_state_dict(dense_delta, payload.shapes)
                state = {}
                for k, v in global_state_dict.items():
                    if torch.is_tensor(v):
                        state[k] = v.detach().cpu().clone()
                for k, v in delta_state.items():
                    if k in global_state_dict:
                        state[k] = global_state_dict[k].detach().cpu() + v
                    else:
                        state[k] = v
            elif use_kd_lora_int8 and isinstance(payload, (bytes, bytearray)):
                # INT8 update: reconstruct the client state from the exact
                # global state used at the beginning of this FL round.
                from config.settings import fed_cfg
                uplink_template = {
                    k: v for k, v in global_state_dict.items()
                    if global_student._is_payload_key(k, downlink=False)
                }
                if getattr(fed_cfg, 'INT8_DELTA_PAYLOAD', True):
                    state = unpack_delta_payload(payload, uplink_template)
                else:
                    state = unpack_payload(payload, uplink_template)
            else:
                # Float32 full-state dict (legacy fallback)
                state = payload
            
            c_updates.append(state)
            if delta_c is not None:
                delta_c_updates.append((delta_c, auv_n_samples.get(sid, 0)))
            valid_sids.append(sid)

        if not c_updates:
            self.intra_state_dict = copy.deepcopy(global_state_dict)
            self.final_state_dict = copy.deepcopy(global_state_dict)
            return

        total_samples = sum(auv_n_samples.get(sid, 0) for sid in valid_sids)
        if total_samples == 0:
            total_samples = 1

        weights = [auv_n_samples.get(sid, 0) / total_samples for sid in valid_sids]
        
        if lora_aggregation == "svd":
            self.intra_state_dict = svd_lora_aggregate(c_updates, weights)
        elif lora_aggregation == "naive":
            self.intra_state_dict = weighted_state_dict_average(c_updates, weights)
        else:
            raise ValueError(f"Unknown LoRA aggregation strategy: {lora_aggregation}")
        
        if len(delta_c_updates) == len(c_updates):
            delta_c_agg = {}
            total_c_samples = sum(n for _, n in delta_c_updates)
            if total_c_samples <= 0:
                delta_weights = [1.0 / len(delta_c_updates)] * len(delta_c_updates)
            else:
                delta_weights = [n / total_c_samples for _, n in delta_c_updates]
            for k in delta_c_updates[0][0].keys():
                weighted_terms = [
                    d[k] * w
                    for (d, _), w in zip(delta_c_updates, delta_weights)
                    if k in d
                ]
                if weighted_terms:
                    delta_c_agg[k] = sum(weighted_terms)
            self.intra_state_dict['__scaffold_delta_c__'] = delta_c_agg
            self.intra_state_dict['__scaffold_client_count__'] = len(delta_c_updates)
            
        self.final_state_dict = copy.deepcopy(self.intra_state_dict)


class Simulator2D(BaseSimulator):
    def __init__(
        self,
        topo_path: str,
        data_path: str,
        baseline: str,
        test_yaml: str = "datasets/URPC2020.yaml",
        student_ckpt: str = "yolo12n.pt",
        teacher_ckpt: str = "teacher_lora_best.pt",
        device: str = "cpu",
    ):
        super().__init__(topo_path=topo_path, baseline=baseline, device=device)
        self.test_yaml = test_yaml
        self.task_key = "2D"
        self.student_ckpt = student_ckpt
        self.baseline_cfg = parse_baseline_config(self.baseline)
        
        # [CRITICAL FIX] Đồng bộ cờ KD_ACTIVE toàn cục với cấu hình của baseline hiện tại
        # để ngăn chặn việc base_simulator.py đánh giá dư thừa 2 lần/vòng khi chạy các 
        # baseline không sử dụng Gateway KD (như fedkdl_nokd).
        self.fed_cfg.KD_ACTIVE = self.baseline_cfg.use_gateway_kd
        self.fed_cfg.GLOBAL_FT = getattr(self.baseline_cfg, 'use_gateway_proxy_ft', False)
        # Load Data Partition
        from utils.env_manager import EnvironmentManager
        data_part = EnvironmentManager.load_data_partition(data_path)
        self.data_part = data_part
        self.alpha = data_part.alpha
        self.auv_data_indices = data_part.auv_data_indices
        self.auv_yamls = []
        base_yaml_path = self.test_yaml
        if not os.path.exists(base_yaml_path):
            print(f"[Warning] Khong tim thay {base_yaml_path}. Su dung che do synthetic.")
            for i in range(self.net_cfg.N_AUVS):
                self.auv_yamls.append("coco8.yaml")
        else:
            with open(base_yaml_path, 'r') as f:
                base_cfg = yaml.safe_load(f)
            train_path = base_cfg.get('train', '')
            if isinstance(train_path, str) and train_path.endswith('.txt'):
                with open(train_path, 'r') as f:
                    all_images = [line.strip() for line in f.readlines()]
            else:
                dataset_dir = Path(base_yaml_path).parent
                original_path = base_cfg.get('path', '')
                
                # CƠ CHẾ DỰ PHÒNG TÌM ĐƯỜNG DẪN ẢNH (Phòng hờ cấu trúc thư mục trên Linux khác Windows)
                img_dir_candidates = [
                    dataset_dir / original_path / train_path,
                    dataset_dir / original_path.split('/')[0] / train_path,
                    dataset_dir / base_yaml_path.split('/')[-1].split('.')[0] / train_path
                ]
                
                img_dir = None
                for candidate in img_dir_candidates:
                    if candidate.exists() and candidate.is_dir():
                        img_dir = candidate
                        break
                
                if img_dir is None:
                    # Rà quét toàn bộ thư mục datasets để tìm train_path
                    for potential_dir in dataset_dir.glob(f'**/{train_path}'):
                        if potential_dir.is_dir():
                            img_dir = potential_dir
                            break
                            
                if img_dir is None or not img_dir.exists():
                    raise FileNotFoundError(f"CRITICAL: Không thể tìm thấy thư mục ảnh '{train_path}' ở bất kỳ đâu trong '{dataset_dir}'.")

                all_images = [
                    str(path)
                    for path in list_unique_image_files(img_dir)
                ]
                
                if not all_images:
                    raise FileNotFoundError(f"CRITICAL: Tìm thấy thư mục {img_dir} nhưng KHÔNG CÓ ẢNH NÀO bên trong! (Hỗ trợ: jpg, png, jpeg)")
                    
                self.all_images = all_images
                
                # Lưu file txt cho proxy KD sau này
                proxy_txt = Path("datasets/proxy_kd_train.txt")
                proxy_txt.parent.mkdir(parents=True, exist_ok=True)
                
                # Trích xuất riêng Public Data cho Gateway KD (nếu có định nghĩa trong data_part)
                if hasattr(data_part, 'public_data_indices') and data_part.public_data_indices:
                    public_images = [all_images[i] for i in data_part.public_data_indices]
                else:
                    public_images = all_images
                    
                with open(proxy_txt, "w") as f:
                    f.write("\n".join(public_images))
                self.proxy_kd_txt = str(proxy_txt.absolute())
                
            temp_dir = Path(f"datasets/URPC2020/auvs_temp_N{self.net_cfg.N_AUVS}_a{data_part.alpha}_s{data_part.seed}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.auv_yamls = {}
            for sid, idx_list in data_part.auv_data_indices.items():
                c_images = [all_images[i] for i in idx_list]
                txt_path = temp_dir / f"auv_{sid}_train.txt"
                with open(txt_path, 'w') as f:
                    f.write("\n".join(c_images))
                
                # Tạo file val giả chỉ có 1 ảnh để YOLO cache siêu nhanh (0.001s) thay vì cache lại toàn bộ train
                dummy_val_path = temp_dir / f"auv_{sid}_val.txt"
                with open(dummy_val_path, 'w') as f:
                    f.write(c_images[0] + "\n" if len(c_images) > 0 else "")

                c_yaml_path = temp_dir / f"auv_{sid}.yaml"
                c_cfg = base_cfg.copy()
                c_cfg['train'] = str(txt_path.absolute())
                if 'val' in c_cfg:
                    c_cfg['val'] = str(dummy_val_path.absolute())
                original_path = base_cfg.get('path', '')
                c_cfg['path'] = str((Path(base_yaml_path).parent / original_path).absolute())
                with open(c_yaml_path, 'w') as f:
                    yaml.safe_dump(c_cfg, f)
                self.auv_yamls[sid] = str(c_yaml_path)
            
            # Tạo proxy_test.yaml với đường dẫn tuyệt đối để evaluate_od không bị lỗi
            test_cfg = base_cfg.copy()
            original_path = base_cfg.get('path', '')
            if original_path:
                test_cfg['path'] = str((Path(base_yaml_path).parent / original_path).absolute())
            proxy_test_yaml = "datasets/proxy_test.yaml"
            with open(proxy_test_yaml, 'w') as f:
                yaml.safe_dump(test_cfg, f)
            self.test_yaml = str(Path(proxy_test_yaml).absolute())

        # Models
        nc = base_cfg.get('nc', 80) if 'base_cfg' in locals() else 80
        cfg = self.baseline_cfg
        full_param = cfg.full_param
        use_lora = cfg.use_lora
        rank = self.fed_cfg.LORA_R4_RANK if 'r4' in self.baseline else self.fed_cfg.LORA_RANK
        
        self.teacher = None
        if cfg.use_gateway_kd or cfg.local_kd:
            self.teacher = TeacherModel(
                teacher_ckpt,
                rank=rank,
                nc=nc,
                lora_targets=list(getattr(self.fed_cfg, 'LORA_TARGETS', ('Conv',))),
                lora_strategy=getattr(self.fed_cfg, 'LORA_STRATEGY', 'adaptive'),
            )
            self.teacher.yolo.to(self.device)
        self.global_student = StudentModel(
            student_ckpt,
            rank=rank,
            nc=nc,
            full_param=full_param,
            use_lora=use_lora,
            lora_targets=list(getattr(self.fed_cfg, 'LORA_TARGETS', ('Conv',))),
            lora_strategy=getattr(self.fed_cfg, 'LORA_STRATEGY', 'adaptive'),
        )
        
        self.gateway = BaseGateway(initial_state=self.global_student.trainable_state_dict())
        self._relay_topk_compressors = {}

        # [SCAFFOLD] Khởi tạo control variates
        if cfg.scaffold:
            import torch
            self.global_c = {
                k: torch.zeros_like(v)
                for k, v in self.gateway.global_state_dict.items()
                if torch.is_tensor(v) and torch.is_floating_point(v)
            }
            auv_ids = self.auv_yamls.keys() if isinstance(self.auv_yamls, dict) else range(len(self.auv_yamls))
            self.local_c_states = {
                auv_id: {k: torch.zeros_like(v) for k, v in self.global_c.items()}
                for auv_id in auv_ids
            }
        
        self._last_kd_metrics = {}

        if cfg.topk_grad:
            self.relay_model_bits = self._topk_state_bits(self.gateway.global_state_dict)
            print(
                f"[Simulator2D] Relay payload budget: "
                f"{self.relay_model_bits / 8.0 / 1024.0:.1f} KB Top-K "
                f"(rho={self._topk_ratio():.2f})"
            )
        elif cfg.use_int8:
            relay_payload_bytes, relay_payload_kb = pack_payload(self.gateway.global_state_dict)
            self.relay_model_bits = len(relay_payload_bytes) * 8
            print(f"[Simulator2D] Relay payload budget: {relay_payload_kb:.1f} KB INT8/FP16")
        else:
            self.relay_model_bits = sum(t.numel() for t in self.gateway.global_state_dict.values()) * 32
            print(f"[Simulator2D] Relay payload budget: {self.relay_model_bits / 8.0 / 1024.0:.1f} KB Float32")
        
        self._init_network()

    def _init_network(self):
        import os
        import yaml
        import numpy as np
        cfg = self.baseline_cfg

        def auv_yaml_for(s_id):
            if isinstance(self.auv_yamls, dict):
                return self.auv_yamls.get(s_id)
            if 0 <= s_id < len(self.auv_yamls):
                return self.auv_yamls[s_id]
            return None

        # 1. Hàm phụ trợ parse label histogram từ YOLO txt
        def get_label_histogram(auv_yaml_path, num_classes):
            hist = np.zeros(num_classes, dtype=np.float32)
            try:
                with open(auv_yaml_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                train_paths = cfg.get('train', [])
                if isinstance(train_paths, str):
                    train_paths = [train_paths]
                
                label_files = []
                for p in train_paths:
                    if p.endswith('.txt'):
                        with open(p, 'r') as tf:
                            for img_p in tf:
                                img_p = img_p.strip()
                                if not img_p: continue
                                lbl_p = img_p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                                label_files.append(lbl_p)
                    elif os.path.isdir(p):
                        lbl_dir = p.replace('images', 'labels')
                        if os.path.exists(lbl_dir):
                            for f in os.listdir(lbl_dir):
                                if f.endswith('.txt'):
                                    label_files.append(os.path.join(lbl_dir, f))
                
                for lf in set(label_files):
                    if os.path.exists(lf):
                        with open(lf, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    c = int(parts[0])
                                    if 0 <= c < num_classes:
                                        hist[c] += 1
                if hist.sum() > 0:
                    normalized = hist / hist.sum()
                else:
                    normalized = np.ones(num_classes, dtype=np.float32) / num_classes
            except Exception as e:
                print(f"[Warning] Không thể đọc histogram từ {auv_yaml_path}: {e}")
                hist = np.zeros(num_classes, dtype=np.float32)
                normalized = np.ones(num_classes, dtype=np.float32) / num_classes
            return normalized, hist

        # 2. Xây dựng Histogram cho toàn bộ mạng lưới
        N = self.net_cfg.N_AUVS
        M = getattr(self.net_cfg, 'M_RELAYS_2D', self.net_cfg.M_RELAYS)
        model_yaml = getattr(self.global_student.yolo.model, 'yaml', {})
        nc = model_yaml.get('nc', 80) if isinstance(model_yaml, dict) else 80
        if not isinstance(nc, int):
            nc = 80

        if cfg.hfl and getattr(self.fed_cfg, 'BETA_EMD', 0.0) > 0.0:
            print(f"\n[Simulator2D] Building knowledge-aware association (EMD beta={self.fed_cfg.BETA_EMD})...")
            self.auv_label_hists = np.zeros((N, nc), dtype=np.float32)
            self.auv_label_counts = np.zeros((N, nc), dtype=np.float32)
            for s_id in range(N):
                auv_yaml = auv_yaml_for(s_id)
                if auv_yaml is not None:
                    n_hist, c_hist = get_label_histogram(auv_yaml, nc)
                    self.auv_label_hists[s_id] = n_hist
                    self.auv_label_counts[s_id] = c_hist
            
            self.relay_label_hists = np.zeros((M, nc), dtype=np.float32)
            relay_counts = np.zeros(M)
            for s_id, f_id in self.association.items():
                if 0 <= f_id < M:
                    self.relay_label_hists[f_id] += self.auv_label_hists[s_id]
                    relay_counts[f_id] += 1
            for m in range(M):
                if relay_counts[m] > 0:
                    self.relay_label_hists[m] /= relay_counts[m]
                else:
                    self.relay_label_hists[m] = np.ones(nc, dtype=np.float32) / nc

            class DummyTopo:
                def __init__(self, n, m): self.N = n; self.M = m
            
            from detection_2d.knowledge_compression.knowledge_association import knowledge_aware_association
            new_association = knowledge_aware_association(
                topology=DummyTopo(N, M),
                G=self.G,
                auv_label_hists=self.auv_label_hists,
                relay_label_hists=self.relay_label_hists,
                beta=self.fed_cfg.BETA_EMD,
            )
            
            changed = 0
            changes_log = []
            for s, new_f in new_association.items():
                old_f = self.association.get(s, -1)
                if old_f != new_f:
                    changed += 1
                    changes_log.append(f"    - AUV {s}: Relay {old_f} -> Relay {new_f}")
            print(f"[Simulator2D] Knowledge-aware association done. {changed}/{N} AUVs changed relays.")
            if changed > 0:
                for log in changes_log:
                    print(log)
            
            self.association = new_association
        elif cfg.hfl:
            print("\n[Simulator2D] Knowledge-aware association disabled; using physical association.")

        if cfg.hfl:
            self.clusters = {m: [] for m in range(M)}
            for s, f in self.association.items():
                if f in self.clusters:
                    self.clusters[f].append(s)
        else:
            # Flat AUV->Gateway still needs one virtual aggregation group so the
            # common gateway FedAvg path can aggregate all directly connected AUVs.
            self.clusters = {0: sorted(self.association)}

        # 3. Tiến hành cấp phát Worker/Node như bình thường
        for s_id in range(self.net_cfg.N_AUVS):
            auv_yaml = auv_yaml_for(s_id)
            if auv_yaml is not None:
                self.auvs[s_id] = AUVWorker2D(
                    auv_id=s_id,
                    auv_yaml=auv_yaml,
                    battery_init=self.en_cfg.E_INIT,
                )
                if s_id not in self.association:
                    print(f"\n[{'='*40}]\n[AUV {s_id}] BỎ QUA VÌ KHÔNG THỎA MÃN KHOẢNG CÁCH (Out of Range)\n[{'='*40}]\n")

        for m, members in self.clusters.items():
            self.relays[m] = RelayNode2D(
                relay_id=m,
                cluster_members=members,
                battery_init=self.en_cfg.RELAY_E_INIT,
            )

    def get_flop_multiplier(self) -> float:
        cfg = self.baseline_cfg
        if cfg.local_kd:
            # FedKD chạy Teacher (YOLOv12l) Forward pass + Student (YOLOv12n) Full pass.
            # Tỷ lệ FLOPs của YOLOv12l so với YOLOv12n là ~30 lần. Cộng thêm 3 lần cho Student.
            return 33.0 
        elif cfg.full_param:
            # Full param update (Forward + Backward qua tất cả tham số)
            return 3.0
        # Mặc định (LoRA)
        return self.fed_cfg.FLOP_MULTIPLIER[self.task_key]

    def _process_auv(self, s_id: int) -> Tuple[int, Any, float, int, float, float, dict]:
        auv = self.auvs[s_id]
        cfg = self.baseline_cfg

        if s_id == 0 and getattr(self, '_fedprox_mu_override', 0.0) > 0.0:
            print(f"    [!] Adaptive Dropout Active: AUVs are training with FedProx (mu={self._fedprox_mu_override})")

        payload, payload_kb, delta_norm, train_loss, local_metrics = auv.train_and_get_payload(
            global_state=self.gateway.global_state_dict,
            epochs=self.fed_cfg.LOCAL_EPOCHS,
            lr=getattr(self, 'current_lr', self.fed_cfg.LOCAL_LR),
            device=self.device,
            baseline=self.baseline,
            fedprox_mu_override=getattr(self, '_fedprox_mu_override', 0.0),
            student_ckpt=self.student_ckpt,
            # global_weights cần thiết cho FedProx proximal term
            global_weights=self.gateway.global_state_dict if (
                cfg.fedprox or getattr(self, '_fedprox_mu_override', 0.0) > 0.0
            ) else None,
            global_c=getattr(self, 'global_c', None),
            local_c=self.local_c_states[s_id] if hasattr(self, 'local_c_states') else None,
        )

        use_int8 = cfg.use_int8
        
        from physics_models.energy import e_tx, e_comp
        if payload is not None:
            if not use_int8:
                S_bits = self._payload_to_bits(payload)
            else:
                S_bits = len(payload) * 8  # payload luôn là bytes INT8

            relay_id = self.association.get(s_id, -1)
            if relay_id == -1:
                link_key = ('auv', s_id, 'gateway', 0)
            else:
                link_key = ('auv', s_id, 'relay', relay_id)
                
            e_tx_cost = 0.0
            e_comp_cost = 0.0
            if link_key in self.G:
                link = self.G[link_key]
                e_tx_cost = e_tx(
                    S_bits, link.R_bps, link.SL_min,
                    self.en_cfg.ETA_EA, self.en_cfg.P_C_TX,
                )

                e_comp_cost = e_comp(
                    n_samples=auv.n_samples,
                    local_epochs=self.fed_cfg.LOCAL_EPOCHS,
                    flops_per_sample=self.fed_cfg.MODEL_FLOPS_PER_SAMPLE[self.task_key],
                    flop_multiplier=self.get_flop_multiplier(),
                    epsilon_op=self.en_cfg.EPSILON_OP[self.task_key],
                    f_cpu=self.en_cfg.F_CPU,
                )

                if auv.battery >= (e_tx_cost + e_comp_cost):
                    return s_id, payload, train_loss, auv.n_samples, e_tx_cost, e_comp_cost, local_metrics
                else:
                    auv.alive = False

        return s_id, None, 0.0, 0, 0.0, 0.0, {}


    def _aggregate_intra_relay(self, m: int, relay, payloads, auv_n_samples) -> float:
        cfg = self.baseline_cfg
        use_int8 = cfg.use_int8
        relay.aggregate_intra_cluster(
            global_state_dict=self.gateway.global_state_dict,
            payloads=payloads,
            auv_n_samples=auv_n_samples,
            use_kd_lora_int8=use_int8,
            lora_aggregation=cfg.lora_aggregation,
            global_student=self.global_student,
        )
        return 0.0

    def _topk_ratio(self) -> float:
        return float(
            self.baseline_cfg.topk_ratio
            if self.baseline_cfg.topk_ratio is not None
            else getattr(self.fed_cfg, 'RHO_S', 0.05)
        )

    def _topk_state_bits(self, state) -> float:
        total_params = sum(
            tensor.numel()
            for tensor in state.values()
            if torch.is_tensor(tensor) and torch.is_floating_point(tensor)
        )
        if total_params <= 0:
            return 0.0
        k = max(1, int(total_params * self._topk_ratio()))
        header_bits = 32 + 8
        return float(k * (8 + 32) + header_bits)  # INT8 value + int32 index

    def _transport_topk_relay_state(self, state, relay_id=None):
        from detection_2d.knowledge_compression.topk_sparsification import (
            TopKCompressor,
            flatten_state_dict,
            unflatten_state_dict,
        )

        delta_state = {
            key: value.cpu() - self.gateway.global_state_dict[key].cpu()
            for key, value in state.items()
            if torch.is_tensor(value) and key in self.gateway.global_state_dict
            and torch.is_floating_point(value)
            and torch.is_floating_point(self.gateway.global_state_dict[key])
        }
        delta_flat, shapes = flatten_state_dict(delta_state)
        total_params = len(delta_flat)
        compressor_key = relay_id if relay_id is not None else "__r2r__"
        compressor = self._relay_topk_compressors.get(compressor_key)
        if compressor is None or compressor.total_params != total_params:
            compressor = TopKCompressor(total_params=total_params, rho_s=self._topk_ratio())
            self._relay_topk_compressors[compressor_key] = compressor
            print(
                f"[Relay {relay_id}] [Top-K R2G] Init sparse update compressor: "
                f"total_params={total_params}, K={compressor.K} "
                f"(rho={self._topk_ratio():.2f})"
            )

        topk_indices, topk_values = compressor.compress(delta_flat)
        payload = SparseINT8Payload(
            topk_indices=topk_indices,
            topk_values=topk_values,
            total_params=total_params,
            shapes=shapes,
        )
        sparse_delta = payload.decompress()
        delta_sparse_state = unflatten_state_dict(sparse_delta, shapes)
        transported = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            if (
                key in self.gateway.global_state_dict
                and torch.is_tensor(self.gateway.global_state_dict[key])
                and torch.is_floating_point(value)
                and torch.is_floating_point(self.gateway.global_state_dict[key])
            ):
                transported[key] = self.gateway.global_state_dict[key].clone().cpu()
            else:
                transported[key] = value.clone().cpu()
        for key, delta in delta_sparse_state.items():
            transported[key] = self.gateway.global_state_dict[key].cpu() + delta
        return transported

    def _transport_relay_state(self, state, relay_id=None):
        """Apply the modeled Relay link codec to R2R and R2G learning states."""
        if self.baseline_cfg.topk_grad:
            # Top-K is only applied on the constrained AUV -> Relay hop. A
            # second sparse projection on Relay -> Gateway drops too much YOLO
            # signal, so the relay forwards its aggregated full state.
            return state
        if not self.baseline_cfg.use_int8:
            return state
        uplink_template = {
            k: v for k, v in self.gateway.global_state_dict.items()
            if self.global_student._is_payload_key(k, downlink=False)
        }
        if getattr(self.fed_cfg, 'INT8_DELTA_PAYLOAD', True):
            payload, _ = pack_delta_payload(state, self.gateway.global_state_dict)
            return unpack_delta_payload(payload, uplink_template)
        payload, _ = pack_payload(state)
        return unpack_payload(payload, uplink_template)

    def _pre_warm_dataset_cache(self):
        """
        [OPTION B] Pre-warm YOLO label cache cho tất cả AUV datasets TRƯỚC vòng FL đầu tiên.
        Mục tiêu: loại bỏ chi phí scan/index labels lặp lại ở mỗi vòng (tiết kiệm ~15-25%).
        
        Cơ chế: Chạy Ultralytics scan labels + build .cache file song song bằng ThreadPoolExecutor.
        Các vòng FL sau sẽ load trực tiếp từ file cache → không cần scan lại.
        """
        import concurrent.futures
        from pathlib import Path

        if not hasattr(self, 'auv_yamls') or not self.auv_yamls:
            return

        yamls = list(self.auv_yamls.values()) if isinstance(self.auv_yamls, dict) else list(self.auv_yamls)
        print(f"\n[Option B] Pre-warming YOLO label cache cho {len(yamls)} AUV datasets (song song)...")

        def _warm_one(yaml_path: str):
            try:
                import yaml as _yaml
                from pathlib import Path as _P
                from ultralytics.data import YOLODataset
                with open(yaml_path, 'r') as f:
                    cfg = _yaml.safe_load(f)
                train_path = cfg.get('train', '')
                if not train_path:
                    return yaml_path, False
                # Chạy scan để build label cache file (*.cache)
                ds = YOLODataset(
                    img_path=train_path,
                    imgsz=640,
                    augment=False,
                    batch_size=1,
                    data=cfg,
                    task='detect',
                )
                # Trigger cache build nếu chưa có
                _ = len(ds)
                return yaml_path, True
            except Exception as e:
                return yaml_path, f"ERR:{e}"

        max_w = min(len(yamls), 8)  # Tối đa 8 workers để không quá tải I/O
        ok_count = 0
        err_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as ex:
            futures = [ex.submit(_warm_one, y) for y in yamls]
            for fut in concurrent.futures.as_completed(futures):
                _, status = fut.result()
                if status is True:
                    ok_count += 1
                else:
                    err_count += 1
        print(f"[Option B] Cache pre-warm xong: {ok_count} OK, {err_count} skipped/error.\n")


    def _compute_payload_bits(self, payloads: Dict) -> float:
        if not payloads:
            return self.relay_model_bits

        bits_list = [
            self._payload_to_bits(payload)
            for payload in payloads.values()
        ]
        
        return np.mean(bits_list) if bits_list else 0.0

    def _compute_relay_model_bits(self) -> float:
        multiplier = 2 if self.baseline_cfg.scaffold else 1
        return self.relay_model_bits * multiplier

    def _build_gateway_proxy_yaml(self) -> str:
        proxy_yaml = getattr(self, 'test_yaml', "coco8.yaml")
        if proxy_yaml != "coco8.yaml" and os.path.exists(proxy_yaml):
            with open(proxy_yaml, 'r') as f:
                p_cfg = yaml.safe_load(f)
            if 'path' in p_cfg and 'val' in p_cfg:
                dataset_dir = Path(proxy_yaml).parent
                base_dir = dataset_dir / p_cfg['path']
                if isinstance(p_cfg['val'], str):
                    p_cfg['val'] = str((base_dir / p_cfg['val']).resolve())
                elif isinstance(p_cfg['val'], list):
                    p_cfg['val'] = [str((base_dir / v).resolve()) for v in p_cfg['val']]

            p_cfg.pop('path', None)
            p_cfg['train'] = getattr(self, 'proxy_kd_txt', '')
            proxy_yaml_abs = "datasets/proxy_kd_data.yaml"
            with open(proxy_yaml_abs, 'w') as f:
                yaml.safe_dump(p_cfg, f)
            proxy_yaml = str(Path(proxy_yaml_abs).absolute())
        return proxy_yaml

    def _gateway_supervised_finetune(self, current_round: int = 1, total_rounds: int = 40):
        """
        Fine-tune the global model on proxy data at the gateway without Teacher KD.
        """
        from detection_2d.trainer import CustomDetectionTrainer

        def _gateway_quality(metrics: dict) -> float:
            if not metrics:
                return 0.0
            return (
                float(metrics.get('mAP50-95', 0.0))
                + 0.25 * float(metrics.get('mAP50', 0.0))
                + 0.35 * float(metrics.get('Rec', 0.0))
                + 0.05 * float(metrics.get('Prec', 0.0))
            )

        def _metric_delta(metrics: dict, baseline: dict, key: str) -> float:
            return float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))

        proxy_yaml = self._build_gateway_proxy_yaml()

        # --- Progressive Proxy FT Logic ---
        # 1. Global Cosine LR Decay: Thay vì giữ LR hằng số, ta giảm dần LR theo số vòng FL
        # để giống hệt Centralized Training, giúp model hội tụ sâu (tiệm cận mốc 0.37 mAP).
        base_lr = self.fed_cfg.PROXY_FT_LR
        lrf_global = getattr(self.fed_cfg, 'PROXY_FT_LRF_GLOBAL', 0.01) # Decay về 1%
        progress = max(0.0, min(1.0, (current_round - 1) / max(1, total_rounds - 1)))
        global_lr = float(base_lr * (lrf_global + (1 - lrf_global) * (1 + np.cos(np.pi * progress)) / 2))

        # 2. Progressive Epochs: Tăng số epoch khi model bão hòa để "đào sâu" proxy data.
        base_epochs = getattr(self.fed_cfg, 'PROXY_FT_EPOCHS', 1)
        dynamic_epochs = base_epochs + int(progress * 2) # Vòng cuối tăng thêm nhẹ, tránh overfit proxy.

        # 3. Progressive Augmentation: Các vòng đầu tắt augment để tránh sốc. Nửa sau bật lại.
        use_augment = (progress > 0.35)

        print(f"[Progressive Proxy-FT] Round {current_round}/{total_rounds} | LR: {global_lr:.6f} | Epochs: {dynamic_epochs} | Augment: {use_augment}")

        overrides = {
            'model': self.student_ckpt,
            'data': proxy_yaml,
            'epochs': dynamic_epochs,
            'batch': self.fed_cfg.PROXY_FT_BATCH_SIZE,
            'device': self.device,
            'project': 'runs/gateway_proxy_ft',
            'name': 'global_proxy_ft',
            'exist_ok': True,
            'verbose': False,
            'save': False,
            'val': False,
            'plots': False,
            'workers': self.fed_cfg.PROXY_FT_WORKERS,
            'close_mosaic': 0,
            'mosaic': 0.0,
            'augment': use_augment,
            'fliplr': 0.5 if use_augment else 0.0,
            'scale': 0.25 if use_augment else 0.0,
            'translate': 0.05 if use_augment else 0.0,
            'hsv_h': 0.010 if use_augment else 0.0,
            'hsv_s': 0.35 if use_augment else 0.0,
            'hsv_v': 0.20 if use_augment else 0.0,
            'erasing': 0.0,
            'optimizer': getattr(self.fed_cfg, 'PROXY_FT_OPTIMIZER', 'AdamW'),
            'lr0': global_lr,
            'lrf': 1.0,
            # [FIX] Warmup 10% (khoảng 20 batch đầu) để AdamW tự tích lũy đà, tránh giật LR ở Round 1
            'warmup_epochs': getattr(self.fed_cfg, 'PROXY_FT_WARMUP_EPOCHS', 0.1) if current_round == 1 else 0.0,
            'warmup_bias_lr': global_lr,
        }

        reuse_proxy_optimizer = bool(getattr(self.fed_cfg, 'PROXY_FT_REUSE_OPTIMIZER', True))
        previous_optimizer_state = (
            getattr(self.gateway, 'proxy_ft_optimizer_state', None)
            if reuse_proxy_optimizer else None
        )
        trainer = CustomDetectionTrainer(
            overrides=overrides,
            student_wrapper=self.global_student,
            cached_optimizer_state=previous_optimizer_state
        )
        # Keep Proxy-FT and KD optimizer settings identical so the ablation
        # isolates teacher supervision rather than differential learning rates.
        trainer.head_lr_multiplier = getattr(
            self.fed_cfg, 'PROXY_FT_HEAD_LR_MULT',
            getattr(self.fed_cfg, 'KD_HEAD_LR_MULT', 4.0),
        )
        trainer.lora_lr_multiplier = getattr(
            self.fed_cfg, 'PROXY_FT_LORA_LR_MULT',
            getattr(self.fed_cfg, 'KD_LORA_LR_MULT', 1.0),
        )
        self.global_student.strip_inference_tensors()
        self.global_student.load_trainable_state_dict(self.gateway.global_state_dict)
        gateway_pre_state = {
            k: v.clone().cpu()
            for k, v in self.gateway.global_state_dict.items()
        }
        
        base_wise_alpha = getattr(self.fed_cfg, 'PROXY_FT_WISE_ALPHA', 1.0)
        # [WiSE-FT Dynamic Decay] Giảm dần alpha theo quá trình huấn luyện.
        # Ở các vòng đầu, model yếu, tin tưởng Proxy FT nhiều (alpha cao).
        # Ở các vòng cuối, model đã tổng hợp được FL tốt, tin tưởng Proxy FT ít lại (alpha thấp)
        # để tránh catastrophic forgetting (xóa nhòa tri thức dưới nước).
        if base_wise_alpha < 1.0:
            min_alpha = getattr(self.fed_cfg, 'PROXY_FT_WISE_ALPHA_MIN', 0.05)
            decay_ratio = max(0.0, min(1.0, (current_round - 1) / max(1, total_rounds - 1)))
            wise_alpha = base_wise_alpha - decay_ratio * (base_wise_alpha - min_alpha)
            print(f"[WiSE-FT Dynamic Decay] Round {current_round}/{total_rounds} -> alpha decayed from {base_wise_alpha} to {wise_alpha:.3f}")
            pre_ft_state = gateway_pre_state
        else:
            wise_alpha = 1.0
            pre_ft_state = gateway_pre_state

        # [WiSE-FT] Dùng pure_aggregated_state (FedAvg thuần, chưa qua server_mix)
        # làm tham chiếu để blend. Tránh double-momentum:
        #   Không có WiSE-FT: server_mix giữ 10% w_old (bình thường)
        #   Có WiSE-FT:       server_mix bị bỏ qua, WiSE-FT blend w_post + w_fedavg_thuần
        if wise_alpha < 1.0:
            pure_state = getattr(self.gateway, 'pure_aggregated_state', None)
            if (
                getattr(self.fed_cfg, 'PROXY_FT_BLEND_REFERENCE', 'pre_gateway') == 'pure_aggregated'
                and pure_state is not None
            ):
                pre_ft_state = {k: v.clone().cpu() for k, v in pure_state.items()}
                print(f"[WiSE-FT] Using pure FedAvg state as blend reference (not server-mixed).")
            else:
                # Fallback: dùng global_state_dict (đã mixed) nếu chưa có pure state
                pre_ft_state = {
                    k: v.clone().cpu()
                    for k, v in gateway_pre_state.items()
                }
                print(f"[WiSE-FT] Using pre-gateway state as blend reference.")
        
        # [MOD] Khi lên gateway thì mở full head để finetune
        head_idx = len(self.global_student.yolo.model.model) - 1
        head_prefix = f'model.{head_idx}.'
        for name, param in self.global_student.yolo.model.named_parameters():
            if head_prefix in name:
                param.requires_grad_(True)
                
        trainer._fl_injected_model = self.global_student.yolo.model
        trainer.model = self.global_student.yolo.model

        print(f"[Gateway Proxy-FT] Fine-tuning global model on proxy data without Teacher KD...")
        trainer.train()
        candidate_optimizer_state = trainer.get_named_optimizer_state()
        post_ft_state = self.global_student.trainable_state_dict(downlink=True)

        def _blend_proxy_state(alpha: float) -> dict:
            if alpha >= 1.0:
                return {k: v.clone().cpu() for k, v in post_ft_state.items()}
            blended_state = {
                k: alpha * post_ft_state[k].to(pre_ft_state[k].device) + (1.0 - alpha) * pre_ft_state[k]
                for k in pre_ft_state
                if k in post_ft_state
            }
            for k, v in pre_ft_state.items():
                if k not in blended_state:
                    blended_state[k] = v.clone()
            for k, v in post_ft_state.items():
                if k not in blended_state:
                    blended_state[k] = v.clone().cpu()
            return blended_state

        alpha_candidates = [float(wise_alpha)]
        for alpha in getattr(self.fed_cfg, 'PROXY_FT_WISE_ALPHA_CANDIDATES', ()):
            alpha = float(alpha)
            if 0.0 < alpha <= 1.0:
                alpha_candidates.append(alpha)
        alpha_candidates = sorted(set(round(a, 4) for a in alpha_candidates), reverse=True)

        # [WiSE-FT] Blend w_post với w_pre theo alpha:
        # w_final = alpha * w_post + (1 - alpha) * w_pre
        # alpha = 1.0 → thuần proxy FT (hành vi cũ)
        # alpha = 0.7 → giữ lại 30% đa dạng FL, tránh overwrite tri thức underwater
        if wise_alpha < 1.0:
            blended_state = _blend_proxy_state(wise_alpha)
            self.global_student.load_trainable_state_dict(blended_state)
            print(f"[Gateway Proxy-FT WiSE-FT] Blended alpha={wise_alpha:.2f}: "
                  f"{wise_alpha*100:.0f}% FT + {(1-wise_alpha)*100:.0f}% pre-FT FL weights.")

        # [ASYMMETRIC DOWNLINK] Gateway broadcasts FULL head + LoRA to AUVs/Relays.
        # Uplink (AUV→Gateway) only sends partial head to save battery.
        # Downlink (Gateway→AUV) sends full head since Gateway has power budget.
        self.gateway.global_state_dict = self.global_student.trainable_state_dict(downlink=True)
        pre_metrics = getattr(self, '_last_pre_gateway_metrics', {})
        candidate_metrics = {}
        accepted = True
        map_delta = 0.0
        map50_delta = 0.0
        rec_delta = 0.0
        quality_delta = 0.0
        if getattr(self.fed_cfg, 'PROXY_FT_ACCEPTANCE_GATE', True) and pre_metrics:
            print(f"[Gateway Proxy-FT] Evaluating {len(alpha_candidates)} WiSE alpha candidate(s): {alpha_candidates}")
            accept_tol = float(getattr(self.fed_cfg, 'PROXY_FT_ACCEPT_TOL', 5e-4))
            min_map_delta = float(getattr(self.fed_cfg, 'PROXY_FT_MIN_MAP5095_DELTA', -accept_tol))
            min_map50_delta = float(getattr(self.fed_cfg, 'PROXY_FT_MIN_MAP50_DELTA', -accept_tol))
            min_rec_delta = float(getattr(self.fed_cfg, 'PROXY_FT_MIN_REC_DELTA', -accept_tol))
            best_candidate = None

            for alpha in alpha_candidates:
                trial_state = _blend_proxy_state(alpha)
                self.gateway.global_state_dict = {k: v.clone().cpu() for k, v in trial_state.items()}
                trial_metrics = self.evaluate()
                trial_map_delta = _metric_delta(trial_metrics, pre_metrics, 'mAP50-95')
                trial_map50_delta = _metric_delta(trial_metrics, pre_metrics, 'mAP50')
                trial_rec_delta = _metric_delta(trial_metrics, pre_metrics, 'Rec')
                trial_quality_delta = _gateway_quality(trial_metrics) - _gateway_quality(pre_metrics)
                trial_accepted = (
                    trial_map_delta >= min_map_delta
                    and trial_map50_delta >= min_map50_delta
                    and trial_rec_delta >= min_rec_delta
                    and trial_quality_delta >= -accept_tol
                )
                print(
                    f"[Gateway Proxy-FT] alpha={alpha:.3f} | "
                    f"delta_mAP50-95={trial_map_delta:+.5f}, "
                    f"delta_mAP50={trial_map50_delta:+.5f}, "
                    f"delta_Rec={trial_rec_delta:+.5f}, "
                    f"delta_quality={trial_quality_delta:+.5f}, "
                    f"accepted={trial_accepted}"
                )
                candidate_score = _gateway_quality(trial_metrics)
                if trial_accepted and (
                    best_candidate is None or candidate_score > best_candidate['score']
                ):
                    best_candidate = {
                        'alpha': alpha,
                        'state': trial_state,
                        'metrics': trial_metrics,
                        'score': candidate_score,
                        'map_delta': trial_map_delta,
                        'map50_delta': trial_map50_delta,
                        'rec_delta': trial_rec_delta,
                        'quality_delta': trial_quality_delta,
                    }

            accepted = best_candidate is not None

            if accepted:
                wise_alpha = float(best_candidate['alpha'])
                candidate_metrics = best_candidate['metrics']
                map_delta = best_candidate['map_delta']
                map50_delta = best_candidate['map50_delta']
                rec_delta = best_candidate['rec_delta']
                quality_delta = best_candidate['quality_delta']
                self.global_student.load_trainable_state_dict(best_candidate['state'])
                self.gateway.global_state_dict = {
                    k: v.clone().cpu()
                    for k, v in best_candidate['state'].items()
                }
                self.gateway.proxy_ft_optimizer_state = (
                    candidate_optimizer_state if reuse_proxy_optimizer else None
                )
                self._last_gateway_eval_metrics = dict(candidate_metrics)
                print(
                    f"[Gateway Proxy-FT] Accepted candidate: "
                    f"alpha={wise_alpha:.3f}, "
                    f"delta_mAP50-95={map_delta:+.5f}, "
                    f"delta_mAP50={map50_delta:+.5f}, "
                    f"delta_Rec={rec_delta:+.5f}, "
                    f"delta_quality={quality_delta:+.5f}."
                )
            else:
                self.global_student.load_trainable_state_dict(gateway_pre_state)
                self.gateway.global_state_dict = {
                    k: v.clone().cpu()
                    for k, v in gateway_pre_state.items()
                }
                self.gateway.proxy_ft_optimizer_state = previous_optimizer_state
                self._last_gateway_eval_metrics = dict(pre_metrics)
                print(
                    f"[Gateway Proxy-FT] Rejected candidate and rolled back: "
                    f"delta_mAP50-95={map_delta:+.5f}, delta_quality={quality_delta:+.5f}."
                )
        else:
            self.gateway.proxy_ft_optimizer_state = (
                candidate_optimizer_state if reuse_proxy_optimizer else None
            )
            self._last_gateway_eval_metrics = {}
        self._last_kd_metrics = {
            'kd_active': False,
            'kd_epochs': 0,
            'kd_box': 0.0,
            'kd_kl': 0.0,
            'kd_lora': 0.0,
            'kd_scale': 0.0,
            'kd_ratio': 0.0,
            'kd_contrib': 0.0,
            'kd_total': 0.0,
            'kd_weighted': 0.0,
            'gateway_proxy_ft_active': True,
            'gateway_proxy_ft_epochs': overrides['epochs'],
            'gateway_proxy_ft_alpha': float(wise_alpha),
            'gateway_proxy_ft_accepted': bool(accepted),
            'gateway_proxy_ft_delta_map5095': float(map_delta),
            'gateway_proxy_ft_delta_map50': float(map50_delta),
            'gateway_proxy_ft_delta_rec': float(rec_delta),
            'gateway_proxy_ft_delta_quality': float(quality_delta),
        }
        print(f"[Gateway Proxy-FT] Done.")

        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return self._last_kd_metrics

    def _gateway_knowledge_distillation(self):
        """
        Gateway-side Knowledge Distillation (Tier 3) with Adaptive Dropout.
        Sau global aggregation, Gateway dùng Teacher (YOLO12l, GPU mạnh)
        để distill vào global_student trên tập proxy data (coco8.yaml).

        Adaptive KD Dropout: Nếu Prec/Rec/mAP liên tiếp giảm CONSEC_DROP_THRESHOLD vòng
        thì tự động tắt KD vĩnh viễn cho phần còn lại (thuần FL).
        """
        cfg = self.baseline_cfg
        if cfg.use_gateway_proxy_ft:
            return self._gateway_supervised_finetune()

        if not cfg.use_gateway_kd:
            self._last_kd_metrics = {
                'kd_active': False,
                'kd_epochs': 0,
                'kd_box': 0.0,
                'kd_kl': 0.0,
                'kd_lora': 0.0,
                'kd_scale': 0.0,
                'kd_ratio': 0.0,
                'kd_contrib': 0.0,
                'kd_total': 0.0,
                'kd_weighted': 0.0,
            }
            return self._last_kd_metrics

        current_r = getattr(self, 'current_round', 1)
        total_r = self.fed_cfg.GLOBAL_ROUNDS.get("2D", 60)
        phase1_end = max(1, round(total_r * self.fed_cfg.KD_PHASE1_END_FRAC))
        stop_round = max(phase1_end, round(total_r * self.fed_cfg.KD_STOP_FRAC))

        # Preserve KD coverage for very short smoke tests. Normal experiments
        # use every round in phase 1, every 2 in phase 2, then pure FL.
        if total_r < 6:
            kd_interval = 1
        elif current_r <= phase1_end:
            kd_interval = 1
        else:
            kd_interval = 2

        skip_reason = None
        if current_r > stop_round:
            skip_reason = "stopped"
        elif current_r % kd_interval != 0:
            skip_reason = f"interval_{kd_interval}"

        if skip_reason is not None:
            print(
                f"[Gateway KD] Skip round {current_r}/{total_r}: {skip_reason}. "
                f"Schedule phase1<=R{phase1_end}, stop after R{stop_round}."
            )
            self._last_kd_metrics = {
                'kd_active': False,
                'kd_epochs': 0,
                'kd_box': 0.0,
                'kd_kl': 0.0,
                'kd_lora': 0.0,
                'kd_scale': 0.0,
                'kd_ratio': 0.0,
                'kd_contrib': 0.0,
                'kd_total': 0.0,
                'kd_weighted': 0.0,
                'kd_skip_reason': skip_reason,
            }
            return self._last_kd_metrics

        kd_adaptive_dropout = getattr(self.fed_cfg, 'KD_ADAPTIVE_DROPOUT_ENABLED', False)
        consec_drop_threshold = getattr(self.fed_cfg, 'KD_ADAPTIVE_DROP_THRESHOLD', 5)
        if not hasattr(self, '_kd_disabled'):
            self._kd_disabled = False
        if not hasattr(self, '_consec_drop_count'):
            self._consec_drop_count = 0
        if not hasattr(self, '_kd_applied_rounds'):
            self._kd_applied_rounds = []
        if not hasattr(self, '_kd_assessed_rounds'):
            self._kd_assessed_rounds = set()

        # At the next scheduled KD call, assess the previous KD round against
        # the immediately preceding pure-FL round. This avoids confusing normal
        # round-to-round noise with harm caused by KD itself.
        history = getattr(self, '_round_metrics_history', [])
        for kd_round in self._kd_applied_rounds:
            if kd_round in self._kd_assessed_rounds or kd_round < 2 or len(history) < kd_round:
                continue
            before = history[kd_round - 2]
            after = history[kd_round - 1]

            def _quality(metrics):
                return (
                    float(metrics.get('mAP50-95', 0.0))
                    + 0.25 * float(metrics.get('mAP50', 0.0))
                    + 0.25 * float(metrics.get('Rec', 0.0))
                )

            quality_delta = _quality(after) - _quality(before)
            self._kd_assessed_rounds.add(kd_round)
            if quality_delta < 0.0:
                self._consec_drop_count += 1
                print(
                    f"[Gateway KD] Harm check R{kd_round}: Δquality={quality_delta:+.5f} "
                    f"({self._consec_drop_count}/{consec_drop_threshold})."
                )
                if kd_adaptive_dropout and self._consec_drop_count >= consec_drop_threshold:
                    self._kd_disabled = True
            else:
                self._consec_drop_count = 0
                print(f"[Gateway KD] Benefit check R{kd_round}: Δquality={quality_delta:+.5f}.")

        if kd_adaptive_dropout and self._kd_disabled:
            print("[Gateway KD] Skipping KD after repeated measured harm; continuing with pure FL.")
            self._last_kd_metrics = {
                'kd_active': False, 'kd_epochs': 0,
                'kd_box': 0.0, 'kd_kl': 0.0,
                'kd_lora': 0.0, 'kd_scale': 0.0,
                'kd_ratio': 0.0, 'kd_contrib': 0.0,
                'kd_total': 0.0, 'kd_weighted': 0.0,
                'kd_skip_reason': 'adaptive_harm',
            }
            return self._last_kd_metrics

        from detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer

        proxy_yaml = self._build_gateway_proxy_yaml()

        overrides = {
            'model': self.student_ckpt,
            'data': proxy_yaml,
            'epochs': self.fed_cfg.KD_EPOCHS,
            'batch': self.fed_cfg.KD_BATCH_SIZE,
            'device': self.device,
            'project': 'runs/gateway_kd',
            'name': 'global_kd',
            'exist_ok': True,
            'verbose': False,
            'save': False,
            'val': False,
            'plots': False,
            'workers': self.fed_cfg.KD_WORKERS,
            'close_mosaic': 0,
            'mosaic': 0.0,
            'augment': False,
            # [FIX] Tắt hoàn toàn augmentation — cùng lý do với Proxy FT.
            'fliplr': 0.0,
            'scale': 0.0,
            'translate': 0.0,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'erasing': 0.0,
            'optimizer': getattr(self.fed_cfg, 'KD_OPTIMIZER', 'AdamW'),
            'amp': self.fed_cfg.KD_AMP,
            'lr0': self.fed_cfg.KD_LR,
            # [FIX] lrf=1.0: giữ LR flat (không cosine decay). Cùng lý do với Proxy FT.
            'lrf': getattr(self.fed_cfg, 'KD_LRF', 1.0),
            # [FIX] Warmup 10% để AdamW tự tích lũy đà, tránh giật LR ở Round 1
            'warmup_epochs': getattr(self.fed_cfg, 'KD_WARMUP_EPOCHS', 0.1),
            'warmup_bias_lr': self.fed_cfg.KD_LR,
        }
        trainer = KDDetectionTrainer(
            overrides=overrides,
            cached_optimizer_state=getattr(self.gateway, 'kd_optimizer_state', None)
        )
        trainer.head_lr_multiplier = getattr(self.fed_cfg, 'KD_HEAD_LR_MULT', 4.0)
        trainer.lora_lr_multiplier = getattr(self.fed_cfg, 'KD_LORA_LR_MULT', 1.0)
        trainer.kd_temperature = getattr(self.fed_cfg, 'KD_TEMPERATURE', 4.0)
        trainer.student_wrapper = self.global_student
        trainer.logit_kd_only = cfg.logit_kd_only
        trainer.logit_box_kd_only = cfg.logit_box_kd_only
        trainer.logit_proj_kd_only = cfg.logit_proj_kd_only
        
        # [CÂN BẰNG LOSS] stu_lambda được đọc từ config (mặc định 0.5 = cân bằng Supervised/KD)
        trainer.stu_lambda = getattr(self.fed_cfg, 'KD_STU_LAMBDA', 0.50)
        trainer.kd_balance_by_supervised = getattr(
            self.fed_cfg, 'KD_BALANCE_BY_SUPERVISED', True
        )
        trainer.kd_balance_scale_min = getattr(
            self.fed_cfg, 'KD_BALANCE_SCALE_MIN', 0.001
        )
        trainer.kd_balance_scale_max = getattr(
            self.fed_cfg, 'KD_BALANCE_SCALE_MAX', 4.0
        )
        trainer.kd_cls_weight = getattr(self.fed_cfg, 'KD_CLS_WEIGHT', 0.45)
        trainer.kd_box_weight = getattr(self.fed_cfg, 'KD_BOX_WEIGHT', 0.35)
        trainer.kd_proj_weight = getattr(self.fed_cfg, 'KD_PROJ_WEIGHT', 0.20)
        if cfg.logit_proj_kd_only and trainer.kd_proj_weight <= 0.0:
            trainer.kd_proj_weight = 0.10
        trainer.kd_conf_threshold = getattr(self.fed_cfg, 'KD_CONF_THRESHOLD', 0.10)
        trainer.kd_conf_gamma = getattr(self.fed_cfg, 'KD_CONF_GAMMA', 2.0)
        trainer.kd_dfl_weight = getattr(self.fed_cfg, 'KD_DFL_WEIGHT', 1.0)
        trainer.kd_ciou_weight = getattr(self.fed_cfg, 'KD_CIOU_WEIGHT', 0.5)
        trainer.kd_proj_mode = getattr(self.fed_cfg, 'KD_PROJ_MODE', 'lora_spatial_proj')
        trainer.kd_proj_anchor_match = getattr(self.fed_cfg, 'KD_PROJ_ANCHOR_MATCH', True)
        
        kd_lambda = getattr(
            self.fed_cfg,
            'KD_LAMBDA',
            getattr(self.fed_cfg, 'KD_LAMBDA_START', 0.30),
        )
        trainer.kd_lambda = kd_lambda
        print(
            f"[Gateway KD] Round {current_r}/{total_r} | interval={kd_interval} | "
            f"target_kd_ratio={kd_lambda:.3f} | stu_lambda={trainer.stu_lambda:.2f} | "
            f"balanced={trainer.kd_balance_by_supervised} | "
            f"proj_mode={trainer.kd_proj_mode} | anchor_match={trainer.kd_proj_anchor_match}"
        )
        
        trainer.set_teacher(self.teacher.yolo.model)
        
        # [FIX] Đảm bảo model không bị dính cờ inference tensor từ hàm evaluate vòng trước
        self.global_student.strip_inference_tensors()
        
        # [CRITICAL FIX] Load aggregated weights từ các auvs (đang nằm trong gateway)
        # vào global_student TRƯỚC khi chạy Distillation, nếu không KD sẽ train trên tàn dư cũ!
        self.global_student.load_trainable_state_dict(self.gateway.global_state_dict)
        
        # [MOD] Khi lên gateway thì mở full head để KD
        pre_kd_state = {
            k: v.clone().cpu()
            for k, v in self.gateway.global_state_dict.items()
        }
        previous_kd_optimizer_state = getattr(self.gateway, 'kd_optimizer_state', None)

        head_idx = len(self.global_student.yolo.model.model) - 1
        head_prefix = f'model.{head_idx}.'
        for name, param in self.global_student.yolo.model.named_parameters():
            if head_prefix in name:
                param.requires_grad_(True)
                
        trainer._fl_injected_model = self.global_student.yolo.model
        trainer.model = self.global_student.yolo.model
        print(f"[Gateway KD] Distilling global model with Teacher on proxy data...")
        try:
            trainer.train()
            candidate_optimizer_state = trainer.get_named_optimizer_state()
        finally:
            if hasattr(trainer, 'cleanup_kd_hooks'):
                trainer.cleanup_kd_hooks()

        kd_metrics = trainer.get_kd_summary() if hasattr(trainer, 'get_kd_summary') else {
            'kd_active': True,
            'kd_epochs': 0,
            'kd_box': 0.0,
            'kd_kl': 0.0,
            'kd_lora': 0.0,
            'kd_scale': 0.0,
            'kd_ratio': 0.0,
            'kd_contrib': 0.0,
            'kd_total': 0.0,
            'kd_weighted': 0.0,
        }
        self._last_kd_metrics = kd_metrics
        self._kd_applied_rounds.append(current_r)

        if kd_metrics.get('kd_active', False):
            print(
                f"[Gateway KD] Summary | "
                f"Box={kd_metrics.get('kd_box', 0.0):.4f}, "
                f"KL={kd_metrics.get('kd_kl', 0.0):.4f}, "
                f"LoRA_Proj={kd_metrics.get('kd_lora', 0.0):.4f}, "
                f"KD/Sup={kd_metrics.get('kd_ratio', 0.0):.3f}, "
                f"KD Contrib={kd_metrics.get('kd_contrib', 0.0):.4f}, "
                f"Total={kd_metrics.get('kd_total', 0.0):.4f}"
            )

        # [ASYMMETRIC DOWNLINK] Gateway broadcasts FULL head + LoRA to AUVs/Relays.
        # Uplink (AUV→Gateway) only sends partial head to save battery.
        # Downlink (Gateway→AUV) sends full head since Gateway has power budget.
        post_kd_state = self.global_student.trainable_state_dict(downlink=True)

        def _gateway_quality(metrics: dict) -> float:
            if not metrics:
                return 0.0
            return (
                float(metrics.get('mAP50-95', 0.0))
                + 0.25 * float(metrics.get('mAP50', 0.0))
                + 0.25 * float(metrics.get('Rec', 0.0))
            )

        def _metric_delta(after: dict, before: dict, key: str) -> float:
            return float(after.get(key, 0.0)) - float(before.get(key, 0.0))

        def _blend_kd_state(alpha: float) -> dict:
            if alpha >= 1.0:
                return {k: v.clone().cpu() for k, v in post_kd_state.items()}
            blended = {}
            for k, post_v in post_kd_state.items():
                pre_v = pre_kd_state.get(k)
                if pre_v is not None and torch.is_tensor(post_v) and torch.is_tensor(pre_v):
                    if torch.is_floating_point(post_v) and torch.is_floating_point(pre_v):
                        blended[k] = (
                            alpha * post_v.detach().cpu()
                            + (1.0 - alpha) * pre_v.detach().cpu()
                        )
                    else:
                        blended[k] = post_v.detach().cpu().clone()
                else:
                    blended[k] = post_v.detach().cpu().clone()
            return blended

        gate_enabled = bool(getattr(self.fed_cfg, 'KD_ACCEPTANCE_GATE', True))
        pre_metrics = getattr(self, '_last_pre_gateway_metrics', {})
        accepted = True
        best_alpha = 1.0
        best_state = {k: v.clone().cpu() for k, v in post_kd_state.items()}
        best_metrics = {}
        map_delta = 0.0
        map50_delta = 0.0
        rec_delta = 0.0
        quality_delta = 0.0

        if gate_enabled and pre_metrics:
            alpha_candidates = tuple(
                float(a)
                for a in getattr(self.fed_cfg, 'KD_WISE_ALPHA_CANDIDATES', (1.0,))
            )
            min_map_delta = float(getattr(self.fed_cfg, 'KD_MIN_MAP5095_DELTA', 0.0))
            min_map50_delta = float(getattr(self.fed_cfg, 'KD_MIN_MAP50_DELTA', 0.0))
            min_rec_delta = float(getattr(self.fed_cfg, 'KD_MIN_REC_DELTA', 0.0))
            accept_tol = float(getattr(self.fed_cfg, 'KD_ACCEPT_TOL', 0.0))
            best_candidate = None

            for alpha in alpha_candidates:
                trial_state = _blend_kd_state(alpha)
                self.gateway.global_state_dict = {
                    k: v.clone().cpu()
                    for k, v in trial_state.items()
                }
                trial_metrics = self.evaluate()
                trial_map_delta = _metric_delta(trial_metrics, pre_metrics, 'mAP50-95')
                trial_map50_delta = _metric_delta(trial_metrics, pre_metrics, 'mAP50')
                trial_rec_delta = _metric_delta(trial_metrics, pre_metrics, 'Rec')
                trial_quality_delta = _gateway_quality(trial_metrics) - _gateway_quality(pre_metrics)
                trial_accepted = (
                    trial_map_delta >= min_map_delta
                    and trial_map50_delta >= min_map50_delta
                    and trial_rec_delta >= min_rec_delta
                    and trial_quality_delta >= -accept_tol
                )
                print(
                    f"[Gateway KD Gate] alpha={alpha:.2f} | "
                    f"delta_mAP50-95={trial_map_delta:+.5f}, "
                    f"delta_mAP50={trial_map50_delta:+.5f}, "
                    f"delta_Rec={trial_rec_delta:+.5f}, "
                    f"delta_quality={trial_quality_delta:+.5f}, "
                    f"accepted={trial_accepted}"
                )
                candidate_score = _gateway_quality(trial_metrics)
                if trial_accepted and (
                    best_candidate is None or candidate_score > best_candidate['score']
                ):
                    best_candidate = {
                        'alpha': alpha,
                        'state': trial_state,
                        'metrics': trial_metrics,
                        'score': candidate_score,
                        'map_delta': trial_map_delta,
                        'map50_delta': trial_map50_delta,
                        'rec_delta': trial_rec_delta,
                        'quality_delta': trial_quality_delta,
                    }

            accepted = best_candidate is not None
            if accepted:
                best_alpha = best_candidate['alpha']
                best_state = best_candidate['state']
                best_metrics = best_candidate['metrics']
                map_delta = best_candidate['map_delta']
                map50_delta = best_candidate['map50_delta']
                rec_delta = best_candidate['rec_delta']
                quality_delta = best_candidate['quality_delta']
                self.global_student.load_trainable_state_dict(best_state)
                self.gateway.global_state_dict = {
                    k: v.clone().cpu()
                    for k, v in best_state.items()
                }
                self.gateway.kd_optimizer_state = (
                    candidate_optimizer_state if best_alpha >= 0.999 else None
                )
                self._last_gateway_eval_metrics = best_metrics
                print(
                    f"[Gateway KD Gate] Accepted alpha={best_alpha:.2f}: "
                    f"delta_mAP50-95={map_delta:+.5f}, "
                    f"delta_mAP50={map50_delta:+.5f}, "
                    f"delta_quality={quality_delta:+.5f}."
                )
            else:
                self.global_student.load_trainable_state_dict(pre_kd_state)
                self.gateway.global_state_dict = {
                    k: v.clone().cpu()
                    for k, v in pre_kd_state.items()
                }
                self.gateway.kd_optimizer_state = previous_kd_optimizer_state
                self._last_gateway_eval_metrics = pre_metrics
                print("[Gateway KD Gate] Rejected KD candidate; restored pre-KD state.")
        else:
            self.gateway.kd_optimizer_state = candidate_optimizer_state
            self.gateway.global_state_dict = {
                k: v.clone().cpu()
                for k, v in post_kd_state.items()
            }

        kd_metrics.update({
            'gateway_kd_gate_active': bool(gate_enabled and pre_metrics),
            'gateway_kd_accepted': bool(accepted),
            'gateway_kd_alpha': float(best_alpha),
            'gateway_kd_delta_map5095': float(map_delta),
            'gateway_kd_delta_map50': float(map50_delta),
            'gateway_kd_delta_rec': float(rec_delta),
            'gateway_kd_delta_quality': float(quality_delta),
        })
        self._last_kd_metrics = kd_metrics
        print(f"[Gateway KD] Done.")
        
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        return self._last_kd_metrics

    def evaluate(self) -> Dict[str, float]:
        # [FIX] Ultralytics trainer.train() leaves inference-mode tensors on the model
        # (from EMA / post-training hooks). Strip them before calling load_state_dict,
        # otherwise PyTorch raises "Inplace update to inference tensor outside InferenceMode".
        self.global_student.strip_inference_tensors()
        self.global_student.load_trainable_state_dict(self.gateway.global_state_dict)
        res = evaluate_od(self.global_student, self.test_yaml, self.device)
            
        gc.collect()
        torch.cuda.empty_cache()
        return res
