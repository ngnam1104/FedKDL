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
from tasks.detection_2d.models.yolo_wrapper import StudentModel, TeacherModel
from tasks.detection_2d.trainer import local_sgd_od, evaluate_od, evaluate_od_on_auv_train
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload


class AUVWorker2D(BaseWorker):
    def __init__(self, auv_id, auv_yaml, battery_init):
        super().__init__(auv_id, battery_init)
        self.auv_yaml = auv_yaml
        # Cache optimizer state (exp_avg / exp_avg_sq) giữa các FL round.
        # None = chưa có (round đầu tiên, optimizer khởi đầu lạnh).
        self._optimizer_state: dict = None

        with open(self.auv_yaml, 'r') as f:
            c_cfg = yaml.safe_load(f)
        with open(c_cfg['train'], 'r') as f:
            self.n_samples = sum(1 for _ in f)

    def train_and_get_payload(
        self, global_state, epochs: int, lr: float, device: str,
        baseline: str, global_weights: dict = None, fedprox_mu_override: float = 0.0,
        nc: int = 4,
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

        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        if baseline in classic_baselines:
            full_param = True
            use_lora = False
            use_int8 = False
        else:
            full_param = 'full_param' in baseline
            use_lora = 'nolora' not in baseline
            use_int8 = 'noint8' not in baseline
            
        fedprox_mu = 0.01 if 'fedprox' in baseline else 0.0
        # Nếu KD bị Adaptive Dropout tắt, fedprox_mu_override sẽ được truyền xuống từ Simulator
        fedprox_mu = max(fedprox_mu, fedprox_mu_override)
        from config.settings import fed_cfg
        rank = 4 if 'r4' in baseline else fed_cfg.LORA_RANK

        local_student = StudentModel("yolo12n.pt", rank=rank, nc=nc, full_param=full_param, use_lora=use_lora)
        local_student.load_trainable_state_dict(global_state)

        # Cấp phát Teacher cục bộ nếu chạy thuật toán FedKD (Local KD)
        local_teacher = None
        if baseline == 'fedkd':
            if not hasattr(self, 'local_teacher'):
                from tasks.detection_2d.models.yolo_wrapper import TeacherModel
                print(f"[Simulator2D] Khởi tạo Local Teacher (YOLO12l) dùng chung cho thuật toán FedKD...")
                # Teacher load sẵn weights pretrained
                self.local_teacher = TeacherModel("yolo12l_pretrained.pt")
                self.local_teacher.yolo.to(device)
            local_teacher = self.local_teacher

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
            cached_optimizer_state=self._optimizer_state,  # None = lạnh, round sau sẽ warm
        )
        # Cập nhật optimizer state để dùng lại round tiếp theo
        if new_opt_state is not None:
            self._optimizer_state = new_opt_state

        # [TỐI ƯU HÓA] Bỏ qua đánh giá local model trên tập train của auv
        # Việc này tiết kiệm 30% tổng thời gian huấn luyện mà không ảnh hưởng kết quả Global
        local_metrics = {}
        print(f"[AUV {self.auv_id}] Local train metrics skipped to save time.")

        if delta_norm < fed_cfg.DELTA_SKIP:
            print(f"[AUV {self.auv_id}] 💤 Lazy Filter Activated (delta={delta_norm:.4f} < {fed_cfg.DELTA_SKIP}). Node is resting (No TX).")
            payload_bytes = None
            payload_kb = 0.0
        elif use_int8:
            payload_bytes, payload_kb = pack_payload(new_state)
            print(f"[AUV {self.auv_id}] Payload: {payload_kb:.1f} KB INT8 "
                  f"(target ≤ {fed_cfg.TARGET_PAYLOAD_KB:.0f} KB)")
        else:
            # Fake packing for simulation (Float32 payload)
            payload_bytes = new_state
            # Calculate bytes based on float32 (4 bytes per param)
            total_params = sum(t.numel() for t in new_state.values())
            payload_kb = (total_params * 4) / 1024.0
            print(f"[AUV {self.auv_id}] Payload: {payload_kb:.1f} KB Float32")

        del local_student
        gc.collect()
        torch.cuda.empty_cache()

        return payload_bytes, payload_kb, delta_norm, train_loss, local_metrics



class RelayNode2D(BaseRelayNode):
    def aggregate_intra_cluster(self, global_state_dict, payloads, auv_n_samples, use_kd_lora_int8=True):
        import torch
        from tasks.detection_2d.knowledge_compression.int8_quantization import unpack_payload
        
        c_updates = []
        valid_sids = []
        for sid in self.cluster_members:
            if sid in payloads:
                if use_kd_lora_int8:
                    state = unpack_payload(payloads[sid], global_state_dict)
                else:
                    state = payloads[sid]
                c_updates.append(state)
                valid_sids.append(sid)

        if not c_updates:
            import copy
            self.intra_state_dict = copy.deepcopy(global_state_dict)
            self.final_state_dict = copy.deepcopy(global_state_dict)
            return

        self.intra_state_dict = {}
        total_samples = sum(auv_n_samples.get(sid, 0) for sid in valid_sids)
        if total_samples == 0:
            total_samples = 1

        for k in c_updates[0]:
            original_dtype = c_updates[0][k].dtype
            weighted_sum = torch.zeros_like(c_updates[0][k].float())
            for i, sid in enumerate(valid_sids):
                weight = auv_n_samples.get(sid, 0) / total_samples
                weighted_sum += c_updates[i][k].float() * weight
            self.intra_state_dict[k] = weighted_sum.to(original_dtype)
            
        import copy
        self.final_state_dict = copy.deepcopy(self.intra_state_dict)


class Simulator2D(BaseSimulator):
    def __init__(
        self,
        topo_path: str,
        data_path: str,
        baseline: str,
        test_yaml: str = "datasets/URPC2020.yaml",
        student_ckpt: str = "yolo12n.pt",
        teacher_ckpt: str = "yolo12l.pt",
        device: str = "cpu",
    ):
        super().__init__(topo_path=topo_path, baseline=baseline, device=device)
        self.test_yaml = test_yaml
        self.task_key = "2D"
        
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

                all_images = []
                for ext in ('*.jpg', '*.png', '*.JPG', '*.JPEG', '*.jpeg'):
                    all_images.extend([str(p.resolve()) for p in img_dir.glob(f'**/{ext}')])
                
                if not all_images:
                    raise FileNotFoundError(f"CRITICAL: Tìm thấy thư mục {img_dir} nhưng KHÔNG CÓ ẢNH NÀO bên trong! (Hỗ trợ: jpg, png, jpeg)")
                    
                all_images.sort()
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
        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        if self.baseline in classic_baselines:
            full_param = True
            use_lora = False
        else:
            full_param = 'full_param' in self.baseline
            use_lora = 'nolora' not in self.baseline
        rank = 4 if 'r4' in self.baseline else self.fed_cfg.LORA_RANK
        
        self.teacher = TeacherModel(teacher_ckpt)
        self.teacher.yolo.to(self.device)
        self.global_student = StudentModel(student_ckpt, rank=rank, nc=nc, full_param=full_param, use_lora=use_lora)
        
        self.gateway = BaseGateway(initial_state=self.global_student.trainable_state_dict())
        self._last_kd_metrics = {}
        
        self.relay_model_bits = sum(t.numel() for t in self.gateway.global_state_dict.values()) * 32
        
        self._init_network()

    def _init_network(self):
        import os
        import yaml
        import numpy as np
        
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
        nc = getattr(self.global_student.yolo.model.yaml, 'nc', 80) if hasattr(self.global_student.yolo.model, 'yaml') else 80
        if isinstance(nc, dict) and 'nc' in nc: nc = nc['nc'] # fallback cho kiểu dict
        elif not isinstance(nc, int): nc = 80

        # Nếu là FedKDL thì bật EMD Clustering
        if 'fedkdl' in self.baseline:
            print(f"\n[Simulator2D] Khởi tạo Knowledge-Aware Association (EMD β=0.2)...")
            self.auv_label_hists = np.zeros((N, nc), dtype=np.float32)
            self.auv_label_counts = np.zeros((N, nc), dtype=np.float32)
            for s_id in range(N):
                if s_id in getattr(self, 'auv_yamls', {}):
                    n_hist, c_hist = get_label_histogram(self.auv_yamls[s_id], nc)
                    self.auv_label_hists[s_id] = n_hist
                    self.auv_label_counts[s_id] = c_hist
            
            # Tính Relay Histogram từ các cụm vật lý ban đầu
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

            # Thực thi Knowledge-Aware Association
            class DummyTopo:
                def __init__(self, n, m): self.N = n; self.M = m
            
            from tasks.detection_2d.knowledge_compression.knowledge_association import knowledge_aware_association
            new_association = knowledge_aware_association(
                topology=DummyTopo(N, M),
                G=self.G,
                auv_label_hists=self.auv_label_hists,
                relay_label_hists=self.relay_label_hists,
                beta=0.0  # TẮT EMD theo yêu cầu: 100% Khoảng cách vật lý
            )
            
            # Đếm số lượng thay đổi
            changed = 0
            changes_log = []
            for s, new_f in new_association.items():
                old_f = self.association.get(s, -1)
                if old_f != new_f:
                    changed += 1
                    changes_log.append(f"    - AUV {s}: Relay {old_f} -> Relay {new_f}")
            print(f"[Simulator2D] EMD Clustering hoàn tất. {changed}/{N} AUVs đã chuyển cụm (Relay) để tối ưu EMD.")
            if changed > 0:
                print("  Chi tiết thay đổi:")
                for log in changes_log:
                    print(log)
            
            # Cập nhật state
            self.association = new_association
            # Cập nhật lại clusters
            self.clusters = {m: [] for m in range(M)}
            for s, f in self.association.items():
                self.clusters[f].append(s)

        # 3. Tiến hành cấp phát Worker/Node như bình thường
        for s_id in range(self.net_cfg.N_AUVS):
            if s_id in getattr(self, 'auv_yamls', {}):
                if s_id in self.association:
                    self.auvs[s_id] = AUVWorker2D(
                        auv_id=s_id,
                        auv_yaml=self.auv_yamls[s_id],
                        battery_init=self.en_cfg.E_INIT,
                    )
                else:
                    print(f"\n[{'='*40}]\n[AUV {s_id}] BỎ QUA VÌ KHÔNG THỎA MÃN KHOẢNG CÁCH (Out of Range)\n[{'='*40}]\n")

        for m, members in self.clusters.items():
            if len(members) > 0:
                self.relays[m] = RelayNode2D(
                    relay_id=m,
                    cluster_members=members,
                )

    def get_flop_multiplier(self) -> float:
        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        if self.baseline == 'fedkd':
            # FedKD chạy Teacher (YOLOv12l) Forward pass + Student (YOLOv12n) Full pass.
            # Tỷ lệ FLOPs của YOLOv12l so với YOLOv12n là ~30 lần. Cộng thêm 3 lần cho Student.
            return 33.0 
        elif 'full_param' in self.baseline or self.baseline in classic_baselines:
            # Full param update (Forward + Backward qua tất cả tham số)
            return 3.0
        # Mặc định (LoRA)
        return self.fed_cfg.FLOP_MULTIPLIER[self.task_key]

    def _process_auv(self, s_id: int) -> Tuple[int, Any, float, int, float, float, dict]:
        auv = self.auvs[s_id]

        if s_id == 0 and getattr(self, '_fedprox_mu_override', 0.0) > 0.0:
            print(f"    [!] Adaptive Dropout Active: AUVs are training with FedProx (mu={self._fedprox_mu_override})")

        payload, payload_kb, delta_norm, train_loss, local_metrics = auv.train_and_get_payload(
            global_state=self.gateway.global_state_dict,
            epochs=self.fed_cfg.LOCAL_EPOCHS,
            lr=getattr(self, 'current_lr', self.fed_cfg.LOCAL_LR),
            device=self.device,
            baseline=self.baseline,
            fedprox_mu_override=getattr(self, '_fedprox_mu_override', 0.0),
            # global_weights cần thiết cho FedProx proximal term
            global_weights=self.gateway.global_state_dict if (
                'fedprox' in self.baseline or getattr(self, '_fedprox_mu_override', 0.0) > 0.0
            ) else None,
        )

        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        use_int8 = 'noint8' not in self.baseline and self.baseline not in classic_baselines
        
        from physics_models.energy import e_tx, e_comp_dynamic
        if payload is not None:
            if not use_int8:
                S_bits = payload_kb * 1024 * 8
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

                e_comp_cost = e_comp_dynamic(
                    n_samples=auv.n_samples,
                    n_local_epochs=self.fed_cfg.LOCAL_EPOCHS,
                    flops_per_sample=self.fed_cfg.MODEL_FLOPS_PER_SAMPLE[self.task_key],
                    flop_multiplier=self.get_flop_multiplier(),
                    epsilon_op=self.en_cfg.EPSILON_OP[self.task_key]
                )

                if auv.battery >= (e_tx_cost + e_comp_cost):
                    return s_id, payload, train_loss, auv.n_samples, e_tx_cost, e_comp_cost, local_metrics
                else:
                    auv.alive = False

        return s_id, None, 0.0, 0, 0.0, 0.0, {}


    def _aggregate_intra_relay(self, m: int, relay, payloads, auv_n_samples) -> float:
        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        use_int8 = 'noint8' not in self.baseline and self.baseline not in classic_baselines
        relay.aggregate_intra_cluster(
            global_state_dict=self.gateway.global_state_dict,
            payloads=payloads,
            auv_n_samples=auv_n_samples,
            use_kd_lora_int8=use_int8
        )
        return 0.0

    def _compute_payload_bits(self, payloads: Dict) -> float:
        if not payloads:
            return self.relay_model_bits
        if 'noint8' in self.baseline:
            # payloads là state_dicts Float32
            return np.mean([sum(t.numel() for t in p.values()) * 32 for p in payloads.values()])
        return np.mean([len(p) * 8 for p in payloads.values()])

    def _compute_relay_model_bits(self) -> float:
        return self.relay_model_bits

    def _gateway_knowledge_distillation(self):
        """
        Gateway-side Knowledge Distillation (Tier 3) with Adaptive Dropout.
        Sau global aggregation, Gateway dùng Teacher (YOLO12l, GPU mạnh)
        để distill vào global_student trên tập proxy data (coco8.yaml).

        Adaptive KD Dropout: Nếu Prec/Rec/mAP liên tiếp giảm CONSEC_DROP_THRESHOLD vòng
        thì tự động tắt KD vĩnh viễn cho phần còn lại (thuần FL).
        """
        classic_baselines = ['fedavg', 'fedprox', 'centralized', 'hfl_selective', 'hfl_nearest', 'hfl_nocoop', 'fedkd']
        if 'nokd' in self.baseline or self.baseline in classic_baselines:
            self._last_kd_metrics = {
                'kd_active': False,
                'kd_epochs': 0,
                'kd_kl': 0.0,
                'kd_hidden': 0.0,
                'kd_attn': 0.0,
                'kd_weighted': 0.0,
            }
            return self._last_kd_metrics

        # --- Adaptive KD Dropout Gate ---
        # Số vòng liên tiếp metrics giảm để kích hoạt ngắt KD
        CONSEC_DROP_THRESHOLD = getattr(self, '_kd_drop_threshold', 5)

        # Khởi tạo trạng thái tracking nếu chưa có
        if not hasattr(self, '_kd_disabled'):
            self._kd_disabled = False
        if not hasattr(self, '_metric_history'):
            self._metric_history = []   # list of (mAP50, Prec, Rec)
        if not hasattr(self, '_consec_drop_count'):
            self._consec_drop_count = 0

        # Nếu KD đã bị tắt trước đó thì bỏ qua luôn
        if self._kd_disabled:
            print(f"[Gateway KD] 🏠 Skipping KD (Adaptive Dropout active — Training strictly with FedProx FL).")
            self._last_kd_metrics = {
                'kd_active': False, 'kd_epochs': 0,
                'kd_kl': 0.0, 'kd_hidden': 0.0,
                'kd_attn': 0.0, 'kd_weighted': 0.0,
            }
            return self._last_kd_metrics

        # Lấy metrics vòng vừa rồi từ lịch sử round logs
        # [MODIFIED] Bỏ qua Adaptive Dropout, ép KD chạy 100% (cả 60 vòng)
        self._kd_disabled = False
        if hasattr(self, '_round_metrics_history') and len(self._round_metrics_history) >= 2:
            prev = self._round_metrics_history[-2]
            curr = self._round_metrics_history[-1]
            prev_score = (prev.get('mAP50', 0) + prev.get('Prec', 0) + prev.get('Rec', 0)) / 3.0
            curr_score = (curr.get('mAP50', 0) + curr.get('Prec', 0) + curr.get('Rec', 0)) / 3.0
            if curr_score < prev_score:
                self._consec_drop_count += 1
                print(f"[Gateway KD] ⚠️  Metrics drop detected ({self._consec_drop_count}/∞) - KD VẪN ĐƯỢC GIỮ NGUYÊN!")
            else:
                self._consec_drop_count = 0

        import os
        from tasks.detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer

        proxy_yaml = getattr(self, 'test_yaml', "coco8.yaml")  # Dùng luôn test_yaml (URPC) làm proxy thay vì tải coco8
        
        # Tạo proxy yaml với absolute path qua file txt để tránh lỗi thư mục Linux của YOLO
        if proxy_yaml != "coco8.yaml" and os.path.exists(proxy_yaml):
            import yaml
            from pathlib import Path
            with open(proxy_yaml, 'r') as f:
                p_cfg = yaml.safe_load(f)
            if 'path' in p_cfg and 'val' in p_cfg:
                # Convert relative val path to absolute before popping 'path'
                dataset_dir = Path(proxy_yaml).parent
                base_dir = dataset_dir / p_cfg['path']
                
                # YOLO có thể dùng chuỗi đường dẫn trực tiếp, ta chuyển nó thành absolute
                if isinstance(p_cfg['val'], str):
                    p_cfg['val'] = str((base_dir / p_cfg['val']).resolve())
                elif isinstance(p_cfg['val'], list):
                    p_cfg['val'] = [str((base_dir / v).resolve()) for v in p_cfg['val']]
            
            p_cfg.pop('path', None)  # Xoá path đi để YOLO bắt buộc dùng đường dẫn tuyệt đối ở dưới
            p_cfg['train'] = getattr(self, 'proxy_kd_txt', '')
            # p_cfg['val'] được giữ nguyên (đã chuyển thành absolute path) thay vì gán bằng train
            
            proxy_yaml_abs = "datasets/proxy_kd_data.yaml"
            with open(proxy_yaml_abs, 'w') as f:
                yaml.safe_dump(p_cfg, f)
            proxy_yaml = str(Path(proxy_yaml_abs).absolute())

        overrides = {
            'model': "yolo12n.pt",
            'data': proxy_yaml,
            'epochs': 2,  # Giữ ở mức 2 epoch theo yêu cầu để tiết kiệm thời gian
            'batch': 8,
            'device': self.device,
            'project': 'runs/gateway_kd',
            'name': 'global_kd',
            'exist_ok': True,
            'verbose': False,
            'save': False,
            'val': False,
            'plots': False,
            'workers': 0,
            'close_mosaic': 0,
            'optimizer': 'AdamW',
            
            # Tăng LR lên 5e-4 để Gradient của SP Loss đủ lớn tác động lên Feature Map
            'lr0': 5e-4,
            
            'warmup_epochs': 0,   # [CRITICAL FIX] Tắt hoàn toàn warmup! Nếu để mặc định warmup_epochs=3 > epochs=1
                                  # thì TOÀN BỘ epoch là warmup phase → warmup_bias_lr=0.1 áp lên bias params
                                  # = gấp 1000 lần lr0=1e-4 → overwrite hoàn toàn detection head bias → mAP tụt!
            'warmup_bias_lr': 0.0, # [CRITICAL FIX] Đảm bảo bias params không bị warmup với lr cao
        }
        trainer = KDDetectionTrainer(overrides=overrides)
        trainer.student_wrapper = self.global_student
        
        # [FIX v3] kd_lambda cố định 0.1 — KD chỉ là regularizer nhẹ, không được overwrite FL.
        # 0.5 quá lớn: kết hợp với Attn Loss×50 gây dao động cực đoan vòng 1→2.
        trainer.kd_lambda = 0.1
        
        trainer.set_teacher(self.teacher.yolo.model)
        
        # [FIX] Đảm bảo model không bị dính cờ inference tensor từ hàm evaluate vòng trước
        self.global_student.strip_inference_tensors()
        
        # [CRITICAL FIX] Load aggregated weights từ các auvs (đang nằm trong gateway)
        # vào global_student TRƯỚC khi chạy Distillation, nếu không KD sẽ train trên tàn dư cũ!
        self.global_student.load_trainable_state_dict(self.gateway.global_state_dict)
        
        trainer.model = self.global_student.yolo.model
        print(f"[Gateway KD] Distilling global model with Teacher on proxy data...")
        trainer.train()

        kd_metrics = trainer.get_kd_summary() if hasattr(trainer, 'get_kd_summary') else {
            'kd_active': True,
            'kd_epochs': 0,
            'kd_kl': 0.0,
            'kd_hidden': 0.0,
            'kd_attn': 0.0,
            'kd_weighted': 0.0,
        }
        self._last_kd_metrics = kd_metrics

        if kd_metrics.get('kd_active', False):
            print(
                f"[Gateway KD] Summary | "
                f"KL={kd_metrics.get('kd_kl', 0.0):.4f}, "
                f"Hidden={kd_metrics.get('kd_hidden', 0.0):.4f}, "
                f"Attn={kd_metrics.get('kd_attn', 0.0):.4f}, "
                f"Weighted={kd_metrics.get('kd_weighted', 0.0):.4f}"
            )

        # Cập nhật global state dict sau KD
        self.gateway.global_state_dict = self.global_student.trainable_state_dict()
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

