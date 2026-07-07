"""
yolo_wrapper.py
Wrapper cho Student (yolo12n + LoRA) và Teacher (YOLO12l, frozen).
Sử dụng fl_core/models/lora.py để inject LoRA.
"""
import torch
from ultralytics import YOLO
from detection_2d.models.lora import inject_lora
import detection_2d.compat  # noqa: F401  register tasks.detection_2d.* shims for old checkpoints


class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    """Bản vá lỗi Pickle cho BatchNorm2d khi bị đóng băng trong PEFT (LoRA)."""
    def train(self, mode=False):
        return super().train(False)


class StudentModel:
    """
    yolo12n + LoRA injection cho Federated Learning.
    Chỉ {lora_A, lora_B, detect head} là trainable và được truyền qua mạng.
    """

    def __init__(self, ckpt: str = "yolo12n.pt", rank: int = 4,
                 lora_targets=None, nc: int = None,
                 full_param: bool = False, use_lora: bool = True,
                 lora_strategy: str = "adaptive"):
        """
        lora_targets: List tên class module để inject LoRA.
            None → ['C2f', 'C3k2'] (mặc định — C2fAttn bị loại vì hidden dim đặc biệt gây shape mismatch)
            Có thể truyền ['Conv'] để adapt domain shift nặng hơn (underwater).
        nc: Số lượng class của dataset. Cần set đúng để khởi tạo head trước khi inject LoRA.
        full_param: Train toàn bộ mô hình, không đóng băng, không LoRA.
        use_lora: Có sử dụng LoRA hay không.
        """
        self.yolo = YOLO(ckpt)
        
        # [FIX BUG] Xóa cờ "inference tensor" do Ultralytics EMA lưu vào file best.pt
        self.strip_inference_tensors()

        self.rank = rank
        self.full_param = full_param
        self.use_lora = use_lora
        self.lora_strategy = lora_strategy

        # Override classes if needed BEFORE injecting LoRA
        if nc is not None and hasattr(self.yolo.model, 'yaml') and self.yolo.model.yaml.get('nc') != nc:
            from ultralytics.nn.tasks import DetectionModel
            cfg = self.yolo.model.yaml.copy()
            cfg['nc'] = nc
            
            # Rebuild model with correct nc
            new_model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
            
            # Transfer weights with shape matching (omitting mismatched classification head weights)
            current_sd = self.yolo.model.state_dict()
            new_sd = new_model.state_dict()
            transfer_sd = {k: v for k, v in current_sd.items() 
                           if k in new_sd and v.shape == new_sd[k].shape}
            
            new_model.load_state_dict(transfer_sd, strict=False)
            
            # [FIX BUG] Khởi tạo stride và bias chuẩn YOLO để tránh mất mAP
            if hasattr(self.yolo.model, 'stride'):
                new_model.stride = self.yolo.model.stride
                m = new_model.model[-1]
                m.stride = new_model.stride
                if hasattr(m, 'bias_init'):
                    m.bias_init()
                    
            self.yolo.model = new_model
            print(f"[StudentModel] Replaced Detection Head for nc={nc}")

        if not self.full_param and self.use_lora:
            from detection_2d.models.lora import LoRAConv2d
            # Kiểm tra xem checkpoint đã chứa sẵn LoRAConv2d chưa
            existing_lora = sum(1 for m in self.yolo.model.modules() if isinstance(m, LoRAConv2d))
            
            if existing_lora > 0:
                # Warmup checkpoint đã chứa LoRAConv2d → không cần inject thêm
                print(f"[StudentModel] Checkpoint đã có {existing_lora} LoRAConv2d layers, bỏ qua inject.")
                head_idx = len(self.yolo.model.model) - 1
                head_prefix = f"model.{head_idx}."
                head_lora = [
                    name for name, module in self.yolo.model.named_modules()
                    if isinstance(module, LoRAConv2d) and name.startswith(head_prefix)
                ]
                if head_lora:
                    raise RuntimeError(
                        f"[StudentModel] Existing checkpoint contains LoRA inside Detect head: "
                        f"{head_lora[:5]}. Regenerate warmup without head LoRA."
                    )
                if lora_strategy in ("fixed", "all"):
                    ranks = sorted({int(m.lora_A.shape[0]) for m in self.yolo.model.modules() if isinstance(m, LoRAConv2d)})
                    if ranks != [int(rank)]:
                        raise RuntimeError(
                            f"[StudentModel] Existing LoRA checkpoint uses ranks={ranks}, "
                            f"but fixed LoRA expects rank={rank}. Regenerate the student warmup checkpoint."
                        )
            else:
                # Base model (vd: yolo12n.pt) → cần inject LoRA mới
                is_nano = '12n' in ckpt.lower() or '11n' in ckpt.lower() or '8n' in ckpt.lower()
                actual_strategy = lora_strategy
                actual_targets = ['Conv'] if is_nano else lora_targets

                injected = inject_lora(self.yolo.model, target_layer_names=actual_targets, rank=rank, strategy=actual_strategy)
                print(
                    f"[StudentModel] Injected LoRA into {injected} layers "
                    f"(Targets: {actual_targets}, Strategy: {actual_strategy}, Rank: {rank})."
                )
                
                # Load lại weights LoRA từ checkpoint NẾU CÓ (vd: warmup đã bake vào .pt nhưng YOLO vứt khi load)
                checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
                if 'model' in checkpoint:
                    ckpt_state = checkpoint['model'].state_dict() if hasattr(checkpoint['model'], 'state_dict') else checkpoint['model']
                    lora_state = {k: v for k, v in ckpt_state.items() if 'lora_' in k}
                    if len(lora_state) > 0:
                        self.yolo.model.load_state_dict(lora_state, strict=False)
                        print(f"[StudentModel] Recovered {len(lora_state)} LoRA tensors from {ckpt}!")

        if self.full_param:
            for param in self.yolo.model.parameters():
                param.requires_grad_(True)
        else:
            # Đóng băng tất cả, trừ payload keys. BatchNorm nằm trong payload
            # và được aggregate toàn cục giống các tensor trainable khác.
            for name, param in self.yolo.model.named_parameters():
                if self._is_payload_key(name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.yolo.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.yolo.model.parameters())
        mode_str = "Full Params" if self.full_param else ("LoRA+Head" if self.use_lora else "Head Only")
        print(f"[StudentModel] Trainable ({mode_str}): {trainable:,} / {total:,} params "
              f"({100*trainable/total:.1f}%)")

    def _is_payload_key(self, k: str, downlink: bool = False) -> bool:
        head_idx = len(self.yolo.model.model) - 1
        head_prefix = f'model.{head_idx}.'
        # [CRITICAL FIX] ĐÃ BỎ FedBN: Bắt buộc gửi toàn bộ tham số BatchNorm 
        # (weight, bias, running_mean, running_var) qua mạng để tổng hợp. 
        # Nếu không, Global Model sẽ dùng BN stats cũ rích từ Round 0 để đánh giá 
        # Conv weights mới, gây hiện tượng FL không hội tụ (mAP giảm) ở các vòng lẻ!
        if 'bn' in k or 'running' in k or 'tracked' in k:
            return True
        # FlexLoRA gửi lora_A và lora_B lên Server để phân rã SVD
        if ('lora_B' in k or 'lora_A' in k) and self.use_lora:
            if k.startswith(head_prefix):
                return False
            return True
        
        # Head payload strategy (model.21):
        #   cv3 (classification): FULL branch → 48 KB (quan trọng nhất cho KD alignment)
        #   cv2 (box regression): chỉ output conv cuối (.2.*) → 12 KB (LoRA đã xử lý phần còn lại)
        #   dfl: 16 params, giữ luôn
        # Tổng head: ~60 KB | Tổng payload: ~187 KB / 300 KB budget.
        if head_prefix in k:
            # [Asymmetric] Downlink broadcasts FULL head (Gateway has power, AUV has limited uplink battery)
            if downlink:
                return True
            # Full cv3 branch (all 3 scales)
            if f'{head_prefix}cv3.' in k:
                return True
            # cv2: lấy lớp bottleneck (.1) và output cuối (.2) cho cả 3 scale
            # Lý do: .0 là feature extractor (LoRA đã xử lý), .1-.2 là quan trọng cho box reg
            if f'{head_prefix}cv2.' in k:
                parts = k.split('.')
                # format: model.21.cv2.<scale>.<layer>.<...>
                if len(parts) >= 5 and parts[4] in ('1', '2'):
                    return True
            # dfl weights
            if f'{head_prefix}dfl.' in k:
                return True
            return False
                
        # [NOTE] BN affine (weight, bias) bị ĐÓNG hoàn toàn để khớp với Teacher
        # (Teacher checkpoint có 0/410 BN params mở → Student phải nhất quán).
        # FrozenBatchNorm2d trong __init__ đã xử lý việc này.
        return False

    def strip_inference_tensors(self):
        """Xóa cờ inference tensor khỏi toàn bộ model (để tránh lỗi khi quay lại Train sau Eval)."""
        for param in self.yolo.model.parameters():
            param.data = param.data.clone().detach()
        for buf in self.yolo.model.buffers():
            buf.data = buf.data.clone().detach()

    def trainable_state_dict(self, downlink: bool = False) -> dict:
        """
        Trả về chỉ các tensor cần truyền qua mạng:
          - Nếu full_param: toàn bộ model
          - Nếu dùng LoRA: lora_A, lora_B, và head
          - Nếu không dùng LoRA (nolora): chỉ head
        """
        if self.full_param:
            # Truyền toàn bộ
            return {k: v.cpu().clone() for k, v in self.yolo.model.state_dict().items()}

        return {k: v.cpu().clone()
                for k, v in self.yolo.model.state_dict().items()
                if self._is_payload_key(k, downlink=downlink)}

    def load_trainable_state_dict(self, state_dict: dict):
        """Nạp state dict (LoRA + Head partial) từ server aggregate."""
        if not state_dict:
            return
            
        # Lấy device hiện tại của model
        try:
            device = next(self.yolo.model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

        # [FIX BUG] Tránh dùng load_state_dict vì hàm này dùng param.copy_() gây lỗi Inplace update
        # trên các inference tensors (VD: model.0.conv.bias vốn bị đóng băng).
        # Ta chỉ cập nhật .data cho các tensor thực sự nhận được từ server (LoRA + Head).
        for name, param in self.yolo.model.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].clone().detach().to(device=device, dtype=param.dtype)
                
        for name, buf in self.yolo.model.named_buffers():
            if name in state_dict:
                buf.data = state_dict[name].clone().detach().to(device=device, dtype=buf.dtype)

    def bake_lora(self):
        """
        Gộp LoRA vào Conv weight gốc và THAY THẾ LoRAConv2d → Conv2d thường.
        Hàm này bắt buộc phải gọi trước khi chạy student.yolo.val() để tránh việc 
        thuật toán fuse() của Ultralytics vứt bỏ LoRAConv2d.
        """
        from detection_2d.models.lora import LoRAConv2d
        import torch.nn as nn
        
        merged_count = 0
        for parent_name, parent_module in list(self.yolo.model.named_modules()):
            for child_name, child_module in list(parent_module.named_children()):
                if not isinstance(child_module, LoRAConv2d):
                    continue

                with torch.no_grad():
                    lora_weight = (child_module.lora_B @ child_module.lora_A).view(
                        child_module.weight.shape
                    ) * child_module.scaling
                    baked_weight = child_module.weight.data + lora_weight

                    new_conv = nn.Conv2d(
                        in_channels=child_module.in_channels,
                        out_channels=child_module.out_channels,
                        kernel_size=child_module.kernel_size,
                        stride=child_module.stride,
                        padding=child_module.padding,
                        dilation=child_module.dilation,
                        groups=child_module.groups,
                        bias=child_module.bias is not None,
                        padding_mode=child_module.padding_mode,
                    )
                    new_conv.weight.data = baked_weight
                    if child_module.bias is not None:
                        new_conv.bias.data = child_module.bias.data.clone()

                setattr(parent_module, child_name, new_conv)
                merged_count += 1
        
        if merged_count > 0:
            print(f"[StudentModel] Baked {merged_count} LoRA layers into base weights.")
        return merged_count


class TeacherModel:
    """
    YOLO12l frozen — Oracle KD. Không tham gia FL.
    Chỉ dùng để lấy soft-logits trong KDDetectionTrainer.

    Hỗ trợ 2 loại checkpoint:
      - yolo12l_lora_baked.pt  : LoRA đã được bake vào base weights (ưu tiên).
                                  Load trực tiếp bằng YOLO() — an toàn với fuse().
      - yolo12l_lora_pretrained.pt : Checkpoint Ultralytics gốc. Nếu phát hiện
                                     LoRAConv2d bên trong thì tự bake trước khi dùng.
      - yolo12l.pt / yolo12l_pretrained.pt : YOLO gốc không LoRA.
    """

    def __init__(self, ckpt: str = "yolo12l.pt", rank: int = None, nc: int = 4,
                 lora_targets=None, lora_strategy: str = "adaptive"):
        from detection_2d.models.lora import LoRAConv2d

        self.yolo = YOLO(ckpt)

        # Kiểm tra file có chứa LoRAConv2d không
        n_lora = sum(1 for m in self.yolo.model.modules() if isinstance(m, LoRAConv2d))

        if n_lora > 0:
            print(f"[TeacherModel] Loaded {ckpt} — phát hiện {n_lora} LoRAConv2d. "
                  f"GIỮ NGUYÊN để trích xuất LoRA Projections cho KD.")
            if lora_strategy in ("fixed", "all") and rank is not None:
                ranks = sorted({int(m.lora_A.shape[0]) for m in self.yolo.model.modules() if isinstance(m, LoRAConv2d)})
                if ranks != [int(rank)]:
                    raise RuntimeError(
                        f"[TeacherModel] Existing LoRA teacher uses ranks={ranks}, "
                        f"but fixed LoRA expects rank={rank}. Retrain yolo12l_lora_pretrained.pt."
                    )
        elif rank is not None:
            actual_targets = list(lora_targets) if lora_targets is not None else ['Conv']
            injected = inject_lora(
                self.yolo.model,
                target_layer_names=actual_targets,
                rank=rank,
                strategy=lora_strategy,
            )
            print(
                f"[TeacherModel] Loaded clean {ckpt} — injected {injected} LoRAConv2d layers "
                f"(Targets: {actual_targets}, Strategy: {lora_strategy}, Rank: {rank}) "
                f"for projection KD/pretraining."
            )
        else:
            print(f"[TeacherModel] Loaded {ckpt} — không có LoRAConv2d (clean checkpoint).")

        # Đóng băng toàn bộ
        for param in self.yolo.model.parameters():
            param.data = param.data.clone().detach()
            param.requires_grad = False
        for buf in self.yolo.model.buffers():
            buf.data = buf.data.clone().detach()

        self.yolo.model.eval()
        print(f"[TeacherModel] Frozen, eval mode. ✅")

    def get_outputs(self, imgs: torch.Tensor):
        """Forward pass không gradient — dùng trong KD criterion."""
        with torch.no_grad():
            return self.yolo.model(imgs)
