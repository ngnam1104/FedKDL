"""
yolo_wrapper.py
Wrapper cho Student (yolo12n + LoRA) và Teacher (YOLO12l, frozen).
Sử dụng fl_core/models/lora.py để inject LoRA.
"""
import torch
from ultralytics import YOLO
from tasks.detection_2d.models.lora import inject_lora


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
                 full_param: bool = False, use_lora: bool = True):
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
            # [CRITICAL FIX for NANO LORA]
            # YOLO Nano quá nhỏ, nếu skip các shallow layers (0-3) thì nó không thể học lại 
            # bộ lọc màu sắc (color shift) cho domain dưới nước, dẫn đến mAP cực kỳ thấp.
            # Ta tự động chuyển sang tiêm LoRA toàn diện ('Conv', strategy='all') nếu là bản 'n'.
            is_nano = '12n' in ckpt.lower() or '11n' in ckpt.lower() or '8n' in ckpt.lower()
            
            actual_strategy = "all" if is_nano else "adaptive"
            actual_targets = ['Conv'] if is_nano else lora_targets
            
            # FlexLoRA: Không khóa seed, cho phép A khởi tạo ngẫu nhiên và được train độc lập ở mỗi AUV
            injected = inject_lora(self.yolo.model, target_layer_names=actual_targets, rank=rank, strategy=actual_strategy)
            print(f"[StudentModel] Injected LoRA into {injected} layers (Targets: {actual_targets}, Strategy: {actual_strategy}).")

        if self.full_param:
            for param in self.yolo.model.parameters():
                param.requires_grad_(True)
        else:
            # Đóng băng tất cả, trừ payload keys
            for name, param in self.yolo.model.named_parameters():
                if self._is_payload_key(name):
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            
            # [CRITICAL FIX v11] Đóng băng vĩnh viễn BatchNorm statistics!
            # Nếu dùng LoRA (full_param=False), Backbone bị đóng băng. 
            # Tuy nhiên, nếu không khóa BatchNorm, nó vẫn sẽ cập nhật running_mean/var 
            # và dùng Batch statistics để chuẩn hóa trong lúc train, gây ra sự sai lệch nghiêm trọng
            # giữa các AUV và khi suy luận (đó là lý do mô hình bị nổ ở Round 4-5).
            
            for name, module in self.yolo.model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    # Bỏ đóng băng BN của Detection Head để tránh lỗi Domain Shift
                    if 'model.22' in name or 'model.23' in name:
                        continue
                    module.__class__ = FrozenBatchNorm2d
                    module.eval()

        trainable = sum(p.numel() for p in self.yolo.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.yolo.model.parameters())
        mode_str = "Full Params" if self.full_param else ("LoRA+Head" if self.use_lora else "Head Only")
        print(f"[StudentModel] Trainable ({mode_str}): {trainable:,} / {total:,} params "
              f"({100*trainable/total:.1f}%)")

    # Keys của lớp output classifier trong YOLO26 Detect head (nc-specific):
    #   cv3.0.2, cv3.1.2, cv3.2.2  (one2many branch)
    #   one2one_cv3.0.2, one2one_cv3.1.2, one2one_cv3.2.2  (one2one branch)
    # Tổng kích thước: ~2KB INT8 (nc=4, URPC2020)
    _HEAD_OUTPUT_SUFFIXES = (
        '.cv3.0.2.weight', '.cv3.1.2.weight', '.cv3.2.2.weight',
        '.cv3.0.2.bias',   '.cv3.1.2.bias',   '.cv3.2.2.bias',
        '.one2one_cv3.0.2.weight', '.one2one_cv3.1.2.weight', '.one2one_cv3.2.2.weight',
        '.one2one_cv3.0.2.bias',   '.one2one_cv3.1.2.bias',   '.one2one_cv3.2.2.bias',
    )

    def _is_payload_key(self, k: str) -> bool:
        # FlexLoRA gửi cả lora_A và lora_B lên Server để phân rã SVD
        if ('lora_B' in k or 'lora_A' in k) and self.use_lora:
            return True
        # Gửi toàn bộ BatchNorm của Detection Head lên Server để tổng hợp (FedBN -> Global BN)
        if ('model.22' in k or 'model.23' in k) and 'bn' in k:
            return True
        for suffix in self._HEAD_OUTPUT_SUFFIXES:
            if k.endswith(suffix):
                return True
        return False

    def strip_inference_tensors(self):
        """Xóa cờ inference tensor khỏi toàn bộ model (để tránh lỗi khi quay lại Train sau Eval)."""
        for param in self.yolo.model.parameters():
            param.data = param.data.clone().detach()
        for buf in self.yolo.model.buffers():
            buf.data = buf.data.clone().detach()

    def trainable_state_dict(self) -> dict:
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
                if self._is_payload_key(k)}

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
        from tasks.detection_2d.models.lora import LoRAConv2d
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

    def __init__(self, ckpt: str = "yolo12l.pt", rank: int = None, nc: int = 4):
        from tasks.detection_2d.models.lora import LoRAConv2d

        self.yolo = YOLO(ckpt)

        # Kiểm tra file có chứa LoRAConv2d không
        n_lora = sum(1 for m in self.yolo.model.modules() if isinstance(m, LoRAConv2d))

        if n_lora > 0:
            print(f"[TeacherModel] Loaded {ckpt} — phát hiện {n_lora} LoRAConv2d. "
                  f"GIỮ NGUYÊN để trích xuất LoRA Projections cho KD.")
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

