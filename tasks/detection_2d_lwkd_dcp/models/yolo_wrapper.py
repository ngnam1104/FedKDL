"""
yolo_wrapper.py
Wrapper cho Student (YOLO26n + LoRA) và Teacher (YOLO12l, frozen).
Sử dụng fl_core/models/lora.py để inject LoRA.
"""
import torch
from ultralytics import YOLO
from tasks.detection_2d.models.lora import inject_lora


class StudentModel:
    """
    YOLO11n + LoRA injection cho Federated Learning.
    Chỉ {lora_A, lora_B, detect head} là trainable và được truyền qua mạng.
    """

    def __init__(self, ckpt: str = "yolo11n.pt", rank: int = 4,
                 lora_targets=None, nc: int = None,
                 full_param: bool = False, use_lora: bool = True):
        """
        lora_targets: List tên class module để inject LoRA.
            None → ['C2f', 'C3k2', 'C2fAttn'] (mặc định theo YOLO11)
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
            injected = inject_lora(self.yolo.model, target_layer_names=lora_targets, rank=rank)
            print(f"[StudentModel] Injected LoRA into {injected} Conv2d layers.")

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
        if 'lora_' in k and self.use_lora:
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


class TeacherModel:
    """
    YOLO12l frozen — Oracle KD. Không tham gia FL.
    Chỉ dùng để lấy soft-logits trong KDDetectionTrainer.
    """

    def __init__(self, ckpt: str = "yolo12l.pt"):
        self.yolo = YOLO(ckpt)
        
        # [FIX BUG] Xóa cờ "inference tensor" an toàn
        for param in self.yolo.model.parameters():
            param.data = param.data.clone().detach()
            param.requires_grad = False
        for buf in self.yolo.model.buffers():
            buf.data = buf.data.clone().detach()
                
        self.yolo.model.eval()
        for p in self.yolo.model.parameters():
            p.requires_grad_(False)
        print(f"[TeacherModel] Loaded {ckpt} — frozen, eval mode.")

    def get_outputs(self, imgs: torch.Tensor):
        """Forward pass không gradient — dùng trong KD criterion."""
        with torch.no_grad():
            return self.yolo.model(imgs)
