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
                 lora_targets=None):
        """
        lora_targets: List tên class module để inject LoRA.
            None → ['C2f', 'C3k2', 'C2fAttn'] (mặc định theo YOLO11)
            Có thể truyền ['Conv'] để adapt domain shift nặng hơn (underwater).
        """
        self.yolo = YOLO(ckpt)
        self.rank = rank

        injected = inject_lora(self.yolo.model, target_layer_names=lora_targets, rank=rank)
        print(f"[StudentModel] Injected LoRA into {injected} Conv2d layers.")

        # Đóng băng tất cả, trừ LoRA params và Detection Head
        for name, param in self.yolo.model.named_parameters():
            if 'lora_' in name or 'detect' in name.lower():
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        trainable = sum(p.numel() for p in self.yolo.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.yolo.model.parameters())
        print(f"[StudentModel] Trainable: {trainable:,} / {total:,} params "
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

    def trainable_state_dict(self) -> dict:
        """
        Trả về chỉ các tensor cần truyền qua kênh âm thanh dưới nước:
          - LoRA adapters (lora_A, lora_B): ~72KB (r=4) hoặc ~144KB (r=8) INT8
          - cv3.x.2 output classifier conv: ~2KB INT8 (class-specific, cần update khi nc thay đổi)
        KHÔNG truyền: cv2, cv3 hidden layers, backbone weights — giữ cố định tại Gateway.
        """
        def _is_payload_key(k: str) -> bool:
            if 'lora_' in k:
                return True
            # Tìm suffix cv3.x.2 trong key state dict (prefix là 'model.model[-1].' hoặc tương tự)
            for suffix in self._HEAD_OUTPUT_SUFFIXES:
                if k.endswith(suffix):
                    return True
            return False

        return {k: v.cpu().clone()
                for k, v in self.yolo.model.state_dict().items()
                if _is_payload_key(k)}

    def load_trainable_state_dict(self, state_dict: dict):
        """Nạp state dict (LoRA + Head partial) từ server aggregate."""
        full_sd = self.yolo.model.state_dict()
        for k, v in state_dict.items():
            if k in full_sd:
                full_sd[k] = v.to(next(self.yolo.model.parameters()).device)
        self.yolo.model.load_state_dict(full_sd, strict=False)


class TeacherModel:
    """
    YOLO12l frozen — Oracle KD. Không tham gia FL.
    Chỉ dùng để lấy soft-logits trong KDDetectionTrainer.
    """

    def __init__(self, ckpt: str = "yolo12l.pt"):
        self.yolo = YOLO(ckpt)
        self.yolo.model.eval()
        for p in self.yolo.model.parameters():
            p.requires_grad_(False)
        print(f"[TeacherModel] Loaded {ckpt} — frozen, eval mode.")

    def get_outputs(self, imgs: torch.Tensor):
        """Forward pass không gradient — dùng trong KD criterion."""
        with torch.no_grad():
            return self.yolo.model(imgs)
