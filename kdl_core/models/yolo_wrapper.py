"""
yolo_wrapper.py
Wrapper cho Student (YOLO26n + LoRA) và Teacher (YOLO12l, frozen).
Sử dụng fl_core/models/lora.py để inject LoRA.
"""
import torch
from ultralytics import YOLO
from kdl_core.models.lora import inject_lora


class StudentModel:
    """
    YOLO26n + LoRA injection cho Federated Learning.
    Chỉ {lora_A, lora_B, detect head} là trainable và được truyền qua mạng.
    """

    def __init__(self, ckpt: str = "yolo26n.pt", rank: int = 4,
                 lora_targets=None):
        """
        lora_targets: List tên class module để inject LoRA.
            None → ['C2f', 'C3k2', 'C2fAttn'] (mặc định theo YOLO26/v11/v8)
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

    def trainable_state_dict(self) -> dict:
        """Trả về chỉ các tensor cần truyền: LoRA + Detection Head."""
        return {k: v.cpu().clone()
                for k, v in self.yolo.model.state_dict().items()
                if 'lora_' in k or 'detect' in k.lower()}

    def load_trainable_state_dict(self, state_dict: dict):
        """Nạp state dict (LoRA + Head) từ server aggregate."""
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
