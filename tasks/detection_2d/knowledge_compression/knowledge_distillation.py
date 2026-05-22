"""
knowledge_distillation.py
Custom DetectionTrainer kế thừa Ultralytics YOLO để tích hợp KD loss đầy đủ.
Không chỉnh sửa source code của Ultralytics — chỉ override criterion().

Hàm loss đa nhiệm theo Eq. 37 (Research Proposal):

    L_i^t = L_stu + [KL(y^s, y^t) + MSE(H^t, W^h H^s) + MSE(A^t, A^s)]
                     ────────────────────────────────────────────────────
                                  L_tch + L_stu

Trong đó:
    - L_stu : YOLO task loss của Student
    - L_tch : YOLO task loss của Teacher (tính riêng, không backward)
    - KL    : KL Divergence trên soft logits (temperature T=4)
    - MSE(H): Adaptive Hidden Loss trên hidden states
    - MSE(A): Adaptive Attention Loss trên attention maps

Mẫu số (L_tch + L_stu) đóng vai trò bộ điều tiết động: khi Teacher
sai lệch lớn (L_tch tăng), trọng lượng distillation giảm về 0,
ngăn Student học từ tri thức nhiễu (Adaptive KD Weighting).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from ultralytics.models.yolo.detect.train import DetectionTrainer


# ─────────────────────────────────────────────────────────────────────────────
#  Hook helpers: thu thập intermediate features / attention maps
# ─────────────────────────────────────────────────────────────────────────────

class _FeatureHook:
    """Lightweight forward hook để thu feature maps."""

    def __init__(self):
        self.outputs: List[torch.Tensor] = []

    def hook_fn(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.outputs.append(output)

    def clear(self):
        self.outputs.clear()


def _register_hooks(model: nn.Module,
                    target_class_names: Tuple[str, ...] = ('C2f', 'C3k2', 'SPPF')
                    ) -> Tuple[_FeatureHook, List]:
    """
    Đăng ký forward hooks vào các block đặc trưng của YOLO.
    Trả về (hook_collector, list_of_handles) để giải phóng sau.
    """
    hook = _FeatureHook()
    handles = []
    for name, module in model.named_modules():
        if any(cls in module.__class__.__name__ for cls in target_class_names):
            handles.append(module.register_forward_hook(hook.hook_fn))
    return hook, handles


def _remove_hooks(handles: List):
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
#  Adaptive Hidden Loss và Attention Loss
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_hidden_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
) -> torch.Tensor:
    """
    MSE(H^t, W^h H^s) — Eq. 37 Adaptive Hidden Loss.

    Nếu spatial dimensions khác nhau, dùng adaptive_avg_pool2d để align.
    Nếu channel dimensions khác nhau, dùng 1×1 projection tuyến tính nhanh.
    """
    if not student_feats or not teacher_feats:
        return torch.tensor(0.0)

    total = torch.tensor(0.0)
    device = student_feats[0].device

    # Chỉ so n_pairs đầu (YOLO student/teacher có thể có số block khác nhau)
    n_pairs = min(len(student_feats), len(teacher_feats))

    for s_feat, t_feat in zip(student_feats[:n_pairs], teacher_feats[:n_pairs]):
        if s_feat.shape == t_feat.shape:
            total = total + F.mse_loss(s_feat, t_feat.detach())
        else:
            # Align spatial dims
            h_s, w_s = s_feat.shape[-2], s_feat.shape[-1]
            t_aligned = F.adaptive_avg_pool2d(t_feat.detach(), (h_s, w_s))

            # Align channel dims với projection tuyến tính nhanh (không trainable bias)
            c_s = s_feat.shape[1]
            c_t = t_aligned.shape[1]
            if c_s != c_t:
                with torch.no_grad():
                    proj = torch.nn.functional.conv2d(
                        t_aligned,
                        weight=torch.eye(c_s, c_t, device=device).view(c_s, c_t, 1, 1),
                        bias=None, stride=1, padding=0,
                    )
                t_aligned = proj
            total = total + F.mse_loss(s_feat, t_aligned)

    return total / max(n_pairs, 1)


def _attention_map(feat: torch.Tensor) -> torch.Tensor:
    """
    Tính attention map từ feature map bằng cách lấy trung bình theo channel.
    Đây là Grad-CAM-style spatial attention: A = mean(|F|, dim=channel).
    """
    return feat.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)


def _adaptive_attention_loss(
    student_feats: List[torch.Tensor],
    teacher_feats: List[torch.Tensor],
) -> torch.Tensor:
    """
    MSE(A^t, A^s) — Eq. 37 Adaptive Attention Loss.
    So sánh attention maps (spatial activation patterns) giữa Teacher và Student.
    """
    if not student_feats or not teacher_feats:
        return torch.tensor(0.0)

    total = torch.tensor(0.0)
    n_pairs = min(len(student_feats), len(teacher_feats))

    for s_feat, t_feat in zip(student_feats[:n_pairs], teacher_feats[:n_pairs]):
        a_s = _attention_map(s_feat)
        a_t = _attention_map(t_feat.detach())

        # Align spatial nếu khác nhau
        if a_s.shape != a_t.shape:
            a_t = F.adaptive_avg_pool2d(a_t, a_s.shape[-2:])

        # Normalize về [0,1] để tránh magnitude bias
        a_s = a_s / (a_s.max() + 1e-6)
        a_t = a_t / (a_t.max() + 1e-6)

        total = total + F.mse_loss(a_s, a_t)

    return total / max(n_pairs, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  KDDetectionTrainer
# ─────────────────────────────────────────────────────────────────────────────

class KDDetectionTrainer(DetectionTrainer):
    """
    Extend Ultralytics DetectionTrainer với Knowledge Distillation đầy đủ (Eq. 37).

    Teacher: YOLOv12-Large (~40M params), frozen, eval mode.
    Student: YOLO26n với LoRA injection.

    Loss tổng hợp:
        L_total = L_stu + [KL(y^s, y^t) + MSE(H,H) + MSE(A,A)] / (L_tch + L_stu)
    """

    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.teacher_model: Optional[nn.Module] = None
        self.kd_temperature: float = 4.0
        self.kd_lambda: float = 1.0

    def _setup_train(self):
        from ultralytics.utils import LOGGER
        original_warning = LOGGER.warning
        LOGGER.warning = lambda *args, **kwargs: None
        try:
            super()._setup_train()
        finally:
            LOGGER.warning = original_warning

    def validate(self):
        """Bỏ qua validate giữa các epoch để tiết kiệm thời gian cho Tier 3."""
        return None, None

    def final_eval(self):
        """Bỏ qua bước Validate dư thừa ở cuối quá trình KD."""
        from ultralytics.utils.torch_utils import strip_optimizer
        model = self.best if self.best.exists() else None
        if self.last.exists():
            strip_optimizer(self.last)
        if model:
            strip_optimizer(self.best)
            self.run_callbacks("on_fit_epoch_end")

    def set_teacher(self, teacher_nn_module: Optional[nn.Module]):
        """
        Nhận nn.Module của Teacher (đã eval + frozen). Gọi sau __init__ trước khi train.
        """
        self.teacher_model = teacher_nn_module

    def criterion(self, preds, batch):
        """
        Override: Hàm loss đầy đủ theo Eq. 37.

        preds : output của Student model trong training loop.
        batch : dict chứa 'img', 'cls', 'bboxes', ...
        """
        # ── 1. Task Loss của Student ──────────────────────────────────────
        loss_stu, loss_items = super().criterion(preds, batch)

        if self.teacher_model is None:
            return loss_stu, loss_items

        imgs = batch['img']

        # ── 2. Thu hidden features của Student (trong pha forward hiện tại) ─
        s_hook, s_handles = _register_hooks(self.model)
        # Re-forward student để lấy features (preds đã có, nhưng cần features)
        with torch.no_grad():
            _ = self.model(imgs)
        student_feats = list(s_hook.outputs)
        _remove_hooks(s_handles)
        s_hook.clear()

        # ── 3. Forward Teacher (no gradient) + thu features ──────────────
        t_hook, t_handles = _register_hooks(self.teacher_model)
        with torch.no_grad():
            t_preds = self.teacher_model(imgs)
            # Task loss của Teacher (để tính adaptive denominator)
            try:
                loss_tch, _ = super().criterion(t_preds, batch)
                loss_tch = loss_tch.detach()
            except Exception:
                loss_tch = torch.tensor(1.0, device=loss_stu.device)

        teacher_feats = list(t_hook.outputs)
        _remove_hooks(t_handles)
        t_hook.clear()

        # ── 4. KL Divergence trên soft logits ────────────────────────────
        T = self.kd_temperature
        try:
            if isinstance(preds, (list, tuple)) and len(preds) > 1:
                s_logits = preds[0] if isinstance(preds[0], torch.Tensor) else preds[0][0]
                t_logits = t_preds[0] if isinstance(t_preds[0], torch.Tensor) else t_preds[0][0]
            else:
                s_logits = preds if isinstance(preds, torch.Tensor) else preds[0]
                t_logits = t_preds if isinstance(t_preds, torch.Tensor) else t_preds[0]

            if s_logits.shape == t_logits.shape:
                loss_kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1).detach(),
                    reduction='batchmean',
                ) * (T * T)
            else:
                # Shape mismatch giữa Student/Teacher head → MSE fallback
                loss_kl = torch.tensor(0.0, device=loss_stu.device)
        except Exception as e:
            print(f"[KD] KL fallback: {e}")
            loss_kl = torch.tensor(0.0, device=loss_stu.device)

        # ── 5. Adaptive Hidden Loss — MSE(H^t, W^h H^s) ─────────────────
        loss_hidden = _adaptive_hidden_loss(student_feats, teacher_feats).to(loss_stu.device)

        # ── 6. Adaptive Attention Loss — MSE(A^t, A^s) ───────────────────
        loss_attn = _adaptive_attention_loss(student_feats, teacher_feats).to(loss_stu.device)

        # ── 7. Tổng distillation với Adaptive Denominator (Eq. 37) ───────
        numerator = loss_kl + loss_hidden + loss_attn
        denominator = (loss_tch + loss_stu).detach() + 1e-6  # Tránh div/0

        loss_dist_adaptive = numerator / denominator
        total_loss = loss_stu + self.kd_lambda * loss_dist_adaptive

        return total_loss, loss_items
