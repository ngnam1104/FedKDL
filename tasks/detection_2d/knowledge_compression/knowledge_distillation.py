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


def _strip_inference_tensors(module: nn.Module):
    """Clone-detach all params/buffers to drop inference-mode tensor flags."""
    for p in module.parameters():
        p.data = p.data.clone().detach()
    for b in module.buffers():
        b.data = b.data.clone().detach()


def _count_inference_tensors(module: nn.Module) -> tuple[int, list[str]]:
    """Count tensors carrying inference-mode flag and return a few sample names."""
    count = 0
    sample_names = []

    for name, param in module.named_parameters():
        if hasattr(param, 'is_inference') and param.is_inference():
            count += 1
            if len(sample_names) < 5:
                sample_names.append(f"param:{name}")

    for name, buf in module.named_buffers():
        if hasattr(buf, 'is_inference') and buf.is_inference():
            count += 1
            if len(sample_names) < 5:
                sample_names.append(f"buffer:{name}")

    return count, sample_names


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
        self.student_wrapper = None
        self.kd_temperature: float = 4.0
        self.kd_lambda: float = 1.0
        
        # Accumulators for logging KD loss
        self.epoch_box_loss = 0.0
        self.epoch_kl_loss = 0.0
        self.epoch_hidden_loss = 0.0
        self.epoch_attn_loss = 0.0
        self.epoch_kd_loss = 0.0
        self.batch_count = 0
        self.kd_epoch_history = []

    def _setup_train(self):
        from ultralytics.utils import LOGGER
        original_warning = LOGGER.warning
        LOGGER.warning = lambda *args, **kwargs: None

        student_before, student_samples_before = _count_inference_tensors(self.model)
        if student_before > 0:
            LOGGER.info(
                f"[InferenceCheck][KD] Student has {student_before} inference tensors before strip. "
                f"Samples: {student_samples_before}"
            )

        teacher_before = 0
        teacher_samples_before = []
        if self.teacher_model is not None:
            teacher_before, teacher_samples_before = _count_inference_tensors(self.teacher_model)
            if teacher_before > 0:
                LOGGER.info(
                    f"[InferenceCheck][KD] Teacher has {teacher_before} inference tensors before strip. "
                    f"Samples: {teacher_samples_before}"
                )

        # Ensure both student/teacher are regular tensors before training graph is built.
        _strip_inference_tensors(self.model)
        if self.teacher_model is not None:
            _strip_inference_tensors(self.teacher_model)

        student_after, student_samples_after = _count_inference_tensors(self.model)
        if student_after > 0:
            raise RuntimeError(
                f"[InferenceCheck][KD] Student still has {student_after} inference tensors after strip. "
                f"Samples: {student_samples_after}"
            )

        if self.teacher_model is not None:
            teacher_after, teacher_samples_after = _count_inference_tensors(self.teacher_model)
            if teacher_after > 0:
                raise RuntimeError(
                    f"[InferenceCheck][KD] Teacher still has {teacher_after} inference tensors after strip. "
                    f"Samples: {teacher_samples_after}"
                )
        
        # Callback để log KD loss ra console
        def log_kd_loss(trainer):
            if hasattr(trainer, 'batch_count') and trainer.batch_count > 0:
                mean_box = trainer.epoch_box_loss / trainer.batch_count
                mean_kl = trainer.epoch_kl_loss / trainer.batch_count
                mean_hidden = trainer.epoch_hidden_loss / trainer.batch_count
                mean_attn = trainer.epoch_attn_loss / trainer.batch_count
                mean_weighted = trainer.epoch_kd_loss / trainer.batch_count

                print(
                    f"[KD Epoch {trainer.epoch + 1}] "
                    f"Box: {mean_box:.4f} | "
                    f"KL: {mean_kl:.4f} | "
                    f"Hidden: {mean_hidden:.4f} | "
                    f"Attn: {mean_attn:.4f} | "
                    f"Weighted KD: {mean_weighted:.4f}"
                )

                trainer.kd_epoch_history.append({
                    'epoch': int(trainer.epoch + 1),
                    'box': float(mean_box),
                    'kl': float(mean_kl),
                    'hidden': float(mean_hidden),
                    'attn': float(mean_attn),
                    'weighted': float(mean_weighted),
                    'batches': int(trainer.batch_count),
                })
                trainer.epoch_box_loss = 0.0
                trainer.epoch_kl_loss = 0.0
                trainer.epoch_hidden_loss = 0.0
                trainer.epoch_attn_loss = 0.0
                trainer.epoch_kd_loss = 0.0
                trainer.batch_count = 0
                
        self.add_callback("on_train_epoch_end", log_kd_loss)

        # [FIX] Prevent ModelEMA from deepcopying a bound method (circular reference to Trainer)
        # which contains an unpicklable DataLoaderIter from the previous FL round.
        from ultralytics.utils.torch_utils import unwrap_model
        model_unwrapped = unwrap_model(self.model)
        if getattr(model_unwrapped, 'criterion', None) is getattr(self, '_kd_criterion_wrapper', None):
            if hasattr(self, 'original_criterion'):
                model_unwrapped.criterion = self.original_criterion
            else:
                del model_unwrapped.criterion

        try:
            super()._setup_train()
        finally:
            LOGGER.warning = original_warning
            
        # [FIX] Ultralytics stores the criterion inside the model (self.model.criterion) lazily,
        # not in the Trainer. We must initialize it and wrap it there.
        from ultralytics.utils.torch_utils import unwrap_model
        model_unwrapped = unwrap_model(self.model)
        
        if getattr(model_unwrapped, "criterion", None) is None:
            model_unwrapped.criterion = model_unwrapped.init_criterion()
            
        if not hasattr(self, 'original_criterion'):
            self.original_criterion = model_unwrapped.criterion
            model_unwrapped.criterion = self._kd_criterion_wrapper

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        
        if self.student_wrapper and not self.student_wrapper.full_param:
            payload_keys = set(self.student_wrapper.trainable_state_dict().keys())
            for k, v in model.named_parameters():
                if k in payload_keys:
                    v.requires_grad = True
                else:
                    v.requires_grad = False
                
        # Filter frozen parameters from optimizer to prevent errors
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if getattr(p, 'requires_grad', False)]
            
        return optimizer

    def validate(self):
        """Bỏ qua validate giữa các epoch để tiết kiệm thời gian cho Tier 3."""
        return {}, 0.0

    def final_eval(self):
        """Bỏ qua bước Validate dư thừa ở cuối quá trình KD."""
        from ultralytics.utils.torch_utils import strip_optimizer, unwrap_model
        
        # [FIX] Trả lại hàm loss gốc để tránh memory leak/circular reference sang Trainer cũ
        model_unwrapped = unwrap_model(self.model)
        if hasattr(self, 'original_criterion'):
            model_unwrapped.criterion = self.original_criterion
            
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

    def _kd_criterion_wrapper(self, preds, batch):
        """
        Override: Hàm loss đầy đủ theo Eq. 37.

        preds : output của Student model trong training loop.
        batch : dict chứa 'img', 'cls', 'bboxes', ...
        """
        # ── 1. Task Loss của Student ──────────────────────────────────────
        loss_stu, loss_items = self.original_criterion(preds, batch)

        if self.teacher_model is None:
            return loss_stu, loss_items

        imgs = batch['img']

        # ── 2. Thu hidden features của Student (trong pha forward hiện tại) ─
        if self.batch_count == 0:
            print(f"\n[KD - Batch 1] Đang forward Student ({self.model.__class__.__name__}) để trích xuất features...")
        s_hook, s_handles = _register_hooks(self.model)
        # Forward lại student để thu features. Không dùng inference/no_grad để tránh
        # tạo tensor không theo dõi version counter trong graph huấn luyện.
        _ = self.model(imgs)
        student_feats = list(s_hook.outputs)
        _remove_hooks(s_handles)
        s_hook.clear()

        # ── 3. Forward Teacher (no gradient) + thu features ──────────────
        if self.batch_count == 0:
            print(f"[KD - Batch 1] Đang forward Teacher ({self.teacher_model.__class__.__name__}) để trích xuất features...")
        t_hook, t_handles = _register_hooks(self.teacher_model)
        with torch.no_grad():
            t_preds = self.teacher_model(imgs)
            # Task loss của Teacher (để tính adaptive denominator)
            try:
                loss_tch, _ = self.original_criterion(t_preds, batch)
                loss_tch = loss_tch.detach()
            except Exception:
                loss_tch = torch.tensor(1.0, device=loss_stu.device)

        teacher_feats = list(t_hook.outputs)
        _remove_hooks(t_handles)
        t_hook.clear()

        # ── 4. KL Divergence trên soft logits ────────────────────────────
        if self.batch_count == 0:
            print("[KD - Batch 1] Đang tính toán Distillation Loss (KL/Hidden/Attn)...")
        T = self.kd_temperature
        try:
            def _extract_cls(p):
                # 1. Handle YOLOv11 new dict format
                train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
                if isinstance(train_out, dict) and 'scores' in train_out:
                    return train_out['scores']  # shape: [B, NC, Anchors]
                
                # 2. Handle older list format
                feats = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], list)) else p
                if isinstance(feats, list) and len(feats) > 0:
                    cat = torch.cat([xi.view(xi.shape[0], xi.shape[1], -1) for xi in feats], 2)
                    reg_max = 16
                    return cat[:, reg_max * 4:, :]
                
                # 3. Handle inference tensor format
                if isinstance(p, torch.Tensor):
                    # Usually [B, 4+NC, Anchors] for inference format (boxes already decoded to 4)
                    return p[:, 4:, :]
                
                return None

            s_cls = _extract_cls(preds)
            t_cls = _extract_cls(t_preds)
            
            if s_cls is not None and t_cls is not None and s_cls.shape == t_cls.shape:
                num_anchors = s_cls.shape[2]
                loss_kl = F.kl_div(
                    F.log_softmax(s_cls / T, dim=1),
                    F.softmax(t_cls / T, dim=1).detach(),
                    reduction='batchmean',
                ) * (T * T) / num_anchors
            else:
                if self.batch_count == 0:
                    s_shape = s_cls.shape if s_cls is not None else None
                    t_shape = t_cls.shape if t_cls is not None else None
                    print(f"[KD Warning] Extract cls failed or shape mismatch. s_cls: {s_shape}, t_cls: {t_shape}")
                loss_kl = torch.tensor(0.0, device=loss_stu.device)

        except Exception as e:
            import traceback
            print(f"[KD] KL fallback Error: {e}")
            traceback.print_exc()
            loss_kl = torch.tensor(0.0, device=loss_stu.device)

        # ── 4b. Bounding Box Distillation (MSE) ─────────────────────────
        try:
            def _extract_bboxes(p):
                # 1. Handle YOLOv11 new dict format
                train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
                if isinstance(train_out, dict):
                    if 'bboxes' in train_out:
                        return train_out['bboxes']
                    elif 'pred_bboxes' in train_out:
                        return train_out['pred_bboxes']
                    elif 'boxes' in train_out:
                        return train_out['boxes']
                
                # 2. Handle older list format
                feats = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], list)) else p
                if isinstance(feats, list) and len(feats) > 0:
                    cat = torch.cat([xi.view(xi.shape[0], xi.shape[1], -1) for xi in feats], 2)
                    reg_max = 16
                    return cat[:, :reg_max * 4, :]
                
                # 3. Handle inference tensor format
                if isinstance(p, torch.Tensor):
                    return p[:, :4, :]
                
                return None

            s_box = _extract_bboxes(preds)
            t_box = _extract_bboxes(t_preds)

            if s_box is not None and t_box is not None and s_box.shape == t_box.shape:
                loss_box_kd = F.mse_loss(s_box, t_box.detach())
            else:
                if self.batch_count == 0:
                    s_shape = s_box.shape if s_box is not None else None
                    t_shape = t_box.shape if t_box is not None else None
                    print(f"[KD Warning] Extract bboxes failed or mismatch. s_box: {s_shape}, t_box: {t_shape}")
                loss_box_kd = torch.tensor(0.0, device=loss_stu.device)
        except Exception as e:
            if self.batch_count == 0:
                print(f"[KD] Box fallback Error: {e}")
            loss_box_kd = torch.tensor(0.0, device=loss_stu.device)

        # ── 5. Adaptive Hidden Loss — MSE(H^t, W^h H^s) ─────────────────
        loss_hidden = _adaptive_hidden_loss(student_feats, teacher_feats).to(loss_stu.device)

        # ── 6. Adaptive Attention Loss — MSE(A^t, A^s) ───────────────────
        loss_attn = _adaptive_attention_loss(student_feats, teacher_feats).to(loss_stu.device)

        # ── 7. Tổng distillation với Adaptive Denominator (Eq. 37) ───────
        # Cấp Tỷ trọng ưu tiên (Priorities) để BOOST RECALL theo hướng Feature-based
        # Khôi phục loss tự nhiên (không chia cho .detach() vì gây nhiễu gradient khi loss nhỏ)
        # Tăng mạnh trọng số Hidden (x10) và Attn (x50) vì độ lớn (magnitude) của chúng quá nhỏ (0.5 và 0.03)
        # Giảm KL và Box xuống để giảm bớt sự khắt khe (strictness) từ Teacher gây tụt Recall.
        loss_dist_adaptive = (loss_kl * 0.5) + (loss_box_kd * 0.5) + (loss_hidden * 10.0) + (loss_attn * 50.0)
        
        # [FIX] Trả lại Supervised Loss (YOLO task loss) với trọng số nhỏ (0.5) 
        # để Student vẫn bám vào Ground Truth thực tế, tránh việc học mù quáng theo Teacher
        total_loss = loss_stu.clone() * 0.5
        
        if total_loss.ndim == 0:
            total_loss = total_loss + self.kd_lambda * loss_dist_adaptive
        else:
            total_loss[0] = total_loss[0] + self.kd_lambda * loss_dist_adaptive
        
        # Tích lũy log (Giá trị ĐÃ SCALE theo trọng số để thấy rõ độ lớn tham gia vào Gradient)
        self.epoch_box_loss += (loss_box_kd.item() * 0.5)
        self.epoch_kl_loss += (loss_kl.item() * 0.5)
        self.epoch_hidden_loss += (loss_hidden.item() * 10.0)
        self.epoch_attn_loss += (loss_attn.item() * 50.0)
        self.epoch_kd_loss += loss_dist_adaptive.item()
        self.batch_count += 1

        return total_loss, loss_items

    def get_kd_summary(self) -> dict:
        """Return round-level KD statistics for external logging/export."""
        if not self.kd_epoch_history:
            return {
                'kd_active': False,
                'kd_epochs': 0,
                'kd_box': 0.0,
                'kd_kl': 0.0,
                'kd_hidden': 0.0,
                'kd_attn': 0.0,
                'kd_weighted': 0.0,
            }

        n = float(len(self.kd_epoch_history))
        return {
            'kd_active': True,
            'kd_epochs': int(len(self.kd_epoch_history)),
            'kd_box': float(sum(e['box'] for e in self.kd_epoch_history) / n),
            'kd_kl': float(sum(e['kl'] for e in self.kd_epoch_history) / n),
            'kd_hidden': float(sum(e['hidden'] for e in self.kd_epoch_history) / n),
            'kd_attn': float(sum(e['attn'] for e in self.kd_epoch_history) / n),
            'kd_weighted': float(sum(e['weighted'] for e in self.kd_epoch_history) / n),
            'kd_last_epoch': self.kd_epoch_history[-1],
            'kd_epoch_history': list(self.kd_epoch_history),
        }
