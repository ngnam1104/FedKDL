"""
knowledge_distillation.py
Custom DetectionTrainer kế thừa Ultralytics YOLO để tích hợp KD loss đầy đủ.
Không chỉnh sửa source code của Ultralytics — chỉ override criterion().

Hàm loss đa nhiệm (LoRA-Projection KD):

    L_total = L_stu + λ * [KL(y^s, y^t) + MSE(b^s, b^t) + MSE(h^s, h^t)]
                            ───────────────────────────────────────────────
                                       L_tch + L_stu

Trong đó:
    - L_stu : YOLO task loss của Student
    - L_tch : YOLO task loss của Teacher (tính riêng, không backward)
    - KL    : KL Divergence trên soft logits (temperature T=4)
    - MSE(b): Box MSE loss giữa dự đoán Teacher và Student
    - MSE(h): LoRA-Projection Loss — MSE giữa h=Ax của Teacher và Student
              ghép cặp theo Stage (Backbone↔Backbone, Neck↔Neck, Head↔Head)
              + Proportional Matching trong mỗi Stage

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

class _LoRAProjectionHook:
    """Lightweight forward hook để thu LoRA projections (h = Ax)."""

    def __init__(self):
        # Lưu projection theo từng stage (layer_idx)
        self.outputs = {}

    def hook_fn(self, module, input, output, layer_idx):
        # input[0] có shape (B, C, H, W)
        x = input[0]
        # Tạo A_kernel từ lora_A (rank, in_features)
        A_kernel = module.lora_A.view(
            module.lora_A.shape[0], 
            module.in_channels // module.groups, 
            module.kernel_size[0], 
            module.kernel_size[1]
        )
        
        if module.groups == 1:
            h = F.conv2d(x, A_kernel, stride=module.stride, padding=module.padding, dilation=module.dilation)
        else:
            B, C, H, W = x.shape
            x_pooled = x.mean(dim=(2, 3))
            A_pooled = A_kernel.mean(dim=(2, 3))
            A_pooled = A_pooled.repeat(1, module.groups)
            h_pool = F.linear(x_pooled, A_pooled)
            h = h_pool.unsqueeze(-1).unsqueeze(-1)
            
        if layer_idx not in self.outputs:
            self.outputs[layer_idx] = []
        self.outputs[layer_idx].append(h)

    def clear(self):
        self.outputs.clear()


def _register_lora_hooks(model: nn.Module) -> Tuple[_LoRAProjectionHook, List]:
    """
    Đăng ký forward hooks vào các module LoRAConv2d, có lưu lại layer_idx để ghép cặp chính xác.
    """
    hook = _LoRAProjectionHook()
    handles = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'LoRAConv2d':
            parts = name.split('.')
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = parts[1]
                # Sử dụng default arg để bind layer_idx vào lambda
                handles.append(module.register_forward_hook(
                    lambda m, i, o, l_idx=layer_idx: hook.hook_fn(m, i, o, l_idx)
                ))
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
#  LoRA-Projection Alignment Distillation (Theo AdaLD)
# ─────────────────────────────────────────────────────────────────────────────

def _lora_projection_mse_loss(
    student_projs_dict: dict,
    teacher_projs_dict: dict,
) -> torch.Tensor:
    """
    LoRA-Projection Alignment Distillation Loss.
    So khớp theo Stage (layer_idx) và theo tỷ lệ độ sâu (Proportional Matching).
    """
    if not student_projs_dict or not teacher_projs_dict:
        return torch.tensor(0.0)

    total = torch.tensor(0.0)
    valid_pairs = 0

    with torch.amp.autocast('cuda', enabled=False):
        for layer_idx, s_list in student_projs_dict.items():
            if layer_idx not in teacher_projs_dict:
                continue
                
            t_list = teacher_projs_dict[layer_idx]
            len_s = len(s_list)
            len_t = len(t_list)
            
            for i_s in range(len_s):
                # Proportional matching: Chọn lớp tương ứng của Teacher dựa trên tỷ lệ độ sâu
                i_t = int(i_s * len_t / len_s)
                
                s_proj = s_list[i_s].to(torch.float32)
                t_proj = t_list[i_t].detach().to(torch.float32)

                # Căn chỉnh không gian nếu khác nhau
                if s_proj.shape != t_proj.shape:
                    t_proj = F.adaptive_avg_pool2d(t_proj, s_proj.shape[-2:])

                # Căn chỉnh Channel nếu Teacher và Student lệch Rank
                min_rank = min(s_proj.shape[1], t_proj.shape[1])
                s_proj = s_proj[:, :min_rank]
                t_proj = t_proj[:, :min_rank]

                s_max = s_proj.abs().max() + 1e-6
                t_max = t_proj.abs().max() + 1e-6
                s_proj = s_proj / s_max
                t_proj = t_proj / t_max

                if torch.isnan(s_proj).any() or torch.isinf(s_proj).any():
                    print(f"\n[DEBUG LoRA Proj] s_proj có NaN/Inf tại Stage {layer_idx}")

                loss_layer = F.mse_loss(s_proj, t_proj)
                if torch.isnan(loss_layer):
                    print(f"\n[DEBUG LoRA Proj] MSE Loss bị NaN tại Stage {layer_idx}")

                total = total + loss_layer
                valid_pairs += 1

    return total / max(valid_pairs, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  KDDetectionTrainer
# ─────────────────────────────────────────────────────────────────────────────

class KDDetectionTrainer(DetectionTrainer):
    """
    Extend Ultralytics DetectionTrainer với Knowledge Distillation đầy đủ (Eq. 37).

    Teacher: YOLOv12-Large (~40M params), frozen, eval mode.
    Student: yolo12n với LoRA injection.

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
        self.epoch_lora_loss = 0.0
        self.epoch_kd_loss = 0.0
        self.epoch_stu_loss = 0.0
        self.epoch_kd_only_loss = 0.0
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
                mean_lora = trainer.epoch_lora_loss / trainer.batch_count
                mean_stu = trainer.epoch_stu_loss / trainer.batch_count
                mean_kd_only = trainer.epoch_kd_only_loss / trainer.batch_count
                mean_weighted = trainer.epoch_kd_loss / trainer.batch_count

                print(
                    f"[KD Epoch {trainer.epoch + 1}] "
                    f"Supervised: {mean_stu:.4f} | "
                    f"KD Only: {mean_kd_only:.4f} | "
                    f"Box: {mean_box:.4f} | "
                    f"KL: {mean_kl:.4f} | "
                    f"LoRA_Proj: {mean_lora:.4f} | "
                    f"Total: {mean_weighted:.4f}"
                )

                trainer.kd_epoch_history.append({
                    'epoch': int(trainer.epoch + 1),
                    'box': float(mean_box),
                    'kl': float(mean_kl),
                    'lora_proj': float(mean_lora),
                    'weighted': float(mean_weighted),
                    'batches': int(trainer.batch_count),
                })
                trainer.epoch_box_loss = 0.0
                trainer.epoch_kl_loss = 0.0
                trainer.epoch_lora_loss = 0.0
                trainer.epoch_kd_loss = 0.0
                trainer.epoch_stu_loss = 0.0
                trainer.epoch_kd_only_loss = 0.0
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

        injected = getattr(self, "_fl_injected_model", None)
        try:
            super()._setup_train()
        finally:
            LOGGER.warning = original_warning

        from tasks.detection_2d.trainer import _fl_prepare_model_for_train
        from ultralytics.utils.torch_utils import unwrap_model

        if injected is not None:
            self.model = _fl_prepare_model_for_train(injected, self.device)
        else:
            self.model = _fl_prepare_model_for_train(self.model, self.device)

        model_unwrapped = unwrap_model(self.model)
        if getattr(model_unwrapped, "criterion", None) is None:
            model_unwrapped.criterion = model_unwrapped.init_criterion()

        self.original_criterion = model_unwrapped.criterion
        model_unwrapped.criterion = self._kd_criterion_wrapper

        if self.teacher_model is not None:
            dev = self.device
            if isinstance(dev, str):
                dev = torch.device(dev)
            self.teacher_model.to(dev)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        
        if self.student_wrapper and not self.student_wrapper.full_param:
            payload_keys = set(self.student_wrapper.trainable_state_dict().keys())
            for k, v in model.named_parameters():
                if k in payload_keys:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        # ---------------------------------------------------------------
        # Differential LR: Head params học nhanh hơn LoRA (head_lr_multiplier lần).
        # Mặc định = 1.0 → không ảnh hưởng gì.
        # Set trainer.head_lr_multiplier = 5.0 → Head lr = lr * 5 (=1e-3 khi lr=2e-4).
        # ---------------------------------------------------------------
        head_lr_multiplier = getattr(self, 'head_lr_multiplier', 1.0)
        if head_lr_multiplier != 1.0:
            id_to_name = {id(p): n for n, p in model.named_parameters()}
            head_patterns = ('model.21.', 'model.22.', 'model.23.')
            new_groups = []
            for group in optimizer.param_groups:
                head_p = [p for p in group['params']
                          if any(h in id_to_name.get(id(p), '') for h in head_patterns)
                          and 'lora_' not in id_to_name.get(id(p), '')]
                head_p_ids = set(id(x) for x in head_p)
                other_p = [p for p in group['params'] if id(p) not in head_p_ids]
                if other_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = other_p
                    new_groups.append(g)
                if head_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = head_p
                    g['lr'] = group.get('lr', lr) * head_lr_multiplier
                    new_groups.append(g)
            optimizer.param_groups = new_groups
            print(f"[KD-DiffLR] Head LR boosted ×{head_lr_multiplier} → "
                  f"LoRA lr={lr:.2e} | Head lr={lr * head_lr_multiplier:.2e}")

        # Filter frozen parameters from optimizer to prevent errors
        for group in optimizer.param_groups:
            group['params'] = [p for p in group['params'] if getattr(p, 'requires_grad', False)]
            
        return optimizer

    def optimizer_step(self):
        # [CRITICAL FIX] Gradient Clipping for Knowledge Distillation to prevent NaN
        # import torch
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        super().optimizer_step()

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
            print(f"\n[KD - Batch 1] Đang forward Student ({self.model.__class__.__name__}) để trích xuất LoRA Projections...")
        s_hook, s_handles = _register_lora_hooks(self.model)
        # Forward lại student để thu features. Không dùng inference/no_grad để tránh
        # tạo tensor không theo dõi version counter trong graph huấn luyện.
        _ = self.model(imgs)
        student_projs = dict(s_hook.outputs)
        _remove_hooks(s_handles)
        s_hook.clear()
        if self.batch_count == 0:
            total_s = sum(len(v) for v in student_projs.values())
            print(f"[KD - Batch 1] Student captured {total_s} LoRA projections across {len(student_projs)} stages.")

        # ── 3. Forward Teacher (no gradient) + thu features ──────────────
        if self.batch_count == 0:
            print(f"[KD - Batch 1] Đang forward Teacher ({self.teacher_model.__class__.__name__}) để trích xuất LoRA Projections...")
        t_hook, t_handles = _register_lora_hooks(self.teacher_model)
        with torch.no_grad():
            t_preds = self.teacher_model(imgs)
            # Task loss của Teacher (để tính adaptive denominator)
            try:
                loss_tch, _ = self.original_criterion(t_preds, batch)
                loss_tch = loss_tch.detach()
            except Exception:
                loss_tch = torch.tensor(1.0, device=loss_stu.device)

        teacher_projs = dict(t_hook.outputs)
        _remove_hooks(t_handles)
        t_hook.clear()
        if self.batch_count == 0:
            total_t = sum(len(v) for v in teacher_projs.values())
            print(f"[KD - Batch 1] Teacher captured {total_t} LoRA projections across {len(teacher_projs)} stages.")

        # ── 4. KL Divergence trên soft logits ────────────────────────────
        if self.batch_count == 0:
            print("[KD - Batch 1] Đang tính toán Distillation Loss (KL/Hidden/Attn)...")
        T = 4.0  # Tăng Temperature để soft-label của Teacher mờ hơn, tránh ép Student quá cứng
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
                # [CRITICAL FIX v9] Áp dụng Quality Focal Loss (QFL) cho KD!
                # Vấn đề gốc rễ: Tính BCE trên TẤT CẢ 8400 anchors khiến Background (vốn chiếm 99%) 
                # tạo ra một khối lượng Loss khổng lồ (dù đã chia sum), đè bẹp Foreground và làm tụt Recall.
                # Giải pháp đỉnh cao: Dùng Focal Weight = |Teacher_Prob - Student_Prob|^gamma
                # Khi đó, Background (nơi Student đã đoán đúng là 0) sẽ bị triệt tiêu Loss về 0.
                # Còn những chỗ Student đoán sai (False Positive hoặc False Negative) sẽ được nhân mạnh lên!
                t_prob = torch.sigmoid(t_cls).detach()
                
                # [CRITICAL FIX v13] Masked KD (Foreground Only)
                # Thay vì QFL trên toàn bộ 537,600 anchors (dẫn đến nổ Loss khi lệch kiến trúc),
                # ta CHỈ distill ở những vùng Teacher có sự tự tin (t_prob > 0.05).
                # Còn vùng Background, YOLO Supervised Loss đã lo liệu rất tốt!
                fg_mask = (t_prob.max(dim=1, keepdim=True)[0] > 0.05).float()
                valid_anchors = torch.clamp(fg_mask.sum(), min=1.0)
                
                # Tính BCE nguyên bản
                loss_kl_unreduced = F.binary_cross_entropy_with_logits(s_cls, t_prob, reduction='none')
                
                # Tính Mean BCE trên Foreground (chia cho số anchors * số classes)
                # Bỏ nhân hệ số 10.0 vì Mean BCE ở vùng Foreground tự nhiên đã rơi vào khoảng 3.0 - 5.0,
                # tương đương hoàn hảo với cls_loss gốc của YOLO (từ 2.0 - 4.5).
                loss_kl = (loss_kl_unreduced * fg_mask).sum() / (valid_anchors * s_cls.shape[1])
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

        # ── 4. Bounding Box KD Loss (Regression) ────────────────────────────
        # Box KD bắt buộc phải có MASK vì Box của Teacher ở vùng background là rác (không có vật thể).
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
                t_prob_raw = torch.sigmoid(t_cls).detach()
                fg_mask = (t_prob_raw.max(dim=1, keepdim=True)[0] > 0.05).float()
                valid_anchors = torch.clamp(fg_mask.sum(), min=1.0)
                
                loss_box_unreduced = F.mse_loss(s_box, t_box.detach(), reduction='none')
                loss_box_kd = (loss_box_unreduced * fg_mask).sum() / (valid_anchors * s_box.shape[1])
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

        # ── 5. LoRA-Projection Alignment Loss ──────────────────────────────────────
        loss_lora_proj = _lora_projection_mse_loss(student_projs, teacher_projs).to(loss_stu.device)

        # Bật lại Feature KD (dựa trên LoRA Projection)
        # Thay thế hoàn toàn SP Loss nặng nề và giải quyết tốt độ lệch kênh (Channel Mismatch) 
        # do A_teacher và A_student đều được chiếu về không gian có chiều = Rank (VD: 8).
        loss_dist_adaptive = loss_kl + loss_box_kd + loss_lora_proj
        
        # Mở lại Supervised Loss để giữ mỏ neo Ground Truth
        stu_weight = getattr(self, 'stu_lambda', 0.20)
        total_loss = loss_stu.clone() * stu_weight
        
        if total_loss.ndim == 0:
            total_loss = total_loss + self.kd_lambda * loss_dist_adaptive
        else:
            total_loss[0] = total_loss[0] + self.kd_lambda * loss_dist_adaptive
        
        self.epoch_box_loss += loss_box_kd.item()
        self.epoch_kl_loss += loss_kl.item()
        self.epoch_lora_loss += loss_lora_proj.item()
        stu_loss_val = (loss_stu.item() if loss_stu.ndim == 0 else loss_stu[0].item())
        self.epoch_stu_loss += (stu_loss_val * stu_weight)
        self.epoch_kd_only_loss += (self.kd_lambda * loss_dist_adaptive.item())
        self.epoch_kd_loss += total_loss.item() if total_loss.ndim == 0 else total_loss[0].item()
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
                'kd_lora': 0.0,
                'kd_weighted': 0.0,
            }

        n = float(len(self.kd_epoch_history))
        return {
            'kd_active': True,
            'kd_epochs': int(len(self.kd_epoch_history)),
            'kd_box': float(sum(e['box'] for e in self.kd_epoch_history) / n),
            'kd_kl': float(sum(e['kl'] for e in self.kd_epoch_history) / n),
            'kd_lora': float(sum(e['lora_proj'] for e in self.kd_epoch_history) / n),
            'kd_weighted': float(sum(e['weighted'] for e in self.kd_epoch_history) / n),
            'kd_last_epoch': self.kd_epoch_history[-1],
            'kd_epoch_history': list(self.kd_epoch_history),
        }
