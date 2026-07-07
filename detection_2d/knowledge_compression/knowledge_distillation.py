"""
knowledge_distillation.py
Custom DetectionTrainer kế thừa Ultralytics YOLO để tích hợp KD loss đầy đủ.
Không chỉnh sửa source code của Ultralytics — chỉ override criterion().

L_total = λ_sup L_sup + ρ_t (w_cls L_cls + w_box L_box + w_proj L_proj).
Mỗi nhánh KD được chuẩn hóa theo đóng góp supervised rồi phân bổ theo trọng số
cấu hình. ρ_t giảm dần theo vòng FL; confidence mask của Teacher loại background
yếu, còn box KD kết hợp DFL-KL và CIoU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import dist2bbox, make_anchors


def _compose_balanced_kd(
    loss_stu: torch.Tensor,
    stu_weight: float,
    component_values: dict[str, torch.Tensor],
    component_weights: dict[str, float],
    kd_lambda: float,
    balance_by_supervised: bool,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine KD components with a single-level weighted sum.

    L_kd = kd_lambda * sum(w_i * L_i) / sum(w_i)   (only active branches)

    `balance_by_supervised` is kept in the signature for backward compat but
    is now ignored — the old double-scaling logic was the root cause of KD
    harming mAP (see implementation_plan.md).
    """
    supervised_scalar = loss_stu if loss_stu.ndim == 0 else loss_stu.sum()
    supervised_weighted = supervised_scalar.detach().abs() * float(stu_weight)

    active_names = [
        name
        for name, value in component_values.items()
        if component_weights[name] > 0.0 and value.detach().abs().item() > 1e-12
    ]
    active_weight_sum = sum(component_weights[name] for name in active_names)

    # Simple weighted combination — no inflation, no double scale
    weighted_components: dict[str, torch.Tensor] = {
        name: loss_stu.new_tensor(0.0)
        for name in component_values
    }
    for name in active_names:
        value = component_values[name]
        normalized_weight = component_weights[name] / max(active_weight_sum, 1e-12)
        weighted_components[name] = value * normalized_weight * float(kd_lambda)

    weighted_kd = sum(
        weighted_components.values(),
        loss_stu.new_tensor(0.0),
    )
    kd_scale = loss_stu.new_tensor(float(kd_lambda))
    actual_kd_ratio = (
        weighted_kd.detach().abs() / supervised_weighted.clamp_min(1e-8)
    )
    return (
        supervised_weighted,
        weighted_components,
        weighted_kd,
        kd_scale,
        actual_kd_ratio,
    )



# ─────────────────────────────────────────────────────────────────────────────
#  Hook helpers: thu thập intermediate features / attention maps
# ─────────────────────────────────────────────────────────────────────────────

class _LoRAProjectionHook:
    """Lightweight forward hook để thu LoRA projections (h = Ax)."""

    def __init__(self, spatial_size: int = 8):
        # Lưu projection theo từng stage (layer_idx)
        self.outputs = {}
        self.spatial_size = spatial_size

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

        # Projection KD only needs the low-rank response pattern. Retaining all
        # 95 full-resolution maps can consume many GB and defeats small batches.
        if h.shape[-2] > self.spatial_size or h.shape[-1] > self.spatial_size:
            h = F.adaptive_avg_pool2d(h, (self.spatial_size, self.spatial_size))

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

def _projection_zero_like(projs_dict: dict) -> torch.Tensor:
    for proj_list in projs_dict.values():
        for tensor in proj_list:
            if torch.is_tensor(tensor):
                return tensor.new_tensor(0.0)
    return torch.tensor(0.0)


def _matched_teacher_index(i_s: int, len_s: int, len_t: int, anchor_match: bool = True) -> int:
    """Map a student LoRA projection to a teacher projection within the same YOLO stage."""
    if len_t <= 1:
        return 0
    if len_s <= 1:
        return 0
    if not anchor_match:
        return min(int(i_s * len_t / len_s), len_t - 1)

    if i_s == 0:
        return 0
    if i_s == len_s - 1:
        return len_t - 1
    if len_t <= 2:
        return min(max(i_s, 0), len_t - 1)

    ratio = (i_s - 1) / max(len_s - 2, 1)
    return min(len_t - 2, max(1, 1 + int(round(ratio * (len_t - 3)))))


def _projection_kl_loss_pair(
    s_h: torch.Tensor,
    t_h: torch.Tensor,
    temperature: float,
    projection_mode: str,
) -> torch.Tensor | None:
    s_h = s_h.to(torch.float32)
    t_h = t_h.detach().to(torch.float32)

    if s_h.shape[-2:] != t_h.shape[-2:]:
        t_h = F.adaptive_avg_pool2d(t_h, s_h.shape[-2:])

    mode = projection_mode.lower()
    if mode in {"lora_rank_proj", "rank", "rank_spatial"}:
        if s_h.shape[1] != t_h.shape[1]:
            return None
        s_flat = s_h.flatten(2)
        t_flat = t_h.flatten(2)
    elif mode in {"lora_spatial_proj", "spatial", "spatial_attention"}:
        # LoRA factors are not canonical: the same B@A can rotate rank axes.
        # Collapse rank into a spatial energy map, then distill where each
        # stage responds instead of forcing rank dimension i to match exactly.
        s_flat = s_h.pow(2).mean(dim=1, keepdim=True).flatten(2)
        t_flat = t_h.pow(2).mean(dim=1, keepdim=True).flatten(2)
    else:
        raise ValueError(
            f"Unknown KD projection mode: {projection_mode}. "
            "Use 'lora_spatial_proj' or 'lora_rank_proj'."
        )

    t_prob = F.softmax(t_flat / temperature, dim=-1)
    s_log_prob = F.log_softmax(s_flat / temperature, dim=-1)
    kl = F.kl_div(s_log_prob, t_prob, reduction='none').sum(dim=-1).mean()
    return kl * (temperature ** 2)


def _lora_projection_kl_loss(
    student_projs_dict: dict,
    teacher_projs_dict: dict,
    temperature: float = 4.0,
    projection_mode: str = "lora_spatial_proj",
    anchor_match: bool = True,
) -> torch.Tensor:
    """
    LoRA-Projection Alignment Distillation Loss (AdaLD, arXiv:2509.01750).

    Instead of MSE on raw projections h = A·x, compute KL divergence on
    softmax(h/T) along the spatial dimension.  This is scale-invariant and
    only requires the *relative activation pattern* to match, not the
    absolute magnitude — which differs fundamentally between architectures
    with different input channel counts.

    `lora_spatial_proj` collapses rank into spatial energy maps before KL.
    `lora_rank_proj` keeps per-rank KL and skips mismatched ranks.
    """
    if not student_projs_dict or not teacher_projs_dict:
        return _projection_zero_like(student_projs_dict)

    losses = []

    with torch.amp.autocast('cuda', enabled=False):
        for layer_idx, s_list in student_projs_dict.items():
            if layer_idx not in teacher_projs_dict:
                continue

            t_list = teacher_projs_dict[layer_idx]
            len_s = len(s_list)
            len_t = len(t_list)

            for i_s in range(len_s):
                i_t = _matched_teacher_index(
                    i_s, len_s, len_t, anchor_match=anchor_match
                )
                loss_layer = _projection_kl_loss_pair(
                    s_list[i_s],
                    t_list[i_t],
                    temperature=temperature,
                    projection_mode=projection_mode,
                )
                if loss_layer is not None and not torch.isnan(loss_layer):
                    losses.append(loss_layer)

    if not losses:
        return _projection_zero_like(student_projs_dict)
    return torch.stack(losses).mean()




# ─────────────────────────────────────────────────────────────────────────────
#  KDDetectionTrainer
# ─────────────────────────────────────────────────────────────────────────────

class KDDetectionTrainer(DetectionTrainer):
    """
    Extend Ultralytics DetectionTrainer với Knowledge Distillation đầy đủ (Eq. 37).

    Teacher: YOLOv12-Large (~40M params), frozen, eval mode.
    Student: yolo12n với LoRA injection.

    Loss:
        L_total = stu_weight * L_stu + scale * (L_KL + L_box + L_projection)

    Gateway KD chooses a detached scale so the KD term contributes the
    configured fraction of the weighted supervised loss.
    """

    def __init__(self, overrides=None, _callbacks=None, cached_optimizer_state=None):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.teacher_model: Optional[nn.Module] = None
        self.student_wrapper = None
        self.cached_optimizer_state = cached_optimizer_state
        self.kd_temperature: float = 4.0
        self.kd_lambda: float = 1.0
        self.kd_balance_by_supervised: bool = False
        self.kd_balance_scale_min: float = 0.001
        self.kd_balance_scale_max: float = 4.0
        self.kd_cls_weight: float = 0.45
        self.kd_box_weight: float = 0.35
        self.kd_proj_weight: float = 0.0
        self.kd_conf_threshold: float = 0.10
        self.kd_conf_gamma: float = 2.0
        self.kd_dfl_weight: float = 1.0
        self.kd_ciou_weight: float = 0.5
        self.kd_proj_mode: str = "lora_spatial_proj"
        self.kd_proj_anchor_match: bool = True
        self.logit_kd_only: bool = False
        self.logit_box_kd_only: bool = False
        self.logit_proj_kd_only: bool = False
        
        # Accumulators for logging KD loss
        self.epoch_box_loss = 0.0
        self.epoch_kl_loss = 0.0
        self.epoch_lora_loss = 0.0
        self.epoch_kd_loss = 0.0
        self.epoch_stu_loss = 0.0
        self.epoch_kd_only_loss = 0.0
        self.epoch_kd_scale = 0.0
        self.epoch_kd_ratio = 0.0
        self.epoch_cls_contrib = 0.0
        self.epoch_box_contrib = 0.0
        self.epoch_proj_contrib = 0.0
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
                mean_kd_scale = trainer.epoch_kd_scale / trainer.batch_count
                mean_kd_ratio = trainer.epoch_kd_ratio / trainer.batch_count
                mean_cls_contrib = trainer.epoch_cls_contrib / trainer.batch_count
                mean_box_contrib = trainer.epoch_box_contrib / trainer.batch_count
                mean_proj_contrib = trainer.epoch_proj_contrib / trainer.batch_count
                mean_weighted = trainer.epoch_kd_loss / trainer.batch_count

                print(
                    f"[KD Epoch {trainer.epoch + 1}] "
                    f"Supervised: {mean_stu:.4f} | "
                    f"KD Only: {mean_kd_only:.4f} | "
                    f"Box: {mean_box:.4f} | "
                    f"KL: {mean_kl:.4f} | "
                    f"LoRA_Proj: {mean_lora:.4f} | "
                    f"KD Scale: {mean_kd_scale:.4f} | "
                    f"KD/Sup: {mean_kd_ratio:.3f} | "
                    f"Contrib C/B/P: {mean_cls_contrib:.4f}/"
                    f"{mean_box_contrib:.4f}/{mean_proj_contrib:.4f} | "
                    f"Total: {mean_weighted:.4f}"
                )

                trainer.kd_epoch_history.append({
                    'epoch': int(trainer.epoch + 1),
                    'box': float(mean_box),
                    'kl': float(mean_kl),
                    'lora_proj': float(mean_lora),
                    'stu_loss': float(mean_stu),
                    'kd_scale': float(mean_kd_scale),
                    'kd_ratio': float(mean_kd_ratio),
                    'kd_contrib': float(mean_kd_only),
                    'kd_only_loss': float(mean_kd_only),
                    'kd_loss': float(mean_weighted),
                    'cls_contrib': float(mean_cls_contrib),
                    'box_contrib': float(mean_box_contrib),
                    'proj_contrib': float(mean_proj_contrib),
                    'total': float(mean_weighted),
                    'weighted': float(mean_weighted),
                    'batches': int(trainer.batch_count),
                })
                trainer.epoch_box_loss = 0.0
                trainer.epoch_kl_loss = 0.0
                trainer.epoch_lora_loss = 0.0
                trainer.epoch_kd_loss = 0.0
                trainer.epoch_stu_loss = 0.0
                trainer.epoch_kd_only_loss = 0.0
                trainer.epoch_kd_scale = 0.0
                trainer.epoch_kd_ratio = 0.0
                trainer.epoch_cls_contrib = 0.0
                trainer.epoch_box_contrib = 0.0
                trainer.epoch_proj_contrib = 0.0
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

        self.accumulate = 1

        from detection_2d.trainer import _fl_prepare_model_for_train
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

        # Restore optimizer state AFTER the injected model is placed so that
        # id(param) mapping is correct (optimizer was built against the injected model).
        if self.cached_optimizer_state:
            print(f"[OptimState KD] Warm-start cache detected: {len(self.cached_optimizer_state)} tensors.")
            self._restore_optimizer_state(self.cached_optimizer_state)
        else:
            print("[OptimState KD] Cold-start optimizer (no cache from previous FL round).")

        if self.teacher_model is not None:
            dev = self.device
            if isinstance(dev, str):
                dev = torch.device(dev)
            self.teacher_model.to(dev)

        # Register hooks before the first training forward.
        use_projection_hooks = (
            not getattr(self, 'logit_kd_only', False)
            and not getattr(self, 'logit_box_kd_only', False)
            and max(float(getattr(self, 'kd_proj_weight', 0.0)), 0.0) > 0.0
        )
        if use_projection_hooks:
            if getattr(self, '_s_hooks_registered', False) or getattr(self, '_t_hooks_registered', False):
                self.cleanup_kd_hooks()
            self._s_hook, self._s_handles = _register_lora_hooks(self.model)
            self._s_hooks_registered = True
            if self.teacher_model is not None:
                self._t_hook, self._t_handles = _register_lora_hooks(self.teacher_model)
                self._t_hooks_registered = True

    def cleanup_kd_hooks(self):
        """Remove KD hooks and release any feature tensors they still own."""
        if getattr(self, '_s_hooks_registered', False):
            for handle in self._s_handles: handle.remove()
            self._s_hook.clear()
            self._s_hooks_registered = False
        if getattr(self, '_t_hooks_registered', False):
            for handle in self._t_handles: handle.remove()
            self._t_hook.clear()
            self._t_hooks_registered = False

    def _restore_optimizer_state(self, named_state: dict):
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}
        all_params = [p for g in self.optimizer.param_groups for p in g['params']]
        restored = 0
        for param in all_params:
            name = id_to_name.get(id(param))
            if name is None or name not in named_state: continue
            cached = named_state[name]
            self.optimizer.state[param] = {
                k: v.clone().to(param.device) if isinstance(v, torch.Tensor) else v
                for k, v in cached.items()
            }
            restored += 1
        if restored: print(f"[OptimState KD] Restored {restored} param states.")

    def get_named_optimizer_state(self) -> dict:
        id_to_name = {id(p): n for n, p in self.model.named_parameters()}
        named_state = {}
        for param, state in self.optimizer.state.items():
            name = id_to_name.get(id(param))
            if name is None: continue
            named_state[name] = {
                k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in state.items()
            }
        return named_state


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
        lora_lr_multiplier = getattr(self, 'lora_lr_multiplier', 1.0)
        if head_lr_multiplier != 1.0 or lora_lr_multiplier != 1.0:
            id_to_name = {id(p): n for n, p in model.named_parameters()}
            head_patterns = ('model.21.', 'model.22.', 'model.23.')
            new_groups = []
            for group in optimizer.param_groups:
                head_p = [p for p in group['params']
                          if any(h in id_to_name.get(id(p), '') for h in head_patterns)
                          and 'lora_' not in id_to_name.get(id(p), '')]
                head_p_ids = set(id(x) for x in head_p)
                
                lora_p = [p for p in group['params']
                          if 'lora_' in id_to_name.get(id(p), '') and id(p) not in head_p_ids]
                lora_p_ids = set(id(x) for x in lora_p)

                other_p = [p for p in group['params'] 
                           if id(p) not in head_p_ids and id(p) not in lora_p_ids]
                           
                if other_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = other_p
                    new_groups.append(g)
                if head_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = head_p
                    g['lr'] = group.get('lr', lr) * head_lr_multiplier
                    new_groups.append(g)
                if lora_p:
                    g = {k: v for k, v in group.items() if k != 'params'}
                    g['params'] = lora_p
                    g['lr'] = group.get('lr', lr) * lora_lr_multiplier
                    new_groups.append(g)
            optimizer.param_groups = new_groups
            print(f"[KD-DiffLR] Multipliers: Head ×{head_lr_multiplier}, LoRA ×{lora_lr_multiplier} → "
                  f"Base lr={lr:.2e} | LoRA lr={lr * lora_lr_multiplier:.2e} | Head lr={lr * head_lr_multiplier:.2e}")

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
        self.cleanup_kd_hooks()
            
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
        if self.teacher_model is not None:
            # [FIX] Khóa chặt Teacher ở eval mode và tắt gradient
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

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
        use_projection_kd = (
            not getattr(self, 'logit_kd_only', False)
            and not getattr(self, 'logit_box_kd_only', False)
            and max(float(getattr(self, 'kd_proj_weight', 0.0)), 0.0) > 0.0
        )
        use_box_kd = (
            not getattr(self, 'logit_kd_only', False)
            and not getattr(self, 'logit_proj_kd_only', False)
        )

        # ── 2. Thu hidden features của Student (trong pha forward hiện tại) ─
        if use_projection_kd:
            if not getattr(self, '_s_hooks_registered', False):
                self._s_hook, self._s_handles = _register_lora_hooks(self.model)
                self._s_hooks_registered = True
                raise RuntimeError("Student KD hooks were registered too late for the current batch")
            
            student_projs = dict(self._s_hook.outputs)
            self._s_hook.clear()
            if self.batch_count == 0:
                total_s = sum(len(v) for v in student_projs.values())
                print(f"[KD - Batch 1] Student captured {total_s} LoRA projections across {len(student_projs)} stages.")
        else:
            student_projs = {}

        # ── 3. Forward Teacher (no gradient) + thu features ──────────────
        if use_projection_kd:
            if not getattr(self, '_t_hooks_registered', False):
                self._t_hook, self._t_handles = _register_lora_hooks(self.teacher_model)
                self._t_hooks_registered = True
                print(f"[KD] Đã gắn Hook vĩnh viễn cho Teacher ({self.teacher_model.__class__.__name__}).")
            
            with torch.no_grad():
                self.teacher_model.eval()  # [FIX] Ép eval mode trước mỗi forward để chống Batch Norm drift
                t_preds = self.teacher_model(imgs)

            teacher_projs = dict(self._t_hook.outputs)
            self._t_hook.clear()
            if self.batch_count == 0:
                total_t = sum(len(v) for v in teacher_projs.values())
                print(f"[KD - Batch 1] Teacher captured {total_t} LoRA projections across {len(teacher_projs)} stages.")
        else:
            teacher_projs = {}
            with torch.no_grad():
                self.teacher_model.eval()  # [FIX] Ép eval mode trước mỗi forward để chống Batch Norm drift
                t_preds = self.teacher_model(imgs)

        # ── 4. KL Divergence trên soft logits ────────────────────────────
        if self.batch_count == 0:
            print("[KD - Batch 1] Đang tính toán Distillation Loss (KL/Hidden/Attn)...")
        temperature = max(float(getattr(self, 'kd_temperature', 4.0)), 1e-6)
        anchor_weight = None
        try:
            def _extract_cls(p):
                # 1. Handle YOLOv11 new dict format
                train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
                if isinstance(train_out, dict):
                    if 'scores' in train_out: return train_out['scores']
                    elif 'cls' in train_out: return train_out['cls']
                    elif 'pred_scores' in train_out: return train_out['pred_scores']
                
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
                # Teacher confidence suppresses the dominant background anchors
                # so KD focuses on locations carrying object evidence.
                teacher_confidence = torch.sigmoid(t_cls).detach()
                t_prob = torch.sigmoid(t_cls / temperature).detach()

                anchor_confidence = teacher_confidence.max(dim=1, keepdim=True)[0]
                confidence_threshold = float(getattr(self, 'kd_conf_threshold', 0.10))
                confidence_gamma = float(getattr(self, 'kd_conf_gamma', 2.0))
                anchor_weight = (
                    (anchor_confidence > confidence_threshold).to(anchor_confidence.dtype)
                    * anchor_confidence.pow(confidence_gamma)
                )
                weight_sum = anchor_weight.sum().clamp_min(1e-8)
                
                # True KL Divergence cho Sigmoid: BCE(s, t) - H(t)
                # Vì BCEWithLogits tính cả Entropy của t_prob, ta phải trừ đi để KL min = 0.
                bce_loss = F.binary_cross_entropy_with_logits(
                    s_cls / temperature,
                    t_prob,
                    reduction='none',
                )
                
                # Tính Entropy của Teacher: H(t) = BCE(t_prob, t_prob)
                # Dùng BCEWithLogits thay vì BCE để an toàn với AMP (autocast)
                teacher_entropy = F.binary_cross_entropy_with_logits(
                    t_cls / temperature,
                    t_prob,
                    reduction='none',
                )
                
                loss_kl_unreduced = (bce_loss - teacher_entropy) * (temperature ** 2)
                
                # Tránh lỗi sai số dấu phẩy động làm KL bị âm
                loss_kl_unreduced = loss_kl_unreduced.clamp_min(0.0)
                
                loss_kl = (
                    (loss_kl_unreduced * anchor_weight).sum()
                    / (weight_sum * s_cls.shape[1])
                )
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
        loss_box_kd = torch.tensor(0.0, device=loss_stu.device)
        if use_box_kd:
            try:
                def _extract_bboxes_and_feats(p):
                    # 1. Handle YOLOv11 new dict format
                    train_out = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], dict)) else p
                    if isinstance(train_out, dict):
                        if 'bboxes' in train_out:
                            return train_out['bboxes'], train_out.get('feats')
                        elif 'pred_bboxes' in train_out:
                            return train_out['pred_bboxes'], train_out.get('feats')
                        elif 'boxes' in train_out:
                            return train_out['boxes'], train_out.get('feats')
                    
                    # 2. Handle older list format
                    feats = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], list)) else p
                    if isinstance(feats, list) and len(feats) > 0:
                        cat = torch.cat([xi.view(xi.shape[0], xi.shape[1], -1) for xi in feats], 2)
                        reg_max = 16
                        return cat[:, :reg_max * 4, :], feats
                    
                    # 3. Handle inference tensor format
                    if isinstance(p, torch.Tensor):
                        return p[:, :4, :], None
                    
                    return None, None

                s_box, s_feats = _extract_bboxes_and_feats(preds)
                t_box, _ = _extract_bboxes_and_feats(t_preds)

                if (
                    s_box is not None
                    and t_box is not None
                    and s_box.shape == t_box.shape
                    and s_box.shape[1] > 4
                    and s_box.shape[1] % 4 == 0
                    and s_feats is not None
                    and anchor_weight is not None
                ):
                    batch_size, channels, anchors = s_box.shape
                    reg_max = channels // 4
                    s_dfl = s_box.permute(0, 2, 1).reshape(batch_size, anchors, 4, reg_max)
                    t_dfl = t_box.detach().permute(0, 2, 1).reshape(batch_size, anchors, 4, reg_max)
                    box_anchor_weight = anchor_weight.squeeze(1)
                    box_weight_sum = box_anchor_weight.sum().clamp_min(1e-8)

                    t_dfl_prob = F.softmax(t_dfl, dim=-1)
                    dfl_kl = F.kl_div(
                        F.log_softmax(s_dfl, dim=-1),
                        t_dfl_prob,
                        reduction='none',
                    ).sum(dim=-1)
                    loss_dfl_kd = (
                        (dfl_kl * box_anchor_weight.unsqueeze(-1)).sum()
                        / (box_weight_sum * 4.0)
                    )

                    anchor_points, _ = make_anchors(
                        s_feats,
                        self.original_criterion.stride,
                        0.5,
                    )
                    projection = torch.arange(
                        reg_max,
                        device=s_dfl.device,
                        dtype=s_dfl.dtype,
                    )
                    s_distance = F.softmax(s_dfl, dim=-1).matmul(projection)
                    t_distance = t_dfl_prob.matmul(projection)
                    s_decoded = dist2bbox(s_distance, anchor_points, xywh=False)
                    t_decoded = dist2bbox(t_distance, anchor_points, xywh=False)
                    ciou = bbox_iou(
                        s_decoded,
                        t_decoded.detach(),
                        xywh=False,
                        CIoU=True,
                    ).squeeze(-1)
                    loss_ciou_kd = (
                        ((1.0 - ciou) * box_anchor_weight).sum()
                        / box_weight_sum
                    )

                    loss_box_kd = (
                        loss_dfl_kd
                        + float(getattr(self, 'kd_ciou_weight', 0.5)) * loss_ciou_kd
                    )
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
        loss_lora_proj = torch.tensor(0.0, device=loss_stu.device)
        if use_projection_kd:
            loss_lora_proj = _lora_projection_kl_loss(
                student_projs,
                teacher_projs,
                temperature=temperature,
                projection_mode=getattr(self, 'kd_proj_mode', 'lora_spatial_proj'),
                anchor_match=getattr(self, 'kd_proj_anchor_match', True),
            ).to(loss_stu.device)

        # Keep ground-truth supervision as the reference scale. Gateway KD
        # interprets kd_lambda as the target KD/supervised contribution ratio.
        stu_weight = getattr(self, 'stu_lambda', 0.5)
        total_loss = loss_stu.clone() * stu_weight

        component_values = {
            'cls': loss_kl,
            'box': loss_box_kd,
            'proj': loss_lora_proj,
        }
        component_weights = {
            'cls': max(float(getattr(self, 'kd_cls_weight', 0.45)), 0.0),
            'box': max(float(getattr(self, 'kd_box_weight', 0.35)), 0.0),
            'proj': max(float(getattr(self, 'kd_proj_weight', 0.20)), 0.0),
        }
        (
            supervised_weighted,
            weighted_components,
            weighted_kd,
            kd_scale,
            actual_kd_ratio,
        ) = _compose_balanced_kd(
            loss_stu=loss_stu,
            stu_weight=stu_weight,
            component_values=component_values,
            component_weights=component_weights,
            kd_lambda=self.kd_lambda,
            balance_by_supervised=getattr(self, 'kd_balance_by_supervised', False),
            scale_min=self.kd_balance_scale_min,
            scale_max=self.kd_balance_scale_max,
        )
        
        if total_loss.ndim == 0:
            total_loss = total_loss + weighted_kd
        else:
            total_loss[0] = total_loss[0] + weighted_kd

        self.epoch_box_loss += loss_box_kd.item()
        self.epoch_kl_loss += loss_kl.item()
        self.epoch_lora_loss += loss_lora_proj.item()
        self.epoch_stu_loss += supervised_weighted.item()
        self.epoch_kd_only_loss += weighted_kd.item()
        self.epoch_kd_scale += kd_scale.item()
        self.epoch_kd_ratio += actual_kd_ratio.item()
        self.epoch_cls_contrib += weighted_components['cls'].item()
        self.epoch_box_contrib += weighted_components['box'].item()
        self.epoch_proj_contrib += weighted_components['proj'].item()
        self.epoch_kd_loss += total_loss.item() if total_loss.ndim == 0 else total_loss.sum().item()
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
                'kd_scale': 0.0,
                'kd_ratio': 0.0,
                'kd_contrib': 0.0,
                'kd_total': 0.0,
                'kd_cls_contrib': 0.0,
                'kd_box_contrib': 0.0,
                'kd_proj_contrib': 0.0,
                'kd_weighted': 0.0,
            }

        n = float(len(self.kd_epoch_history))
        return {
            'kd_active': True,
            'kd_epochs': int(len(self.kd_epoch_history)),
            'kd_box': float(sum(e['box'] for e in self.kd_epoch_history) / n),
            'kd_kl': float(sum(e['kl'] for e in self.kd_epoch_history) / n),
            'kd_lora': float(sum(e['lora_proj'] for e in self.kd_epoch_history) / n),
            'kd_stu_loss': float(sum(e['stu_loss'] for e in self.kd_epoch_history) / n),
            'kd_scale': float(sum(e['kd_scale'] for e in self.kd_epoch_history) / n),
            'kd_ratio': float(sum(e['kd_ratio'] for e in self.kd_epoch_history) / n),
            'kd_contrib': float(sum(e['kd_only_loss'] for e in self.kd_epoch_history) / n),
            'kd_total': float(sum(e['kd_loss'] for e in self.kd_epoch_history) / n),
            'kd_cls_contrib': float(sum(e['cls_contrib'] for e in self.kd_epoch_history) / n),
            'kd_box_contrib': float(sum(e['box_contrib'] for e in self.kd_epoch_history) / n),
            'kd_proj_contrib': float(sum(e['proj_contrib'] for e in self.kd_epoch_history) / n),
            # Backward-compatible alias: historical logs used kd_weighted for
            # the complete KD training objective rather than KD contribution.
            'kd_weighted': float(sum(e['weighted'] for e in self.kd_epoch_history) / n),
            'kd_last_epoch': self.kd_epoch_history[-1],
            'kd_epoch_history': list(self.kd_epoch_history),
        }
