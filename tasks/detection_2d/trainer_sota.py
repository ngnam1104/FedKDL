"""
trainer.py — SOTA Baseline (Jiang et al., 2025)
Local SGD + Local KD (Teacher chạy TẠI AUV, không phải Gateway).
Kết hợp Dark Channel Prior (DCP) để tiền xử lý ảnh.

Điểm khác biệt cốt lõi so với FedKDL (tasks/detection_2d/trainer.py):
  - Teacher YOLO12l được tải và chạy trực tiếp tại từng AUV AUV.
  - DCP được áp dụng lên ảnh input trước khi đưa vào YOLO.
  - Student train toàn bộ tham số (full_param=True) — không dùng LoRA.
  - Payload truyền lên Relay là toàn bộ model Float32 (~5.4 MB), không nén INT8.
"""
import torch
import copy
import os
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import strip_optimizer
from config.settings import fed_cfg


# ─────────────────────────────────────────────────────────────────────────────
#  DCP Augmented Dataset: Monkey-patch YOLO to apply DCP on-the-fly
# ─────────────────────────────────────────────────────────────────────────────

def _patch_dataloader_with_dcp(trainer_instance):
    """
    Monkey-patch YOLO trainer's dataset to apply DCP on each image.
    Override __getitem__ để chèn DCP vào pipeline đọc ảnh.
    """
    from tasks.detection_2d.knowledge_compression.dcp import apply_dcp_to_image_array

    original_get_item = trainer_instance.train_loader.dataset.__class__.__getitem__

    def _dcp_getitem(self, index):
        item = original_get_item(self, index)
        # item['img'] là tensor [C, H, W] float32 trong [0, 1]
        img = item.get('img', None)
        if img is not None and isinstance(img, torch.Tensor):
            # Chuyển về numpy BGR uint8
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np_bgr = img_np[:, :, ::-1]  # RGB → BGR cho OpenCV/DCP
            try:
                enhanced_bgr = apply_dcp_to_image_array(img_np_bgr)
                # Chuyển lại tensor float32 [C, H, W]
                enhanced_rgb = enhanced_bgr[:, :, ::-1]
                item['img'] = torch.from_numpy(
                    enhanced_rgb.astype(np.float32) / 255.0
                ).permute(2, 0, 1).to(img.device)
            except Exception:
                # Nếu DCP lỗi (ảnh bị hỏng), giữ nguyên ảnh gốc
                pass
        return item

    trainer_instance.train_loader.dataset.__class__.__getitem__ = _dcp_getitem


# ─────────────────────────────────────────────────────────────────────────────
#  Local KD Trainer — Teacher chạy TẠI AUV (Jiang et al., 2025)
# ─────────────────────────────────────────────────────────────────────────────

class LocalKDTrainer(DetectionTrainer):
    """
    SOTA trainer: KD được thực hiện cục bộ tại AUV (không phải Gateway).
    Teacher (YOLO12l) Forward Pass được gọi trong mỗi batch → rất tốn VRAM & CPU.
    """

    def __init__(self, overrides=None, _callbacks=None, teacher_nn=None,
                 student_wrapper=None, kd_temperature: float = 4.0):
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.teacher_nn = teacher_nn        # nn.Module của Teacher (frozen)
        self.student_wrapper = student_wrapper
        self.kd_temperature = kd_temperature
        self._use_dcp = True                # Cờ bật DCP

    def _setup_train(self):
        from ultralytics.utils import LOGGER
        orig_warn = LOGGER.warning
        LOGGER.warning = lambda *a, **k: None
        try:
            super()._setup_train()
        finally:
            LOGGER.warning = orig_warn

        # [SOTA] Sau khi setup xong, monkey-patch DCP vào dataloader
        if self._use_dcp:
            try:
                _patch_dataloader_with_dcp(self)
                print("[SOTA][DCP] Dark Channel Prior đã được nhúng vào DataLoader.")
            except Exception as e:
                print(f"[SOTA][DCP] Không thể nhúng DCP: {e}. Bỏ qua DCP.")

        # Override criterion
        from ultralytics.utils.torch_utils import unwrap_model
        model_unwrapped = unwrap_model(self.model)
        if getattr(model_unwrapped, "criterion", None) is None:
            model_unwrapped.criterion = model_unwrapped.init_criterion()
        if not hasattr(self, '_original_criterion'):
            self._original_criterion = model_unwrapped.criterion
            model_unwrapped.criterion = self._local_kd_criterion

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        # [SOTA] Train TOÀN BỘ tham số — không lọc LoRA
        return optimizer

    def validate(self):
        return {}, 0.0

    def final_eval(self):
        from ultralytics.utils.torch_utils import strip_optimizer, unwrap_model
        model_unwrapped = unwrap_model(self.model)
        if hasattr(self, '_original_criterion'):
            model_unwrapped.criterion = self._original_criterion
        if self.last.exists():
            strip_optimizer(self.last)

    def _local_kd_criterion(self, preds, batch):
        """
        Local KD Loss theo Jiang et al. (2025) Eq. 9:
        L = (1 - alpha) * L_CE(Student, GT) + alpha * T^2 * KL(Student || Teacher)
        Ở đây ta cài đặt alpha=0.5, T=4 (theo bài báo).
        """
        import torch.nn.functional as F

        # 1. Student task loss (CE + box + dfl)
        loss_stu, loss_items = self._original_criterion(preds, batch)

        if self.teacher_nn is None:
            return loss_stu, loss_items

        imgs = batch['img']
        T = self.kd_temperature
        alpha = 0.5  # Jiang et al. dùng alpha=0.5 cho trọng số KD

        # 2. Teacher forward (no grad)
        with torch.no_grad():
            t_preds = self.teacher_nn(imgs)

        # 3. KL Divergence trên soft logits (Softmax KD cơ bản — Eq. 9)
        try:
            def _extract_cls(p):
                feats = p[1] if (isinstance(p, tuple) and len(p) > 1 and isinstance(p[1], list)) else p
                if isinstance(feats, list) and len(feats) > 0:
                    cat = torch.cat([xi.view(xi.shape[0], xi.shape[1], -1) for xi in feats], 2)
                    return cat[:, 16 * 4:, :]
                if isinstance(p, torch.Tensor):
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
                loss_kl = torch.tensor(0.0, device=loss_stu.device if loss_stu.ndim > 0 else 'cpu')
        except Exception:
            loss_kl = torch.tensor(0.0)

        # 4. Tổng loss theo Eq. 9
        kd_loss = alpha * loss_kl
        total_loss = loss_stu.clone()
        if total_loss.ndim == 0:
            total_loss = (1 - alpha) * total_loss + kd_loss
        else:
            total_loss[0] = (1 - alpha) * total_loss[0] + kd_loss

        return total_loss, loss_items


# ─────────────────────────────────────────────────────────────────────────────
#  local_sgd_od_sota — hàm huấn luyện cục bộ chính cho SOTA baseline
# ─────────────────────────────────────────────────────────────────────────────

def local_sgd_od_sota(
    student_model,
    teacher_model,          # TeacherModel wrapper — chạy tại AUV
    auv_yaml: str,
    auv_id: int,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str = "cpu",
    use_dcp: bool = True,
) -> tuple:
    """
    Local SGD cho SOTA baseline (Jiang et al., 2025).
    KHÁC với FedKDL:
      - Teacher chạy TẠI AUV (tốn VRAM + pin).
      - DCP tiền xử lý toàn bộ ảnh train.
      - Student train full model (không LoRA).
      - Payload = full model Float32 dict (~5.4 MB).

    Returns:
        (full_state_dict, train_loss)
    """
    print(
        f"[SOTA][AUV {auv_id}] Local KD + DCP={use_dcp}, epochs={epochs}, lr={lr:.6f}"
    )

    state_before = {k: v.clone() for k, v in student_model.yolo.model.state_dict().items()}

    overrides = {
        'model': "yolov8n.pt",
        'data': auv_yaml,
        'cache': getattr(fed_cfg, 'CACHE_DATASET', True),
        'epochs': epochs,
        'batch': batch_size,
        'workers': getattr(fed_cfg, 'DATALOADER_WORKERS', 4),
        'lr0': lr,
        'optimizer': 'AdamW',
        'warmup_epochs': 0.0,
        'lrf': 1.0,
        'cos_lr': False,
        'device': device,
        'amp': False,  # Vô hiệu hóa FP16
        'project': 'runs/fl_sota_auvs',
        'name': f'auv_{auv_id}',
        'exist_ok': True,
        'verbose': False,
        'save': False,
        'val': False,
        'plots': False,
        'workers': 0,
    }

    teacher_nn = teacher_model.yolo.model if teacher_model is not None else None

    trainer = LocalKDTrainer(
        overrides=overrides,
        teacher_nn=teacher_nn,
        student_wrapper=student_model,
        kd_temperature=4.0,
    )
    trainer._use_dcp = use_dcp
    trainer.model = student_model.yolo.model

    trainer.train()

    # Đọc train loss từ CSV
    train_loss = 0.0
    try:
        csv_path = Path(trainer.save_dir) / 'results.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            last_row = df.iloc[-1]
            train_loss = (float(last_row.get('train/box_loss', 0.0)) +
                          float(last_row.get('train/cls_loss', 0.0)) +
                          float(last_row.get('train/dfl_loss', 0.0)))
    except Exception as e:
        print(f"[SOTA] Không thể đọc results.csv: {e}")

    # [SOTA] Payload = TOÀN BỘ model (không LoRA, không INT8)
    full_state = {k: v.cpu().clone() for k, v in student_model.yolo.model.state_dict().items()}

    return full_state, train_loss


def evaluate_od_sota(student_model, test_yaml: str, device: str = "cpu") -> dict:
    """Đánh giá mAP trên test set — dùng chung với FedKDL."""
    import copy, gc
    unfused_model = copy.deepcopy(student_model.yolo.model)
    torch.cuda.empty_cache()
    gc.collect()
    results = student_model.yolo.val(
        data=test_yaml, device=device, verbose=False,
        split='val', half=False, workers=0, batch=16,
    )
    student_model.yolo.model = unfused_model
    torch.cuda.empty_cache()
    gc.collect()
    mp = float(np.mean(results.box.mp)) if hasattr(results.box, 'mp') else 0.0
    mr = float(np.mean(results.box.mr)) if hasattr(results.box, 'mr') else 0.0
    return {
        'mAP50-95': float(results.box.map),
        'mAP50': float(results.box.map50),
        'Prec': mp,
        'Rec': mr,
    }
