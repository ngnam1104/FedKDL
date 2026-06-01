"""
verify_fl_kdl_full.py
Kiểm thử end-to-end một vòng FedKDL — bám Simulator2D / FEDKDL_PIPELINE_REPORT.md.

Load môi trường thật từ topo + data partition (.pkl), không tự chia AUV.
Luồng: Tier1 (local_sgd + INT8) → Tier2 (svd_lora per relay/cluster) → Tier3 (fedavg + KD + eval).
"""
from __future__ import annotations

import argparse
import copy
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import datetime

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

log_filename = f"verify_fl_kdl_full_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = TeeLogger(log_filename)
print(f"Log will be saved to: {log_filename}")

from config.settings import fed_cfg
from federated_core.aggregator import fedavg_global, svd_lora_aggregate
from tasks.detection_2d.knowledge_compression.int8_quantization import pack_payload, unpack_payload
from tasks.detection_2d.models.yolo_wrapper import StudentModel, TeacherModel
from tasks.detection_2d.trainer import evaluate_od, local_sgd_od
from utils.env_manager import EnvironmentManager
from verify_lora import check_lora_injection

# ── Defaults (server paths) ───────────────────────────────────────────────────
DEFAULT_TEACHER_CKPT = Path("/data/nam_nh225051_hust/FedKDL/yolo12l_lora_pretrained.pt")
DEFAULT_STUDENT_CKPT = Path("/data/nam_nh225051_hust/FedKDL/yolo12n_warmup.pt")
DEFAULT_TOPO_PKL = Path(
    "/data/nam_nh225051_hust/FedKDL/environments/2d/topo/N_50/topo_N50_seed1104.pkl"
)
DEFAULT_DATA_PKL = Path(
    "/data/nam_nh225051_hust/FedKDL/environments/2d/data/URPC/N_50/"
    "data_N50_URPC_a0p5_seed1104.pkl"
)
DEFAULT_TEST_YAML = REPO_ROOT / "datasets" / "URPC2020.yaml"


@dataclass
class FedKDLEnv:
    """Môi trường đã load — mirror logic Simulator2D.__init__ (phần data)."""
    topo_path: str
    data_path: str
    N: int
    M: int
    alpha: float
    seed: int
    clusters: Dict[int, List[int]]
    association: Dict[int, int]
    auv_yamls: Dict[int, str]
    n_samples: Dict[int, int]
    test_yaml: str
    proxy_kd_yaml: str


def _resolve_path(path: Path, repo_relative: Optional[str] = None) -> Path:
    if path.exists():
        return path
    if repo_relative:
        local = REPO_ROOT / repo_relative
        if local.exists():
            print(f"[Path] Không thấy {path} — dùng {local}")
            return local
    return path


def _resolve_ckpt(path: Path, fallback_name: str) -> str:
    p = _resolve_path(path, fallback_name)
    if p.exists():
        return str(p)
    print(f"[Path] Cảnh báo: không thấy checkpoint {path}")
    return str(path)


def _collect_all_images(base_yaml: Path) -> List[str]:
    """Giống Simulator2D — thu danh sách ảnh train URPC."""
    with open(base_yaml, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    train_path = base_cfg.get("train", "")
    if isinstance(train_path, str) and train_path.endswith(".txt"):
        with open(train_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    dataset_dir = base_yaml.parent
    original_path = base_cfg.get("path", "")
    img_dir_candidates = [
        dataset_dir / original_path / train_path,
        dataset_dir / str(original_path).split("/")[0] / train_path,
        dataset_dir / base_yaml.stem / train_path,
    ]
    img_dir = next((c for c in img_dir_candidates if c.exists() and c.is_dir()), None)
    if img_dir is None:
        for p in dataset_dir.glob(f"**/{train_path}"):
            if p.is_dir():
                img_dir = p
                break
    if img_dir is None or not img_dir.is_dir():
        print(f"  [warn] Không tìm thấy thư mục {train_path} trực tiếp. Thực hiện quét toàn bộ ảnh trong {dataset_dir}...")
        all_images = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
            for p in dataset_dir.rglob(ext):
                if 'train' in p.parts:
                    all_images.append(str(p))
        if not all_images:
            raise FileNotFoundError(f"Không tìm thấy ảnh nào trong dataset (chứa 'train' trong đường dẫn).")
        images = sorted(set(all_images))
    else:
        all_images = []
        for ext in ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG'):
            all_images.extend([str(p) for p in img_dir.glob(f'**/{ext}')])
        images = sorted(set(all_images))

    if not images:
        raise FileNotFoundError(f"Không có ảnh trong {img_dir}")
    return images


def load_fedkdl_environment(
    topo_path: str,
    data_path: str,
    test_yaml_path: str = str(DEFAULT_TEST_YAML),
    max_pool_images: Optional[int] = None,
) -> FedKDLEnv:
    """
    Load topo + data partition pkl, tạo auv_yamls / proxy KD / test yaml
    (cùng logic với tasks/detection_2d/simulator.py).
    """
    topo = EnvironmentManager.load_topology(topo_path)
    data_part = EnvironmentManager.load_data_partition(data_path)

    base_yaml_path = Path(test_yaml_path)
    if not base_yaml_path.is_absolute():
        base_yaml_path = REPO_ROOT / base_yaml_path
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Thiếu {base_yaml_path}")

    with open(base_yaml_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    all_images = _collect_all_images(base_yaml_path)
    full_pool = len(all_images)
    print(f"[Env] URPC pool: {full_pool} ảnh train")

    if max_pool_images is not None and max_pool_images > 0 and full_pool > max_pool_images:
        data_part, all_images = EnvironmentManager.shrink_image_pool(
            data_part, all_images, max_pool=max_pool_images, seed=data_part.seed
        )
        print(
            f"[Env] DRYTEST: {full_pool} → {len(all_images)} ảnh | "
            f"{len(data_part.auv_data_indices)} AUV | "
            f"proxy={len(data_part.public_data_indices or [])}"
        )

    # Proxy KD — public_data_indices từ partition (20% mặc định khi generate)
    proxy_txt = REPO_ROOT / "datasets" / "proxy_kd_train.txt"
    proxy_txt.parent.mkdir(parents=True, exist_ok=True)
    if getattr(data_part, "public_data_indices", None):
        public_images = [all_images[i] for i in data_part.public_data_indices if i < len(all_images)]
    else:
        public_images = all_images
    with open(proxy_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(public_images))
    print(f"[Env] Proxy KD: {len(public_images)} ảnh → {proxy_txt}")

    alpha_str = str(data_part.alpha).replace(".", "p")
    dry_tag = f"_dry{max_pool_images}" if max_pool_images else ""
    temp_dir = REPO_ROOT / f"datasets/URPC2020/auvs_temp_N{topo.N}_a{alpha_str}_s{data_part.seed}{dry_tag}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    abs_data_root = (base_yaml_path.parent / base_cfg.get("path", "")).resolve()
    auv_yamls: Dict[int, str] = {}
    n_samples: Dict[int, int] = {}

    for sid, idx_list in data_part.auv_data_indices.items():
        c_images = [all_images[i] for i in idx_list if i < len(all_images)]
        if not c_images:
            continue
        n_samples[sid] = len(c_images)

        train_txt = temp_dir / f"auv_{sid}_train.txt"
        dummy_val = temp_dir / f"auv_{sid}_val.txt"
        with open(train_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(c_images))
        with open(dummy_val, "w", encoding="utf-8") as f:
            f.write(c_images[0] + "\n")

        c_cfg = base_cfg.copy()
        c_cfg["train"] = str(train_txt.resolve())
        c_cfg["val"] = str(dummy_val.resolve())
        c_cfg["path"] = str(abs_data_root)
        c_cfg["nc"] = base_cfg.get("nc", 4)
        auv_yaml = temp_dir / f"auv_{sid}.yaml"
        with open(auv_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(c_cfg, f, allow_unicode=True)
        auv_yamls[sid] = str(auv_yaml.resolve())

    # Test yaml (URPC val — giống simulator proxy_test.yaml)
    test_cfg = base_cfg.copy()
    original_path = base_cfg.get("path", "")
    if original_path:
        test_cfg["path"] = str((base_yaml_path.parent / original_path).resolve())
    proxy_test = REPO_ROOT / "datasets" / "proxy_test.yaml"
    with open(proxy_test, "w", encoding="utf-8") as f:
        yaml.safe_dump(test_cfg, f, allow_unicode=True)

    # Proxy KD yaml
    proxy_cfg = base_cfg.copy()
    proxy_cfg.pop("path", None)
    proxy_cfg["train"] = str(proxy_txt.resolve())
    if "val" in test_cfg:
        val_p = test_cfg["val"]
        if original_path and isinstance(val_p, str) and not Path(val_p).is_absolute():
            proxy_cfg["val"] = str((base_yaml_path.parent / original_path / val_p).resolve())
        else:
            proxy_cfg["val"] = val_p
    proxy_kd_yaml = REPO_ROOT / "datasets" / "proxy_kd_data.yaml"
    with open(proxy_kd_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(proxy_cfg, f, allow_unicode=True)

    total_train = sum(n_samples.values())
    print(f"[Env] Topo N={topo.N} M={topo.M} seed={topo.seed}")
    print(f"[Env] Data α={data_part.alpha} seed={data_part.seed} | AUV có data: {len(auv_yamls)}")
    print(f"[Env] Tổng mẫu train (partition): {total_train}")
    for rid, members in sorted(topo.clusters.items()):
        active = [m for m in members if m in auv_yamls]
        if active:
            print(f"  Relay {rid}: {len(active)} AUVs — ids={active[:8]}{'...' if len(active) > 8 else ''}")

    return FedKDLEnv(
        topo_path=topo_path,
        data_path=data_path,
        N=topo.N,
        M=topo.M,
        alpha=data_part.alpha,
        seed=data_part.seed,
        clusters={int(k): list(v) for k, v in topo.clusters.items()},
        association=dict(topo.hfl_association),
        auv_yamls=auv_yamls,
        n_samples=n_samples,
        test_yaml=str(proxy_test.resolve()),
        proxy_kd_yaml=str(proxy_kd_yaml.resolve()),
    )


def _merge_trainable_into_full(model: StudentModel, trainable_sd: Dict, template_sd: Dict) -> Dict:
    merged = copy.deepcopy(template_sd)
    for k, v in trainable_sd.items():
        if k in merged:
            merged[k] = v.clone().detach()
    return merged


def _integrity_check(global_sd_init: Dict, curr_sd: Dict, payload_keys: set) -> Tuple[bool, List[str], int]:
    frozen_keys_changed = []
    bn_keys_changed = []
    for k in global_sd_init:
        old_v = global_sd_init[k].float()
        new_v = curr_sd[k].float()
        if torch.allclose(old_v, new_v, atol=1e-7):
            continue
        if k in payload_keys:
            continue
        if any(x in k for x in ("bn", "batch_norm", "running_mean", "running_var", "num_batches_tracked")):
            bn_keys_changed.append(k)
        elif any(k.startswith(h) for h in ("model.21.", "model.22.", "model.23.")):
            pass # Ignore detection head
        else:
            frozen_keys_changed.append(k)
    return len(frozen_keys_changed) == 0, frozen_keys_changed, len(bn_keys_changed)


def _aggregate_by_clusters(
    gateway_state: Dict,
    clusters: Dict[int, List[int]],
    payloads_bytes: Dict[int, bytes],
    n_samples: Dict[int, int],
) -> Dict:
    """Tier 2: svd_lora_aggregate nội cụm → Tier 3: fedavg_global."""
    relay_states: Dict[int, Dict] = {}
    cluster_totals: Dict[int, int] = {}

    for relay_id, members in sorted(clusters.items()):
        active = [sid for sid in members if sid in payloads_bytes]
        if not active:
            print(f"  Relay {relay_id}: không có payload — giữ gateway state")
            relay_states[relay_id] = copy.deepcopy(gateway_state)
            cluster_totals[relay_id] = 0
            continue

        c_updates = [unpack_payload(payloads_bytes[sid], gateway_state) for sid in active]
        total_n = sum(n_samples.get(sid, 0) for sid in active) or 1
        weights = [n_samples.get(sid, 0) / total_n for sid in active]
        relay_states[relay_id] = svd_lora_aggregate(c_updates, weights)
        cluster_totals[relay_id] = sum(n_samples.get(sid, 0) for sid in active)
        print(
            f"  Relay {relay_id}: {len(active)} AUVs, n={cluster_totals[relay_id]}, "
            f"w={[round(w, 3) for w in weights[:4]]}{'...' if len(weights) > 4 else ''}"
        )

    active_relays = [rid for rid in sorted(relay_states) if cluster_totals.get(rid, 0) > 0]
    if not active_relays:
        raise RuntimeError("Không có relay nào nhận payload.")

    states = [relay_states[r] for r in active_relays]
    samples = [cluster_totals[r] for r in active_relays]
    return fedavg_global(states, samples)


def run_fedkdl_round(
    env: FedKDLEnv,
    student_ckpt: str,
    teacher_ckpt: str,
    device: str,
    local_epochs: int,
    local_lr: float,
    kd_epochs: int,
    max_auvs: Optional[int] = None,
) -> None:
    rank = fed_cfg.LORA_RANK
    nc = 4

    auv_ids = sorted(env.auv_yamls.keys())
    if max_auvs is not None and max_auvs > 0:
        auv_ids = auv_ids[:max_auvs]
        print(f"[Verify] Giới hạn {max_auvs} AUV đầu tiên (dev/quick test)")

    print("=" * 80)
    print(" FEDKDL VERIFY — Một vòng FL (env từ .pkl)")
    print("=" * 80)
    print(f"  Topo    : {env.topo_path}")
    print(f"  Data    : {env.data_path}")
    print(f"  Student : {student_ckpt}")
    print(f"  Teacher : {teacher_ckpt}")
    print(f"  AUVs    : {len(auv_ids)}/{len(env.auv_yamls)} | Relays: {env.M}")
    print(f"  Device  : {device} | rank={rank} | local_epochs={local_epochs} | lr={local_lr}")

    # ── Bước 1: Gateway ──────────────────────────────────────────────────────
    print("\n[Bước 1] Tier 3 — Global Student & Teacher")
    global_student = StudentModel(ckpt=student_ckpt, nc=nc, rank=rank, use_lora=True)
    global_teacher = TeacherModel(ckpt=teacher_ckpt, nc=nc)
    global_student.yolo.model.to(device)
    global_teacher.yolo.model.to(device)
    check_lora_injection(global_student.yolo.model, "GATEWAY GLOBAL STUDENT", base_rank=rank)

    gateway_state = global_student.trainable_state_dict()
    payload_keys = set(gateway_state.keys())
    global_sd_init = copy.deepcopy(global_student.yolo.model.state_dict())

    print("\n[Bước 1b] Eval INIT (trước Local SGD)")
    global_student.strip_inference_tensors()
    eval_init = evaluate_od(global_student, env.test_yaml, device)
    print(
        f"  INIT    mAP50={eval_init['mAP50']:.4f} | mAP50-95={eval_init['mAP50-95']:.4f} | "
        f"Prec={eval_init['Prec']:.4f} | Rec={eval_init['Rec']:.4f}"
    )

    # ── Bước 2–3: Tier 1 ─────────────────────────────────────────────────────
    print("\n[Bước 2–3] Tier 1 — Local SGD + Lazy Filter + INT8")
    payloads_bytes: Dict[int, bytes] = {}
    payload_kbs: Dict[int, float] = {}
    skipped_lazy: List[int] = []

    for auv_id in auv_ids:
        auv_yaml = env.auv_yamls[auv_id]
        n_s = env.n_samples.get(auv_id, 0)
        relay_id = env.association.get(auv_id, -1)
        print(f"\n  --- AUV {auv_id} (Relay {relay_id}, n={n_s}) ---")

        local_student = StudentModel(ckpt=student_ckpt, nc=nc, rank=rank, use_lora=True)
        local_student.load_trainable_state_dict(gateway_state)

        new_state, delta_norm, train_loss, _ = local_sgd_od(
            student_model=local_student,
            auv_yaml=auv_yaml,
            auv_id=auv_id,
            epochs=local_epochs,
            batch_size=fed_cfg.LOCAL_BATCH_SIZE,
            lr=local_lr,
            device=device,
        )
        print(f"  loss={train_loss:.4f} | delta_norm={delta_norm:.6f}")

        if delta_norm < fed_cfg.DELTA_SKIP:
            print(f"  💤 Lazy Filter (delta < {fed_cfg.DELTA_SKIP})")
            skipped_lazy.append(auv_id)
            del local_student
            gc.collect()
            continue

        pbytes, pkb = pack_payload(new_state)
        payloads_bytes[auv_id] = pbytes
        payload_kbs[auv_id] = pkb
        print(f"  Payload INT8: {pkb:.1f} KB (target ≤ {fed_cfg.TARGET_PAYLOAD_KB:.0f} KB)")

        del local_student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not payloads_bytes:
        raise RuntimeError("Không có AUV nào gửi payload.")

    # ── Bước 4–5: Tier 2 + Tier 3 global ─────────────────────────────────────
    print("\n[Bước 4–5] Tier 2 (SVD-LoRA per cluster) → Tier 3 (fedavg_global)")
    global_trainable = _aggregate_by_clusters(
        gateway_state, env.clusters, payloads_bytes, env.n_samples
    )
    global_student.load_trainable_state_dict(global_trainable)
    merged_full = _merge_trainable_into_full(global_student, global_trainable, global_sd_init)
    global_student.yolo.model.load_state_dict(merged_full, strict=False)

    # ── Bước 6: Eval + KD ──────────────────────────────────────────────────────
    print("\n[Bước 6a] Eval TRƯỚC KD")
    global_student.strip_inference_tensors()
    eval_pre = evaluate_od(global_student, env.test_yaml, device)
    print(
        f"  PRE-KD  mAP50={eval_pre['mAP50']:.4f} | mAP50-95={eval_pre['mAP50-95']:.4f} | "
        f"Prec={eval_pre['Prec']:.4f} | Rec={eval_pre['Rec']:.4f}"
    )

    print("\n[Bước 6b] Gateway KD (KDDetectionTrainer)")
    from tasks.detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer

    global_student.strip_inference_tensors()
    global_student.load_trainable_state_dict(global_trainable)

    kd_overrides = {
        "model": student_ckpt,
        "data": env.proxy_kd_yaml,
        "epochs": kd_epochs,
        "batch": 8,
        "workers": 0,
        "lr0": 2e-4,
        "optimizer": "AdamW",
        "warmup_epochs": 1.0,
        "warmup_bias_lr": 0.0,
        "device": device,
        "project": str(REPO_ROOT / "runs" / "verify_fl_kdl"),
        "name": "gateway_kd",
        "exist_ok": True,
        "verbose": False,
        "val": False,
        "save": False,
        "plots": False,
        "close_mosaic": 0,
        "amp": False,
    }
    trainer = KDDetectionTrainer(overrides=kd_overrides)
    trainer.head_lr_multiplier = 5.0  # Diff LR: LoRA 2e-4, Head 1e-3
    trainer.student_wrapper = global_student
    trainer.kd_lambda = 0.5
    trainer.set_teacher(global_teacher.yolo.model)
    trainer._fl_injected_model = global_student.yolo.model
    trainer.model = global_student.yolo.model
    trainer.train()

    kd_summary = trainer.get_kd_summary() if hasattr(trainer, "get_kd_summary") else {}
    if kd_summary.get("kd_active"):
        print(
            f"  KD | KL={kd_summary.get('kd_kl', 0):.4f} | "
            f"Hidden={kd_summary.get('kd_hidden', 0):.4f} | "
            f"Attn={kd_summary.get('kd_attn', 0):.4f}"
        )

    gateway_state = global_student.trainable_state_dict()
    curr_kd_sd = global_student.yolo.model.state_dict()
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[Bước 6c] Eval SAU KD")
    global_student.strip_inference_tensors()
    global_student.load_trainable_state_dict(gateway_state)
    eval_post = evaluate_od(global_student, env.test_yaml, device)
    print(
        f"  POST-KD mAP50={eval_post['mAP50']:.4f} | mAP50-95={eval_post['mAP50-95']:.4f} | "
        f"Prec={eval_post['Prec']:.4f} | Rec={eval_post['Rec']:.4f}"
    )
    print(f"  ΔmAP50 = {eval_post['mAP50'] - eval_pre['mAP50']:+.4f}")

    # ── Bước 7: Integrity ──────────────────────────────────────────────────────
    print("\n[Bước 7] Kiểm tra toàn vẹn backbone")
    ok, errors, n_bn = _integrity_check(global_sd_init, curr_kd_sd, payload_keys)
    if ok:
        print("  ✅ Backbone Conv không bị thay đổi.")
    else:
        print(f"  ❌ {len(errors)} tensor backbone đổi:")
        for n in errors[:8]:
            print(f"     - {n}")
    print(f"  BatchNorm thay đổi: {n_bn} tensors (OK).")
    check_lora_injection(global_student.yolo.model, "GATEWAY SAU KD", base_rank=rank)

    avg_payload = sum(payload_kbs.values()) / len(payload_kbs)
    print("\n" + "=" * 80)
    print(" TÓM TẮT")
    print("=" * 80)
    print(f"  Payload TX     : {len(payloads_bytes)}/{len(auv_ids)} AUV (lazy: {skipped_lazy or 'none'})")
    print(f"  Avg INT8 payload : {avg_payload:.1f} KB")
    print(f"  INIT/PRE/POST mAP50: {eval_init['mAP50']:.4f} → {eval_pre['mAP50']:.4f} → {eval_post['mAP50']:.4f}")
    print(f"  Backbone OK      : {'YES' if ok else 'NO'}")
    if ok:
        print("\n✨ FEDKDL VERIFY (env .pkl): SUCCESS ✨")
    else:
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Verify FedKDL — load env từ topo/data .pkl")
    p.add_argument("--topo", type=str, default=str(DEFAULT_TOPO_PKL))
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA_PKL))
    p.add_argument("--teacher", type=str, default=str(DEFAULT_TEACHER_CKPT))
    p.add_argument("--student", type=str, default=str(DEFAULT_STUDENT_CKPT))
    p.add_argument("--test-yaml", type=str, default=str(DEFAULT_TEST_YAML))
    p.add_argument("--local-epochs", type=int, default=fed_cfg.LOCAL_EPOCHS)
    p.add_argument("--local-lr", type=float, default=fed_cfg.LOCAL_LR)
    p.add_argument("--kd-epochs", type=int, default=2)
    p.add_argument("--max-auvs", type=int, default=0, help="0 = tất cả AUV trong partition")
    p.add_argument(
        "--drytest",
        action="store_true",
        help="Chỉ dùng 500 ảnh train (subset partition, giữ topo N=50)",
    )
    p.add_argument(
        "--max-pool",
        type=int,
        default=500,
        help="Số ảnh train tối đa khi --drytest (mặc định 500)",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    topo_path = _resolve_path(Path(args.topo), "environments/2d/topo/N_50/topo_N50_seed1104.pkl")
    data_path = _resolve_path(
        Path(args.data),
        "environments/2d/data/URPC/N_50/data_N50_URPC_a0p5_seed1104.pkl",
    )
    if not topo_path.exists():
        raise SystemExit(f"Không tìm thấy topo: {topo_path}")
    if not data_path.exists():
        raise SystemExit(f"Không tìm thấy data partition: {data_path}")

    student_ckpt = _resolve_ckpt(Path(args.student), "yolo12n_warmup.pt")
    if not Path(student_ckpt).exists():
        base_ckpt = _resolve_ckpt(REPO_ROOT / "yolo12n.pt", "yolo12n.pt")
        if Path(base_ckpt).exists():
            print(f"[Path] Không có warmup — dùng base student: {base_ckpt}")
            student_ckpt = base_ckpt
    teacher_ckpt = _resolve_ckpt(Path(args.teacher), "yolo12l_lora_pretrained.pt")

    max_pool = args.max_pool if args.drytest else None
    if args.drytest:
        print(f"[Drytest] Subset partition → tối đa {max_pool} ảnh train (topo giữ nguyên)")

    env = load_fedkdl_environment(
        topo_path=str(topo_path),
        data_path=str(data_path),
        test_yaml_path=args.test_yaml,
        max_pool_images=max_pool,
    )

    max_auvs = args.max_auvs if args.max_auvs > 0 else None
    run_fedkdl_round(
        env=env,
        student_ckpt=student_ckpt,
        teacher_ckpt=teacher_ckpt,
        device=device,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        kd_epochs=args.kd_epochs,
        max_auvs=max_auvs,
    )


if __name__ == "__main__":
    main()
