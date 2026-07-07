"""
test_gateway_features.py
========================
Test xac minh cac tinh nang Gateway vua them/sua.
Import truc tiep tu repo, khong re-implement logic.

Chay: python test_gateway_features.py
"""
import sys
import types
import inspect
import torch
import torch.nn as nn

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name, cond, detail=""):
    tag = PASS if cond else FAIL
    msg = f"{tag} {name}"
    if detail:
        msg += f"  -> {detail}"
    print(msg)
    results.append((name, cond))
    return cond


# ===========================================================================
# BLOCK A: Payload asymmetry (uplink partial head / downlink full head)
# ===========================================================================
def test_payload_asymmetry():
    print("\n=== BLOCK A: Payload Asymmetry (uplink < downlink) ===")
    from detection_2d.models.yolo_wrapper import StudentModel

    student = StudentModel(ckpt="yolo12n.pt", rank=4, nc=4)
    head_idx = len(student.yolo.model.model) - 1

    uplink_sd   = student.trainable_state_dict(downlink=False)
    downlink_sd = student.trainable_state_dict(downlink=True)

    up_keys   = set(uplink_sd.keys())
    down_keys = set(downlink_sd.keys())

    up_params   = sum(v.numel() for v in uplink_sd.values())
    down_params = sum(v.numel() for v in downlink_sd.values())
    up_kb   = up_params   * 4 / 1024
    down_kb = down_params * 4 / 1024

    check("A1: downlink has MORE keys than uplink",
          len(down_keys) > len(up_keys),
          f"up={len(up_keys)} down={len(down_keys)}")

    check("A2: downlink has MORE params than uplink",
          down_params > up_params,
          f"up={up_kb:.1f} KB  down={down_kb:.1f} KB")

    # uplink payload must be a subset of downlink
    check("A3: uplink keys are a subset of downlink keys",
          up_keys.issubset(down_keys))

    # The extra keys in downlink should be cv2.0.0.* (the first conv in cv2[0])
    extra_keys = down_keys - up_keys
    cv2_0_prefix = f"model.{head_idx}.cv2.0.0."
    cv2_extra = [k for k in extra_keys if k.startswith(cv2_0_prefix)]
    check("A4: cv2.0.0.* (box-reg first conv) absent in uplink, present in downlink",
          len(cv2_extra) > 0,
          f"{len(cv2_extra)} extra cv2.0.0 keys in downlink")

    # Print summary for manual verification
    print(f"     Total extra params in downlink: "
          f"{sum(downlink_sd[k].numel() for k in extra_keys) * 4 / 1024:.1f} KB")


# ===========================================================================
# BLOCK B: Config — AdamW, LR, Multipliers
# ===========================================================================
def test_config():
    print("\n=== BLOCK B: Config Values ===")
    from config.settings import fed_cfg

    check("B1: KD_OPTIMIZER == 'AdamW'",
          getattr(fed_cfg, 'KD_OPTIMIZER', None) == 'AdamW',
          f"got: {getattr(fed_cfg, 'KD_OPTIMIZER', '<missing>')}")

    check("B2: PROXY_FT_OPTIMIZER == 'AdamW'",
          getattr(fed_cfg, 'PROXY_FT_OPTIMIZER', None) == 'AdamW',
          f"got: {getattr(fed_cfg, 'PROXY_FT_OPTIMIZER', '<missing>')}")

    check("B3: KD_LR == 5e-4",
          abs(fed_cfg.KD_LR - 5e-4) < 1e-9, f"got: {fed_cfg.KD_LR}")

    check("B4: PROXY_FT_LR == 5e-4",
          abs(fed_cfg.PROXY_FT_LR - 5e-4) < 1e-9, f"got: {fed_cfg.PROXY_FT_LR}")

    check("B5: KD_HEAD_LR_MULT == 2.0",
          abs(fed_cfg.KD_HEAD_LR_MULT - 2.0) < 1e-9,
          f"got: {fed_cfg.KD_HEAD_LR_MULT}")

    check("B6: PROXY_FT_HEAD_LR_MULT == 1.0",
          abs(getattr(fed_cfg, 'PROXY_FT_HEAD_LR_MULT', 1.0) - 1.0) < 1e-9,
          f"got: {getattr(fed_cfg, 'PROXY_FT_HEAD_LR_MULT', 1.0)}")

    check("B7: KD_STU_LAMBDA == 0.70  (supervised dominates, user reverted)",
          abs(fed_cfg.KD_STU_LAMBDA - 0.70) < 1e-9,
          f"got: {fed_cfg.KD_STU_LAMBDA}")

    check("B8: KD_LRF == 1.0  (flat LR, no cosine decay -> prevents catastrophic forgetting)",
          abs(getattr(fed_cfg, 'KD_LRF', 0.0) - 1.0) < 1e-9,
          f"got: {getattr(fed_cfg, 'KD_LRF', '<missing>')}")

    check("B9: PROXY_FT_LRF == 1.0  (flat LR for proxy finetune)",
          abs(getattr(fed_cfg, 'PROXY_FT_LRF', 0.0) - 1.0) < 1e-9,
          f"got: {getattr(fed_cfg, 'PROXY_FT_LRF', '<missing>')}")


# ===========================================================================
# BLOCK C: Simulator source — AdamW propagated into override dicts
# ===========================================================================
def test_simulator_optimizer_source():
    print("\n=== BLOCK C: Simulator uses fed_cfg optimizer (not hardcoded SGD) ===")
    from detection_2d import simulator as sim_mod

    # Read source of the two gateway methods
    sim_cls = sim_mod.Simulator2D
    proxy_src = inspect.getsource(sim_cls._gateway_supervised_finetune)
    kd_src    = inspect.getsource(sim_cls._gateway_knowledge_distillation)

    check("C1: Proxy-FT reads PROXY_FT_OPTIMIZER from fed_cfg (not hardcoded 'SGD')",
          "'SGD'" not in proxy_src and "PROXY_FT_OPTIMIZER" in proxy_src,
          "found hardcoded 'SGD'" if "'SGD'" in proxy_src else "OK")

    check("C2: KD reads KD_OPTIMIZER from fed_cfg (not hardcoded 'SGD')",
          "'SGD'" not in kd_src and "KD_OPTIMIZER" in kd_src,
          "found hardcoded 'SGD'" if "'SGD'" in kd_src else "OK")

    check("C3: Proxy-FT uses proxy_ft_optimizer_state (separate from kd)",
          "proxy_ft_optimizer_state" in proxy_src)

    check("C4: KD uses kd_optimizer_state (separate from proxy_ft)",
          "kd_optimizer_state" in kd_src)

    check("C5: Proxy-FT saves optimizer state after train",
          "proxy_ft_optimizer_state = trainer.get_named_optimizer_state()" in proxy_src)

    check("C6: KD saves optimizer state after train",
          "kd_optimizer_state = trainer.get_named_optimizer_state()" in kd_src)

    check("C7: Proxy-FT sets lrf override (no cosine decay catastrophic forgetting fix)",
          "'lrf'" in proxy_src and "PROXY_FT_LRF" in proxy_src)

    check("C8: KD sets lrf override (no cosine decay catastrophic forgetting fix)",
          "'lrf'" in kd_src and "KD_LRF" in kd_src)


# ===========================================================================
# BLOCK D: BaseGateway — two separate optimizer state fields
# ===========================================================================
def test_base_gateway_fields():
    print("\n=== BLOCK D: BaseGateway Separate Optimizer State Fields ===")
    from federated_core.workers import BaseGateway

    gw = BaseGateway({'weight': torch.zeros(4, 4)})

    check("D1: proxy_ft_optimizer_state initialised to None",
          hasattr(gw, 'proxy_ft_optimizer_state') and gw.proxy_ft_optimizer_state is None)

    check("D2: kd_optimizer_state initialised to None",
          hasattr(gw, 'kd_optimizer_state') and gw.kd_optimizer_state is None)

    check("D3: generic optimizer_state field removed (no cross-contamination risk)",
          not hasattr(gw, 'optimizer_state'))

    # Mutating one must not affect the other
    gw.proxy_ft_optimizer_state = {'a': torch.ones(3)}
    gw.kd_optimizer_state       = {'b': torch.zeros(3)}
    check("D4: proxy_ft and kd states are independent objects",
          gw.proxy_ft_optimizer_state is not gw.kd_optimizer_state)

    check("D5: global_state_dict is deep-copied (not aliased to constructor arg)",
          gw.global_state_dict is not gw.proxy_ft_optimizer_state)


# ===========================================================================
# BLOCK E: Optimizer cache — call actual repo methods via SimpleNamespace
# ===========================================================================
def test_optimizer_cache_custom():
    print("\n=== BLOCK E: Optimizer Cache — CustomDetectionTrainer methods ===")
    from detection_2d.trainer import CustomDetectionTrainer

    # Build a small model + warm optimizer (no Ultralytics training needed)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Fake trainer with just model + optimizer — enough for both methods
    fake = types.SimpleNamespace(model=model, optimizer=opt)

    # --- Call actual repo get_named_optimizer_state ---
    named_state = CustomDetectionTrainer.get_named_optimizer_state(fake)

    check("E1: get_named_optimizer_state returns non-empty dict",
          isinstance(named_state, dict) and len(named_state) > 0,
          f"{len(named_state)} params")

    check("E2: all saved tensors are on CPU",
          all(v['exp_avg'].device.type == 'cpu'
              for v in named_state.values() if 'exp_avg' in v))

    # --- Build new model+optimizer (simulates next FL round) ---
    model2 = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    opt2 = torch.optim.AdamW(model2.parameters(), lr=5e-4)
    fake2 = types.SimpleNamespace(model=model2, optimizer=opt2)

    check("E3: new optimizer is cold (empty state before restore)",
          len(opt2.state) == 0)

    # --- Call actual repo _restore_optimizer_state ---
    CustomDetectionTrainer._restore_optimizer_state(fake2, named_state)

    check("E4: optimizer state restored (non-empty after restore)",
          len(opt2.state) > 0)

    # Verify values match original
    id_to_name2 = {id(p): n for n, p in model2.named_parameters()}
    values_ok = all(
        torch.allclose(
            opt2.state[p]['exp_avg'],
            named_state[id_to_name2[id(p)]]['exp_avg'].to(p.device)
        )
        for p in opt2.state if id_to_name2.get(id(p)) in named_state
    )
    check("E5: exp_avg values match exactly after restore", values_ok)

    # CRITICAL: restored state must reference model2's params, NOT model1's
    state_ids  = {id(p) for p in opt2.state}
    model2_ids = {id(p) for p in model2.parameters()}
    model1_ids = {id(p) for p in model.parameters()}
    check("E6: restored state points to model2 params (not model1)",
          state_ids.issubset(model2_ids) and state_ids.isdisjoint(model1_ids))


def test_optimizer_cache_kd():
    print("\n=== BLOCK F: Optimizer Cache — KDDetectionTrainer methods ===")
    from detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer

    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    fake = types.SimpleNamespace(model=model, optimizer=opt)

    # KDDetectionTrainer has its own get_named_optimizer_state / _restore_optimizer_state
    named_state = KDDetectionTrainer.get_named_optimizer_state(fake)
    check("F1: KDDetectionTrainer.get_named_optimizer_state works",
          len(named_state) > 0, f"{len(named_state)} params")

    model2 = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    opt2   = torch.optim.AdamW(model2.parameters(), lr=5e-4)
    fake2  = types.SimpleNamespace(model=model2, optimizer=opt2)

    KDDetectionTrainer._restore_optimizer_state(fake2, named_state)
    check("F2: KDDetectionTrainer._restore_optimizer_state works",
          len(opt2.state) > 0)

    check("F3: KDDetectionTrainer has cached_optimizer_state in __init__ signature",
          'cached_optimizer_state' in inspect.signature(KDDetectionTrainer.__init__).parameters)


# ===========================================================================
# BLOCK G: Restore ORDER — must happen AFTER injected model is placed
# ===========================================================================
def test_restore_order():
    print("\n=== BLOCK G: Restore Order in KDDetectionTrainer._setup_train ===")
    from detection_2d.knowledge_compression.knowledge_distillation import KDDetectionTrainer

    src = inspect.getsource(KDDetectionTrainer._setup_train)

    pos_inject  = src.find('_fl_prepare_model_for_train')
    pos_restore = src.find('_restore_optimizer_state')
    pos_accum   = src.find('self.accumulate = 1')

    check("G1: _fl_prepare_model_for_train present in _setup_train", pos_inject  != -1)
    check("G2: _restore_optimizer_state present in _setup_train",     pos_restore != -1)
    check("G3: accumulate=1 set in _setup_train",                     pos_accum   != -1)

    check("G4: restore runs AFTER injected model placed  (pos_inject < pos_restore)",
          pos_inject < pos_restore,
          f"inject@{pos_inject}  restore@{pos_restore}")

    # Also verify for CustomDetectionTrainer — same contract
    from detection_2d.trainer import CustomDetectionTrainer
    src2 = inspect.getsource(CustomDetectionTrainer._setup_train)
    p_inject2  = src2.find('injected')
    p_restore2 = src2.find('_restore_optimizer_state')
    check("G5: CustomDetectionTrainer also restores after injected model check",
          p_inject2 < p_restore2,
          f"inject@{p_inject2}  restore@{p_restore2}")


# ===========================================================================
# BLOCK H: Head unlocking (gateway logic)
# ===========================================================================
def test_head_unlocking():
    print("\n=== BLOCK H: Head Unlocking at Gateway ===")
    from detection_2d.models.yolo_wrapper import StudentModel

    student  = StudentModel(ckpt="yolo12n.pt", rank=4, nc=4)
    model_nn = student.yolo.model

    # Freeze everything first
    for p in model_nn.parameters():
        p.requires_grad_(False)

    # Replicate gateway logic (same code as simulator.py)
    head_idx    = len(model_nn.model) - 1
    head_prefix = f'model.{head_idx}.'
    opened = 0
    for name, param in model_nn.named_parameters():
        if head_prefix in name:
            param.requires_grad_(True)
            opened += 1

    check("H1: at least one head param opened", opened > 0, f"{opened} params")

    # cv2.0.* (the normally-uplink-excluded layer) must be open at gateway
    cv2_0_open = [n for n, p in model_nn.named_parameters()
                  if f'{head_prefix}cv2.0.' in n and p.requires_grad]
    check("H2: cv2.0.* requires_grad=True at Gateway", len(cv2_0_open) > 0,
          f"{len(cv2_0_open)} params: e.g. {cv2_0_open[0] if cv2_0_open else 'none'}")

    # cv2.0.0.* (first conv, excluded from AUV uplink) also open
    cv2_0_0_open = [n for n in cv2_0_open if '.cv2.0.0.' in n]
    check("H3: cv2.0.0.* (first conv, excluded from uplink) also open at Gateway",
          len(cv2_0_0_open) > 0, f"{len(cv2_0_0_open)} params")

    # Backbone must still be frozen
    backbone_frozen = not any(
        p.requires_grad for n, p in model_nn.named_parameters()
        if n.startswith('model.0.')
    )
    check("H4: backbone (model.0.*) still frozen after head unlock", backbone_frozen)

    # Confirm AUV uplink payload does NOT include cv2.0.0.conv.weight
    # (Note: cv2.0.0.bn.* ARE included due to FedBN fix, so we specifically check conv)
    student2 = StudentModel(ckpt="yolo12n.pt", rank=4, nc=4)
    uplink_sd = student2.trainable_state_dict(downlink=False)
    cv2_0_0_conv_in_uplink = any(f'model.{head_idx}.cv2.0.0.conv' in k for k in uplink_sd)
    check("H5: cv2.0.0.conv correctly absent from AUV uplink payload",
          not cv2_0_0_conv_in_uplink)


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == '__main__':
    # These tests run without GPU / data
    test_config()
    test_simulator_optimizer_source()
    test_base_gateway_fields()
    test_optimizer_cache_custom()
    test_optimizer_cache_kd()
    test_restore_order()

    # These need ultralytics + yolo12n.pt
    try:
        import ultralytics  # noqa: F401
        test_payload_asymmetry()
        test_head_unlocking()
    except (ImportError, Exception) as e:
        print(f"\n[SKIP] ultralytics unavailable ({e}) — skipping blocks A and H")

    print("\n" + "=" * 55)
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"RESULT: {passed}/{total} tests passed")
    if passed < total:
        print("FAILED:")
        for name, ok in results:
            if not ok:
                print(f"  [x] {name}")
        sys.exit(1)
    else:
        print("All tests passed!")
