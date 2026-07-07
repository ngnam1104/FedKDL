"""
compat.py
─────────
Compatibility shim for checkpoints that were serialised (torch.save / pickle)
when the package lived at ``tasks.detection_2d.*``.

Python pickle embeds the fully-qualified class name at save time.  After the
package was moved to ``detection_2d.*``, those checkpoints cannot be loaded
unless Python can resolve the old module path.

Importing this module (or calling ``register()``) registers lightweight
``sys.modules`` aliases so that::

    tasks.detection_2d.models.lora.LoRAConv2d

is silently resolved to::

    detection_2d.models.lora.LoRAConv2d

Call this **before** any ``torch.load`` / ``YOLO(ckpt)`` that may touch an
old checkpoint.

Usage
-----
::

    import detection_2d.compat  # side-effect: registers shims

    # or, more explicitly:
    from detection_2d.compat import register
    register()
"""

import sys
import types


def register() -> None:
    """Register ``tasks.detection_2d.*`` as aliases to ``detection_2d.*``.

    Safe to call multiple times; subsequent calls are no-ops.
    """
    import detection_2d
    import detection_2d.models
    import detection_2d.models.lora
    import detection_2d.models.yolo_wrapper
    import detection_2d.knowledge_compression

    _aliases = {
        "tasks": None,                                         # root shim
        "tasks.detection_2d": detection_2d,
        "tasks.detection_2d.models": detection_2d.models,
        "tasks.detection_2d.models.lora": detection_2d.models.lora,
        "tasks.detection_2d.models.yolo_wrapper": detection_2d.models.yolo_wrapper,
        "tasks.detection_2d.knowledge_compression": detection_2d.knowledge_compression,
    }

    # Lazily import sub-modules that might not be loaded yet
    try:
        import detection_2d.trainer
        _aliases["tasks.detection_2d.trainer"] = detection_2d.trainer
    except ImportError:
        pass
    try:
        import detection_2d.knowledge_compression.knowledge_distillation as _kd
        _aliases["tasks.detection_2d.knowledge_compression.knowledge_distillation"] = _kd
    except ImportError:
        pass
    try:
        import detection_2d.knowledge_compression.int8_quantization as _q
        _aliases["tasks.detection_2d.knowledge_compression.int8_quantization"] = _q
    except ImportError:
        pass

    if sys.modules.get("tasks") is None:
        tasks_shim = types.ModuleType("tasks")
        sys.modules.setdefault("tasks", tasks_shim)
    else:
        tasks_shim = sys.modules["tasks"]

    # Attach detection_2d as an attribute of the shim so attribute lookups work
    if not hasattr(tasks_shim, "detection_2d"):
        tasks_shim.detection_2d = detection_2d

    for mod_name, mod_obj in _aliases.items():
        if mod_name == "tasks":
            continue
        sys.modules.setdefault(mod_name, mod_obj)


# Auto-register when module is imported
register()
