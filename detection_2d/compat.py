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


# ---------------------------------------------------------------------------
# Mapping: old tasks.detection_2d.X  →  detection_2d.X
# ---------------------------------------------------------------------------
_ALIAS_MAP = {
    "tasks.detection_2d":                                          "detection_2d",
    "tasks.detection_2d.models":                                   "detection_2d.models",
    "tasks.detection_2d.models.lora":                              "detection_2d.models.lora",
    "tasks.detection_2d.models.yolo_wrapper":                      "detection_2d.models.yolo_wrapper",
    "tasks.detection_2d.knowledge_compression":                    "detection_2d.knowledge_compression",
    "tasks.detection_2d.trainer":                                  "detection_2d.trainer",
    "tasks.detection_2d.knowledge_compression.knowledge_distillation":
        "detection_2d.knowledge_compression.knowledge_distillation",
    "tasks.detection_2d.knowledge_compression.int8_quantization":
        "detection_2d.knowledge_compression.int8_quantization",
    "tasks.detection_2d.knowledge_compression.topk_sparsification":
        "detection_2d.knowledge_compression.topk_sparsification",
}


class _TasksDetectionFinder:
    """A sys.meta_path finder that redirects ``tasks.detection_2d.*`` imports
    to the real ``detection_2d.*`` package without triggering circular imports.

    It does NOT import anything eagerly — resolution happens only when Python
    actually tries to import a matching module name.
    """

    def find_module(self, fullname, path=None):  # Python 3 legacy hook
        if fullname in _ALIAS_MAP:
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        real_name = _ALIAS_MAP[fullname]
        # Import the real module (may already be in sys.modules)
        __import__(real_name)
        real_mod = sys.modules[real_name]
        sys.modules[fullname] = real_mod
        return real_mod


def register() -> None:
    """Register backward-compat aliases for ``tasks.detection_2d.*``.

    * Installs a meta_path finder so future imports resolve correctly.
    * Also aliases any ``tasks.detection_2d.*`` modules that are already
      present in sys.modules right now (e.g. lora which may already be loaded).
    * Ensures a ``tasks`` shim module exists in sys.modules.

    Safe to call multiple times.
    """
    # Ensure finder is installed exactly once
    if not any(isinstance(f, _TasksDetectionFinder) for f in sys.meta_path):
        sys.meta_path.append(_TasksDetectionFinder())

    # Ensure a 'tasks' stub exists in sys.modules
    if "tasks" not in sys.modules:
        sys.modules["tasks"] = types.ModuleType("tasks")

    # Alias any already-loaded detection_2d sub-modules right now
    # (without triggering new imports that could cause circular deps).
    for old_name, new_name in _ALIAS_MAP.items():
        if old_name not in sys.modules and new_name in sys.modules:
            sys.modules[old_name] = sys.modules[new_name]


# Auto-register on import
register()
