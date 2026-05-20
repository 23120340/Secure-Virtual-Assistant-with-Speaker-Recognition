"""Core package — dùng cho Tuần 2 (CLI/script) và Tuần 3 (web)."""

# ──────────────────────────────────────────────────────────────────────────
# Workaround cho SpeechBrain v1.x LazyModule bug trên Windows.
#
# Crash chain (xem traceback gốc khi ASR_BACKEND=phowhisper):
#   from transformers import pipeline
#     → transformers.integrations.sonicmoe → @torch._dynamo.allow_in_graph
#     → torch._library.custom_ops._register_to_dispatcher
#     → torch.library._register_fake → inspect.get_source(stacklevel)
#     → inspect.getframeinfo → inspect.getmodule(obj, file)
#     → ismodule(speechbrain.integrations.k2_fsa_LazyModule) → True
#     → hasattr(LazyModule, "__file__")
#     → LazyModule.__getattr__("__file__") → ensure_module()
#       → importlib.import_module("speechbrain.integrations.k2_fsa")
#       → ModuleNotFoundError: No module named 'k2'
#     → SpeechBrain wrap thành ImportError("Lazy import failed")
#     → hasattr() KHÔNG catch ImportError → bubble up → crash.
#
# Trong SpeechBrain v1.x có guard tại importutils.py:89-92 dự định trả
# AttributeError khi caller là inspect.py, NHƯNG check là:
#     filename.endswith("/inspect.py")
# Trên Windows path là `C:\Python314\Lib\inspect.py` (backslash) → check
# luôn False → guard miss → bug.
#
# Fix: patch `LazyModule.ensure_module` để chuẩn hoá path-separator check
# cross-platform. Còn stub `k2` để nếu user thật sự dùng `import k2` ở chỗ
# khác cũng không crash (defense in depth — không expose attribute thật).
import os as _os
import sys as _sys
import types as _types

# Stub k2 (defense in depth): vô hại, chỉ giúp `import k2` succeed.
if "k2" not in _sys.modules:
    _k2_stub = _types.ModuleType("k2")
    _k2_stub.__version__ = "0.0.0-stub"
    _k2_stub.__doc__ = (
        "Stub for k2 — installed by secva to work around SpeechBrain lazy-import "
        "bug. KHÔNG cài k2 thật; nếu cần tính năng k2-fsa, install k2 từ "
        "https://k2-fsa.github.io/k2/installation/from_wheels.html"
    )
    _sys.modules["k2"] = _k2_stub


def _patch_speechbrain_lazymodule():
    """Patch LazyModule.ensure_module → cross-platform inspect.py detection.

    Bug gốc dùng `filename.endswith("/inspect.py")` chỉ match POSIX path.
    Thay bằng `basename == "inspect.py"` (cross-platform).
    """
    try:
        from speechbrain.utils import importutils
    except Exception:
        return  # speechbrain chưa cài → không cần patch

    LazyModule = getattr(importutils, "LazyModule", None)
    if LazyModule is None:
        return

    _original_ensure_module = LazyModule.ensure_module

    def _patched_ensure_module(self, stacklevel):
        # Imports nội bộ — không phụ thuộc module-level names (đã bị del).
        import inspect
        import os.path
        import sys
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except (AttributeError, ValueError):
            importer_frame = None

        # Cross-platform: basename check thay vì endswith("/inspect.py")
        if importer_frame is not None and os.path.basename(importer_frame.filename) == "inspect.py":
            raise AttributeError()

        return _original_ensure_module(self, stacklevel)

    # Idempotent: chỉ patch 1 lần
    if not getattr(LazyModule, "_secva_patched", False):
        LazyModule.ensure_module = _patched_ensure_module
        LazyModule._secva_patched = True


_patch_speechbrain_lazymodule()
del _os, _sys, _types, _patch_speechbrain_lazymodule
