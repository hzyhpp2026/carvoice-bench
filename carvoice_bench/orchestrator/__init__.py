"""Orchestrator 子包 — 全链路编排与时间对齐"""

def __getattr__(name):
    _lazy = {
        "Orchestrator": ("carvoice_bench.orchestrator.timeline", "Orchestrator"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
