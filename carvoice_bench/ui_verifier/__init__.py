"""UI 校验子包 — 使用延迟导入避免安装时依赖链错误"""

def __getattr__(name):
    _lazy = {
        "UIDetector": ("carvoice_bench.ui_verifier.detector", "UIDetector"),
        "TemplateMatcher": ("carvoice_bench.ui_verifier.template_matcher", "TemplateMatcher"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
