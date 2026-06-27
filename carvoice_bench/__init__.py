"""
carvoice-bench — 车载语音自动化评测框架
======================================
ASR延迟打点 / TTS音频分析 / CAN日志解析 / UI状态校验 / 全链路报告
"""

__version__ = "0.5.0"


def __getattr__(name):
    """支持延迟导入，避免安装时依赖链错误"""
    _lazy_imports = {
        "Config": ("carvoice_bench.config", "Config"),
        "Orchestrator": ("carvoice_bench.orchestrator.timeline", "Orchestrator"),
        "ReportGenerator": ("carvoice_bench.report.report_api", "ReportGenerator"),
    }
    if name in _lazy_imports:
        import importlib
        mod_path, cls_name = _lazy_imports[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
