"""ASR 评测子包"""

def __getattr__(name):
    _lazy = {
        "ASREngine": ("carvoice_bench.asr_bench.engine", "ASREngine"),
        "ASRLatencyMeasurer": ("carvoice_bench.asr_bench.latency", "ASRLatencyMeasurer"),
        "WERCalculator": ("carvoice_bench.asr_bench.wer", "WERCalculator"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
