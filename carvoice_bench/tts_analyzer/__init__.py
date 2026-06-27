"""TTS 分析子包"""

def __getattr__(name):
    _lazy = {
        "MOSPredictor": ("carvoice_bench.tts_analyzer.mos", "MOSPredictor"),
        "ProsodyAnalyzer": ("carvoice_bench.tts_analyzer.prosody", "ProsodyAnalyzer"),
        "EmotionMatcher": ("carvoice_bench.tts_analyzer.emotion", "EmotionMatcher"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
