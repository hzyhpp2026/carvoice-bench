"""Online execution helpers for cloud-backed car voice tests."""

from importlib import import_module


def __getattr__(name):
    lazy = {
        "AliyunDashScopeClient": ("carvoice_bench.online.aliyun", "AliyunDashScopeClient"),
        "OnlineAudioIO": ("carvoice_bench.online.audio_io", "OnlineAudioIO"),
    }
    if name in lazy:
        module_name, attr = lazy[name]
        module = import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
