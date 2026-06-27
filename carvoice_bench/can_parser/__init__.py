"""CAN 日志解析子包"""

def __getattr__(name):
    _lazy = {
        "CANLogParser": ("carvoice_bench.can_parser.parser", "CANLogParser"),
        "DBCParser": ("carvoice_bench.can_parser.dbc", "DBCParser"),
        "CANSignalMatcher": ("carvoice_bench.can_parser.matcher", "CANSignalMatcher"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
