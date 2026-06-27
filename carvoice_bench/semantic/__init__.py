"""Semantic parsing adapters."""

from importlib import import_module


def __getattr__(name):
    lazy = {
        "SemanticParser": ("carvoice_bench.semantic.parser", "SemanticParser"),
        "RuleSemanticParser": ("carvoice_bench.semantic.rule_parser", "RuleSemanticParser"),
    }
    if name in lazy:
        module_name, attr = lazy[name]
        module = import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
