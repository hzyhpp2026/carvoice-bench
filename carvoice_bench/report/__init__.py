"""报告生成子包"""

def __getattr__(name):
    _lazy = {
        "HTMLReportGenerator": ("carvoice_bench.report.html_report", "HTMLReportGenerator"),
        "PDFReportGenerator": ("carvoice_bench.report.pdf_report", "PDFReportGenerator"),
        "ReportGenerator": ("carvoice_bench.report.report_api", "ReportGenerator"),
    }
    if name in _lazy:
        import importlib
        mod_path, cls_name = _lazy[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
