"""报告生成器 — 支持 HTML / PDF 两种格式"""

import json
import logging
from pathlib import Path
from typing import Optional

from carvoice_bench import Config
from carvoice_bench.report.html_report import HTMLReportGenerator
from carvoice_bench.report.pdf_report import PDFReportGenerator

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    评测报告生成器

    支持:
    - 交互式 HTML 报告（含时间轴动画、失败截图、根因定位）
    - PDF 报告（含综合评分表）
    """

    def __init__(self, config: Config):
        self.config = config
        self.html_gen = HTMLReportGenerator(config)
        self.pdf_gen = PDFReportGenerator(config)

    def generate(self, report_data: dict,
                 formats: Optional[list[str]] = None) -> dict:
        """
        生成报告

        Args:
            report_data: 评测数据 (来自 Orchestrator.run())
            formats: 格式列表 ["html", "pdf"]，默认全部

        Returns: {format: file_path, ...}
        """
        if formats is None:
            formats = ["html", "pdf"]

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        if "html" in formats:
            html_path = self.html_gen.generate(report_data, str(output_dir / "report.html"))
            paths["html"] = html_path

        if "pdf" in formats:
            pdf_path = self.pdf_gen.generate(report_data, str(output_dir / "report.pdf"))
            paths["pdf"] = pdf_path

        # 同时导出 CSV 格式的表格数据
        self._export_csv(report_data, output_dir)

        logger.info("报告已生成: %s", paths)
        return paths

    def _export_csv(self, report_data: dict, output_dir: Path):
        """导出 CSV 格式的详细评测数据"""
        import csv

        # ASR 结果
        asr_rows = []
        for case in report_data.get("cases", []):
            asr = case.get("asr", {})
            if asr:
                asr_rows.append({
                    "case_id": case["case_id"],
                    "utterance": asr.get("utterance", ""),
                    "expected": asr.get("expected_asr", ""),
                    "result": asr.get("asr_result", ""),
                    "wer": asr.get("wer", ""),
                    "wer_c": asr.get("wer_c", ""),
                    "cer": asr.get("cer", ""),
                    "latency_ms": asr.get("e2e_latency_ms", ""),
                    "confidence": asr.get("confidence", ""),
                })
        if asr_rows:
            with open(output_dir / "asr_results.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=asr_rows[0].keys())
                writer.writeheader()
                writer.writerows(asr_rows)

        # TTS 结果
        tts_rows = []
        for case in report_data.get("cases", []):
            tts = case.get("tts", {})
            if tts:
                tts_rows.append({
                    "case_id": case["case_id"],
                    "mos": tts.get("mos", ""),
                    "mos_method": tts.get("mos_method", ""),
                    "naturalness": tts.get("naturalness", ""),
                    "smoothness": tts.get("smoothness", ""),
                    "speed": tts.get("prosody", {}).get("speed_syllables_per_sec", ""),
                    "pitch_mean": tts.get("prosody", {}).get("pitch_mean_hz", ""),
                    "pause_count": tts.get("prosody", {}).get("pause_count", ""),
                })
        if tts_rows:
            with open(output_dir / "tts_mos_scores.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=tts_rows[0].keys())
                writer.writeheader()
                writer.writerows(tts_rows)

        logger.info("CSV 数据已导出到 %s", output_dir)
