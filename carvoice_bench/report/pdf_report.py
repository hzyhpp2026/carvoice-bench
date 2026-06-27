"""PDF 报告生成器 — 基于 WeasyPrint"""

import logging
from pathlib import Path
from typing import Optional

from carvoice_bench import Config

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """PDF 报告生成器"""

    def __init__(self, config: Config):
        self.config = config

    def generate(self, report_data: dict, output_path: str) -> str:
        """
        生成 PDF 报告

        使用 WeasyPrint 将 HTML 转为 PDF。
        如果 WeasyPrint 不可用，回退为纯文本 PDF。
        """
        # 先用 HTML 生成器生成临时 HTML
        from carvoice_bench.report.html_report import HTMLReportGenerator

        html_gen = HTMLReportGenerator(self.config)
        tmp_html = str(Path(self.config.output_dir) / "_tmp_report.html")
        html_gen.generate(report_data, tmp_html)

        # 转 PDF
        try:
            from weasyprint import HTML
            HTML(filename=tmp_html).write_pdf(output_path)
            logger.info("PDF 报告已生成: %s", output_path)
        except ImportError:
            logger.warning("weasyprint 未安装，生成纯文本 PDF 回退")
            self._generate_text_fallback(report_data, output_path)
        except Exception as e:
            logger.warning("PDF 生成失败 (%s)，生成纯文本回退", e)
            self._generate_text_fallback(report_data, output_path)
        finally:
            # 清理临时文件
            Path(tmp_html).unlink(missing_ok=True)

        return output_path

    def _generate_text_fallback(self, report_data: dict, output_path: str):
        """生成纯文本 PDF 回退"""
        summary = report_data.get("summary", {})
        cases = report_data.get("cases", [])

        lines = []
        lines.append("=" * 60)
        lines.append(f"  {self.config.report_title}")
        lines.append(f"  生成时间: {report_data.get('metadata', {}).get('timestamp', '')}")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"  总用例: {summary.get('total_cases', 0)}")
        lines.append(f"  通过: {summary.get('passed', 0)}")
        lines.append(f"  失败: {summary.get('failed', 0)}")
        lines.append(f"  通过率: {summary.get('pass_rate', 0)*100:.1f}%")
        lines.append("")
        lines.append(f"  平均ASR延迟: {summary.get('avg_asr_latency_ms', 'N/A')}ms")
        lines.append(f"  平均WER: {summary.get('avg_wer', 0)*100:.2f}%")
        lines.append(f"  平均TTS MOS: {summary.get('avg_mos', 'N/A')}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("  用例明细")
        lines.append("-" * 60)
        for case in cases:
            status = "PASS" if case.get("passed") else "FAIL"
            lines.append(f"  [{status}] {case.get('case_id', '')}: {case.get('utterance', '')[:40]}")
            if not case.get("passed") and case.get("fail_reasons"):
                for reason in case["fail_reasons"]:
                    lines.append(f"         - {reason}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("文本回退 PDF 已生成: %s", output_path)
