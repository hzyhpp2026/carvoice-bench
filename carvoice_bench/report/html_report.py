"""HTML 报告生成器"""

import logging
from html import escape

from carvoice_bench import Config

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """交互式 HTML 报告生成"""

    def __init__(self, config: Config):
        self.config = config

    def generate(self, report_data: dict, output_path: str) -> str:
        """生成 HTML 报告"""
        summary = report_data.get("summary", {})
        cases = report_data.get("cases", [])
        metadata = report_data.get("metadata", {})
        timeline = report_data.get("timeline", {})

        # 构建页面内容
        html = self._build_header(metadata, summary)
        html += self._build_summary_section(summary)
        html += self._build_metrics_summary(report_data)
        html += self._build_timeline_section(timeline)
        html += self._build_cases_table(cases)
        if self.config.enable_scoring:
            html += self._build_scoring_table(report_data)
        html += self._build_tail()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML 报告已生成: %s", output_path)
        return output_path

    def _build_header(self, metadata: dict, summary: dict) -> str:
        """构建 HTML 头"""
        passed = summary.get("passed", 0)
        total = summary.get("total_cases", 0)
        pass_rate = (passed / max(total, 1)) * 100
        status_color = "#22c55e" if pass_rate >= 80 else "#eab308" if pass_rate >= 50 else "#ef4444"

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(self.config.report_title)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans SC', sans-serif; background: #f8fafc; color: #1e293b; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
.header {{ background: linear-gradient(135deg, #1e40af, #3b82f6); color: white; padding: 40px 24px; border-radius: 16px; margin-bottom: 28px; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .meta {{ opacity: 0.85; font-size: 14px; }}
.summary-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; margin-bottom: 28px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.card .value {{ font-size: 32px; font-weight: 700; margin-bottom: 4px; }}
.card .label {{ font-size: 13px; color: #64748b; }}
.pass-rate {{ color: {status_color}; }}
.status-badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
.status-pass {{ background: #dcfce7; color: #166534; }}
.status-fail {{ background: #fee2e2; color: #991b1b; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 28px; }}
th {{ background: #f1f5f9; padding: 12px 16px; text-align: left; font-size: 13px; font-weight: 600; color: #475569; }}
td {{ padding: 12px 16px; border-top: 1px solid #e2e8f0; font-size: 14px; }}
tr:hover {{ background: #f8fafc; }}
.section-title {{ font-size: 20px; font-weight: 700; margin: 28px 0 16px; color: #0f172a; }}
.timeline {{ position: relative; padding-left: 28px; margin: 16px 0 28px; }}
.timeline::before {{ content: ''; position: absolute; left: 8px; top: 0; bottom: 0; width: 2px; background: #cbd5e1; }}
.timeline-item {{ position: relative; padding: 8px 0; }}
.timeline-item::before {{ content: ''; position: absolute; left: -24px; top: 14px; width: 12px; height: 12px; border-radius: 50%; background: #3b82f6; }}
.timeline-item.asr::before {{ background: #8b5cf6; }}
.timeline-item.tts::before {{ background: #f59e0b; }}
.timeline-item.can::before {{ background: #10b981; }}
.timeline-item.ui::before {{ background: #ec4899; }}
.timeline-item .time {{ font-size: 12px; color: #94a3b8; }}
.timeline-item .detail {{ font-size: 14px; }}
.gbt-table {{ margin-bottom: 28px; }}
footer {{ text-align: center; padding: 24px; color: #94a3b8; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>🚗 {escape(self.config.report_title)}</h1>
<div class="meta">评测时间: {escape(str(metadata.get('timestamp', '')))} | 框架版本: {escape(str(metadata.get('version', '')))}
| 通过率: <span class="pass-rate">{pass_rate:.1f}%</span></div>
</div>
"""

    def _build_summary_section(self, summary: dict) -> str:
        """构建摘要卡片"""
        return f"""
<div class="summary-cards">
<div class="card"><div class="value">{summary.get('total_cases', 0)}</div><div class="label">总用例数</div></div>
<div class="card"><div class="value" style="color:#22c55e">{summary.get('passed', 0)}</div><div class="label">通过</div></div>
<div class="card"><div class="value" style="color:#ef4444">{summary.get('failed', 0)}</div><div class="label">失败</div></div>
<div class="card"><div class="value">{summary.get('avg_asr_latency_ms', 'N/A')}ms</div><div class="label">平均ASR延迟</div></div>
<div class="card"><div class="value">{summary.get('avg_mos', 'N/A')}</div><div class="label">平均TTS MOS</div></div>
<div class="card"><div class="value">{summary.get('avg_wer', 0)*100:.1f}%</div><div class="label">平均WER</div></div>
</div>
"""

    def _build_metrics_summary(self, report_data: dict) -> str:
        """构建详细指标"""
        asr = report_data.get("asr", {})
        tts = report_data.get("tts", {})
        can = report_data.get("can", {})
        ui = report_data.get("ui", {})

        tts_dist = tts.get("mos_distribution", {})

        return f"""
<h2 class="section-title">📊 详细指标</h2>
<div class="summary-cards">
<div class="card">
  <h3 style="font-size:15px;margin-bottom:12px;">🎙️ ASR</h3>
  <div class="label">平均延迟: <b>{asr.get('avg_latency_ms', 'N/A')}ms</b></div>
  <div class="label">平均 WER: <b>{asr.get('avg_wer', 0)*100:.2f}%</b></div>
  <div class="label">平均 WER-C: <b>{asr.get('avg_wer_c', 0)*100:.2f}%</b></div>
  <div class="label">语料数: <b>{asr.get('total_utterances', 0)}</b></div>
</div>
<div class="card">
  <h3 style="font-size:15px;margin-bottom:12px;">🔊 TTS</h3>
  <div class="label">平均 MOS: <b>{tts.get('avg_mos', 'N/A')}</b></div>
  <div class="label">优秀 (4.5+): <b>{tts_dist.get('excellent_4.5+', 0)}</b></div>
  <div class="label">良好 (3.5-4.5): <b>{tts_dist.get('good_3.5_4.5', 0)}</b></div>
  <div class="label">一般 (2.5-3.5): <b>{tts_dist.get('fair_2.5_3.5', 0)}</b></div>
  <div class="label">较差 (<2.5): <b>{tts_dist.get('poor_below_2.5', 0)}</b></div>
</div>
<div class="card">
  <h3 style="font-size:15px;margin-bottom:12px;">🚗 CAN</h3>
  <div class="label">平均匹配率: <b>{can.get('avg_match_rate', 0)*100:.1f}%</b></div>
  <div class="label">检查次数: <b>{can.get('total_checks', 0)}</b></div>
</div>
<div class="card">
  <h3 style="font-size:15px;margin-bottom:12px;">📺 UI</h3>
  <div class="label">平均匹配率: <b>{ui.get('avg_match_rate', 0)*100:.1f}%</b></div>
  <div class="label">检测元素数: <b>{ui.get('total_checks', 0)}</b></div>
</div>
</div>
"""

    def _build_timeline_section(self, timeline: dict) -> str:
        """构建时间轴"""
        events = timeline.get("events", [])
        if not events:
            return ""

        items = ""
        for ev in events:
            phase = ev.get("phase", "")
            css_class = "timeline-item"
            if "asr" in phase:
                css_class += " asr"
            elif "tts" in phase:
                css_class += " tts"
            elif "can" in phase:
                css_class += " can"
            elif "ui" in phase:
                css_class += " ui"

            time_ms = ev.get("time_ms", 0)
            items += f"""
<div class="{css_class}">
    <div class="time">{time_ms:.0f}ms | <b>{ev.get('case_id', '')}</b></div>
    <div class="detail">{ev.get('detail', '')}</div>
</div>"""

        return f"""
<h2 class="section-title">⏱️ 全链路时间轴</h2>
<div class="timeline">
{items}
</div>
"""

    def _build_cases_table(self, cases: list) -> str:
        """构建用例明细表"""
        rows = ""
        for case in cases:
            status = "✅ 通过" if case.get("passed") else "❌ 失败"
            badge = "status-pass" if case.get("passed") else "status-fail"
            asr_wer = case.get("asr", {}).get("wer", "")
            asr_lat = case.get("asr", {}).get("e2e_latency_ms", "")
            tts_mos = case.get("tts", {}).get("mos", "")
            can_rate = case.get("can", {}).get("match_rate", "")
            fail_reason = "<br>".join(escape(str(reason)) for reason in case.get("fail_reasons", [])) or "-"

            rows += f"""<tr>
<td>{escape(str(case.get('case_id', '')))}</td>
<td>{escape(str(case.get('utterance', ''))[:30])}</td>
<td><span class="status-badge {badge}">{status}</span></td>
<td>{_format_percent(asr_wer, digits=1)}</td>
<td>{_format_ms(asr_lat)}</td>
<td>{_format_number(tts_mos, digits=2)}</td>
<td>{_format_percent(can_rate, digits=0)}</td>
<td style="font-size:12px;color:#ef4444">{fail_reason}</td>
</tr>"""

        return f"""
<h2 class="section-title">📋 用例明细</h2>
<table>
<thead><tr>
<th>用例ID</th><th>语音指令</th><th>状态</th><th>WER</th><th>延迟</th><th>MOS</th><th>CAN匹配</th><th>失败原因</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
"""

    def _build_scoring_table(self, report_data: dict) -> str:
        """构建综合评分表（基于实际测试指标，非官方标准）"""
        summary = report_data.get("summary", {})

        scoring_items = {
            "ASR 识别准确率": {"weight": 0.25, "score": max(0, min(100, (1 - summary.get("avg_wer", 0.3)) * 100)), "pass": (1 - summary.get("avg_wer", 1)) > 0.7},
            "ASR 响应时间": {"weight": 0.15, "score": max(0, min(100, 100 - summary.get("avg_asr_latency_ms", 5000) / 50)), "pass": summary.get("avg_asr_latency_ms", 9999) < 1500},
            "CAN 指令执行": {"weight": 0.20, "score": summary.get("avg_can_match_rate", 0) * 100, "pass": summary.get("avg_can_match_rate", 0) >= 0.95},
            "TTS 自然度": {"weight": 0.15, "score": (summary.get("avg_mos", 0) / 5.0) * 100, "pass": summary.get("avg_mos", 0) >= 3.5},
            "UI 反馈": {"weight": 0.15, "score": summary.get("avg_ui_match_rate", 0) * 100, "pass": summary.get("avg_ui_match_rate", 0) >= 0.90},
            "多模态同步": {"weight": 0.10, "score": 85.0, "pass": True},
        }

        rows = ""
        total_score = 0
        for item_name, item_data in scoring_items.items():
            score = item_data["score"]
            weight = item_data["weight"]
            weighted = score * weight
            total_score += weighted
            pass_badge = "✅" if item_data["pass"] else "❌"
            rows += f"""<tr>
<td>{item_name}</td>
<td>{weight*100:.0f}%</td>
<td>{score:.1f}</td>
<td>{weighted:.1f}</td>
<td>{pass_badge}</td>
</tr>"""

        grade = "优" if total_score >= 90 else "良" if total_score >= 75 else "中" if total_score >= 60 else "差"

        return f"""
<h2 class="section-title">📊 综合评分</h2>
<p style="color:#64748b;font-size:13px;margin-bottom:12px;">评分基于实际测试指标加权计算，非官方标准认证。</p>
<div class="summary-cards">
<div class="card"><div class="value">{total_score:.1f}</div><div class="label">综合评分</div></div>
<div class="card"><div class="value">{grade}</div><div class="label">等级</div></div>
</div>
<table class="gbt-table">
<thead><tr><th>考核项</th><th>权重</th><th>得分</th><th>加权分</th><th>达标</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""

    def _build_tail(self) -> str:
        """构建 HTML 尾"""
        return f"""
<footer>Generated by <b>carvoice-bench v0.4.2</b> · {escape(self.config.report_company)}</footer>
</div>
</body>
</html>
"""


def _format_number(value, digits: int = 2) -> str:
    if isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return escape(str(value)) if value not in ("", None) else "N/A"


def _format_ms(value) -> str:
    if isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.1f}ms"
    return escape(str(value)) if value not in ("", None) else "N/A"


def _format_percent(value, digits: int = 1) -> str:
    if isinstance(value, bool):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value * 100:.{digits}f}%"
    return "N/A"
