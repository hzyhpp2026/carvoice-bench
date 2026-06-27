"""运行 CarVoice Bench 的轻量 mock demo。

该 demo 不加载 Whisper、wav2vec2、OpenCV、soundfile 或 CAN 工具链，
用于验证调度、指标计算和 HTML 报告生成是否完整可用。
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carvoice_bench import Config, Orchestrator, ReportGenerator


def run_demo() -> dict:
    """构造三条车载语音用例，并在 mock 模式下生成报告。"""
    output_dir = ROOT / "examples" / "mock_report"
    audio_dir = ROOT / "examples" / "mock_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    test_plan = {
        "project": {"name": "MockCarVoiceBench", "version": "1.0.0"},
        "test_cases": [
            # 单轮车控用例：覆盖 ASR、TTS、CAN、UI 和语义理解指标。
            {
                "id": "tc-001",
                "utterance": "打开主驾空调到26度",
                "expected_asr": "打开主驾空调到26度",
                "mock_asr_result": "打开主驾空调到26度",
                "mock_latency_ms": 286,
                "mock_mos": 4.42,
                "expected_can_signals": [
                    {"frame_id": "0x2A1", "signals": {"AC_MAIN_DRIVER": 1, "AC_TEMP_SET": 26}}
                ],
                "expected_ui_changes": [
                    {"element": "ac_panel", "state": "visible"},
                    {"element": "temp_display", "value": "26℃"},
                ],
                "expected_semantics": {
                    "intent": "set_climate",
                    "slots": {"zone": "driver", "temperature": 26},
                },
                "timeout_ms": 1500,
            },
            # 多轮会话用例：覆盖最终状态跟踪和上下文继承指标。
            {
                "id": "tc-002",
                "utterance": "导航到公司",
                "expected_asr": "导航到公司",
                "mock_asr_result": "导航到公司",
                "mock_latency_ms": 241,
                "mock_mos": 4.28,
                "expected_ui_changes": [
                    {"element": "nav_map", "state": "visible"},
                ],
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "导航到公司"},
                        {"role": "assistant", "text": "已开始导航到公司"},
                    ],
                    "expected_final_state": {"domain": "navigation", "active_route": "company"},
                },
                "timeout_ms": 1500,
            },
            # 全双工用例：覆盖用户插话、barge-in 行为和全双工事件延迟指标。
            {
                "id": "tc-003",
                "utterance": "播放周杰伦的歌",
                "expected_asr": "播放周杰伦的歌",
                "mock_asr_result": "播放周杰伦的歌",
                "mock_latency_ms": 312,
                "mock_mos": 4.36,
                "expected_can_signals": [
                    {"frame_id": "0x310", "signals": {"MEDIA_PLAY": 1}}
                ],
                "expected_ui_changes": [
                    {"element": "music_player", "state": "visible"},
                ],
                "full_duplex": {
                    "scenario": "user_interruption",
                    "expected_events": [
                        {"type": "user_interrupt", "start_ms": 800, "end_ms": 1300}
                    ],
                    "expected_behavior": {"barge_in_handled": True},
                    "tolerance_ms": 500,
                },
                "timeout_ms": 1500,
            },
        ],
    }

    cfg = Config(
        output_dir=str(output_dir),
        report_title="CarVoice Bench Mock Demo Report",
        mock_mode=True,
        enable_scoring=True,
        verbose=True,
    )
    report_data = Orchestrator(cfg).run(audio_dir=str(audio_dir), test_plan=test_plan)
    paths = ReportGenerator(cfg).generate(report_data, formats=["html"])

    summary = report_data["summary"]
    print("CarVoice Bench mock demo completed")
    print(f"cases: {summary['passed']}/{summary['total_cases']} passed")
    print(f"report: {paths['html']}")
    return report_data


if __name__ == "__main__":
    run_demo()
