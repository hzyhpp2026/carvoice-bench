"""CarVoice Bench 车载语音自动化评测示例"""

import numpy as np
from pathlib import Path


def run_demo():
    """运行一个完整的演示评测流程"""
    import tempfile
    import soundfile as sf
    import yaml
    import json

    from carvoice_bench import Config, Orchestrator, ReportGenerator

    print("=" * 60)
    print("  CarVoice Bench 演示")
    print("=" * 60)

    # 1. 准备测试数据
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"\n📁 临时工作目录: {tmp}")

        # 音频数据（合成正弦波模拟语音）
        audio_dir = tmp / "audio"
        audio_dir.mkdir()
        output_dir = tmp / "report"
        output_dir.mkdir()

        # 生成测试音频
        test_cases = [
            ("打开主驾空调到26度", "whisper_result_open_ac"),
            ("导航到公司", "whisper_result_navi"),
            ("播放周杰伦的歌", "whisper_result_play_music"),
        ]

        for utterance, _ in test_cases:
            sr = 16000
            duration = 0.5
            # 模拟不同频率的"语音"信号
            freq = 300 + hash(utterance) % 200
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            audio = np.sin(2 * np.pi * freq * t) * 0.3
            # 添加包络
            envelope = np.exp(-3 * t)
            audio = audio * envelope
            case_id = f"tc-{hash(utterance) % 1000:03d}"
            sf.write(str(audio_dir / f"{case_id}.wav"), audio, sr)

        # 测试计划
        test_plan = {
            "test_cases": [
                {
                    "id": "tc-001",
                    "utterance": "打开主驾空调到26度",
                    "expected_asr": "打开主驾空调到26度",
                    "timeout_ms": 5000,
                    "expected_can_signals": [
                        {"frame_id": "0x2A1", "signals": {"AC_MAIN_DRIVER": 1, "AC_TEMP_SET": 26}}
                    ],
                    "expected_ui_changes": [
                        {"element": "ac_panel", "state": "visible"},
                        {"element": "temp_display", "value": "26℃"},
                    ],
                },
                {
                    "id": "tc-002",
                    "utterance": "导航到公司",
                    "expected_asr": "导航到公司",
                    "timeout_ms": 5000,
                },
                {
                    "id": "tc-003",
                    "utterance": "播放周杰伦的歌",
                    "expected_asr": "播放周杰伦的歌",
                    "timeout_ms": 5000,
                },
            ]
        }

        with open(tmp / "test_plan.yaml", "w", encoding="utf-8") as f:
            yaml.dump(test_plan, f, allow_unicode=True)

        # 2. 配置运行
        print("\n⚙️  初始化配置...")
        cfg = Config(
            asr_model="whisper-base-zh",
            asr_device="cpu",
            output_dir=str(output_dir),
            verbose=True,
        )

        print("\n🚀 启动评测...")
        orchestrator = Orchestrator(cfg)
        report_data = orchestrator.run(
            audio_dir=str(audio_dir),
            test_plan=test_plan,
        )

        # 3. 生成报告
        print("\n📊 生成报告...")
        gen = ReportGenerator(cfg)
        paths = gen.generate(report_data)

        # 4. 输出结果
        summary = report_data["summary"]
        print(f"\n✅ 评测完成!")
        print(f"   总用例: {summary['total_cases']}")
        print(f"   通过: {summary['passed']}")
        print(f"   失败: {summary['failed']}")
        print(f"   通过率: {summary['pass_rate']*100:.1f}%")
        print(f"\n   📄 HTML 报告: {paths.get('html', 'N/A')}")
        print(f"   📄 PDF 报告:  {paths.get('pdf', 'N/A')}")

        # 打印报告内容摘要
        print(f"\n   报告目录: {output_dir}")
        for f in sorted(output_dir.iterdir()):
            size = f.stat().st_size
            print(f"     {f.name} ({size:,} bytes)")

    print("\n" + "=" * 60)
    print("  演示结束")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
