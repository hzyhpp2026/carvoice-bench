"""CarVoice Bench 单元测试"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import yaml
import pytest


class TestASRBech:
    """ASR 评测模块测试"""

    def test_wer_calculator(self):
        from carvoice_bench.asr_bench.wer import WERCalculator
        calc = WERCalculator()
        # 精确匹配
        result = calc.compute_wer("打开空调", "打开空调")
        assert result["wer"] == 0.0
        # 完全不匹配
        result = calc.compute_wer("打开空调", "关闭窗户")
        assert result["wer"] > 0
        # 上下文WER
        result = calc.compute_wer_c("打开空调", "关闭空调")
        assert "wer_c" in result

    def test_cer(self):
        from carvoice_bench.asr_bench.wer import WERCalculator
        calc = WERCalculator()
        result = calc.compute_cer("打开空调", "打开空调")
        assert result["cer"] == 0.0
        result = calc.compute_cer("打开空调", "打天窗")
        assert result["cer"] > 0

    def test_latency_measurer(self):
        from carvoice_bench.asr_bench.latency import ASRLatencyMeasurer
        measurer = ASRLatencyMeasurer()
        # 日志回放模式
        result = measurer.measure_replay(
            [100.0, 200.0, 300.0],
            [500.0, 650.0, 800.0]
        )
        assert result["sample_count"] == 3
        assert result["min_latency_ms"] > 0
        assert result["p50_latency_ms"] > 0
        assert result["p95_latency_ms"] > 0

    def test_latency_injection(self):
        from carvoice_bench.asr_bench.latency import ASRLatencyMeasurer
        measurer = ASRLatencyMeasurer()
        sr = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32)) * 0.3
        result = measurer.measure_injection(audio, sr, None)
        assert "e2e_latency_ms" in result

    def test_engine_imports(self):
        from carvoice_bench.asr_bench.engine import ASREngine
        # 只测试初始化，不加载实际模型
        assert ASREngine is not None


class TestTTSAnalyzer:
    """TTS 分析模块测试"""

    def test_mos_predictor(self):
        from carvoice_bench.tts_analyzer.mos import MOSPredictor
        predictor = MOSPredictor()
        sr = 24000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)) * 0.3
        result = predictor.predict(audio, sr)
        assert "mos" in result
        assert 1.0 <= result["mos"] <= 5.0

    def test_prosody_analyzer(self):
        from carvoice_bench.tts_analyzer.prosody import ProsodyAnalyzer
        analyzer = ProsodyAnalyzer()
        sr = 24000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32)) * 0.3
        result = analyzer.analyze(audio, "打开空调")
        assert "duration_sec" in result
        assert "speed_syllables_per_sec" in result
        assert "pitch_mean_hz" in result
        assert "pause_count" in result

    def test_emotion_matcher(self):
        from carvoice_bench.tts_analyzer.emotion import EmotionMatcher
        matcher = EmotionMatcher()
        sr = 16000
        audio = np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)) * 0.3
        result = matcher.analyze(audio, sr, expected_emotion="neutral")
        assert "predicted_emotion" in result
        assert "match_score" in result
        assert "emotion_probs" in result


class TestCANParser:
    """CAN 解析模块测试"""

    def test_dbc_parser(self):
        from carvoice_bench.can_parser.dbc import DBCParser
        parser = DBCParser()
        # 无 DBC 文件时的空初始化
        assert parser.signals == {}
        assert parser.frame_names == {}

    def test_can_parser_asc(self):
        from carvoice_bench.can_parser.parser import CANLogParser
        parser = CANLogParser()
        # 无文件时返回空
        frames = parser._parse_asc("/nonexistent.asc")
        assert frames == []

    def test_can_parser_csv(self):
        from carvoice_bench.can_parser.parser import CANLogParser
        import tempfile
        parser = CANLogParser()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp_ms,frame_id,data\n")
            f.write("1000,2A1,021A000000000000\n")
            f.write("1200,2A1,021C000000000000\n")
            csv_path = f.name
        frames = parser._parse_csv(csv_path)
        os.unlink(csv_path)
        assert len(frames) == 2
        assert frames[0]["frame_id_hex"] == "2A1"

    def test_signal_matcher(self):
        from carvoice_bench.can_parser.matcher import CANSignalMatcher
        from carvoice_bench.can_parser.dbc import DBCParser
        dbc = DBCParser()
        matcher = CANSignalMatcher(dbc)
        # 无 CAN 数据时的匹配
        result = matcher.verify_signals(
            [],
            [{"frame_id": "0x2A1", "signals": {"AC_ON": 1}}],
            timeout_ms=1000
        )
        assert result["matched"] is False
        assert result["total_signals"] == 1


class TestUIVerifier:
    """UI 校验模块测试"""

    def test_detector_initialization(self):
        from carvoice_bench.ui_verifier.detector import UIDetector
        detector = UIDetector()
        assert detector.confidence_threshold == 0.45
        assert len(detector.elements) >= 10

    def test_template_matcher(self):
        from carvoice_bench.ui_verifier.template_matcher import TemplateMatcher
        matcher = TemplateMatcher()
        # 无模板时的匹配
        assert matcher.templates == {}

    def test_heuristic_detect(self):
        from carvoice_bench.ui_verifier.detector import UIDetector
        import cv2
        import numpy as np
        detector = UIDetector()
        # 创建测试图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "26C", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        result = detector._heuristic_detect(img, "temp_display",
                                            {"category": "display"})
        assert "present" in result


class TestConfig:
    """配置模块测试"""

    def test_config_defaults(self):
        from carvoice_bench.config import Config
        cfg = Config()
        assert cfg.asr_model == "whisper-base-zh"
        assert cfg.timeout_ms == 5000

    def test_config_from_dict(self):
        from carvoice_bench.config import Config
        cfg = Config.from_dict({"asr_model": "paraformer", "timeout_ms": 10000})
        assert cfg.asr_model == "paraformer"
        assert cfg.timeout_ms == 10000
        assert cfg.asr_device == "cpu"  # default

    def test_online_config_defaults(self):
        from carvoice_bench.config import Config
        cfg = Config(online_mode=True)
        assert cfg.online_mode is True
        assert cfg.cloud_provider == "aliyun"
        assert cfg.env_file == ".env"
        assert cfg.cloud_asr_model
        assert cfg.cloud_tts_model

    def test_env_loader(self):
        from carvoice_bench.utils.env import load_env_file
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("DASHSCOPE_API_KEY=test-key\n# comment\nEMPTY=\n", encoding="utf-8")
            loaded = load_env_file(env_path, override=True)
            assert loaded["DASHSCOPE_API_KEY"] == "test-key"
            assert os.environ["DASHSCOPE_API_KEY"] == "test-key"

    def test_env_model_defaults(self):
        from carvoice_bench.config import Config
        from carvoice_bench.utils.env import first_env, load_env_file
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "ALIYUN_ASR_MODEL=paraformer-test\n"
                "ALIYUN_TTS_MODEL=cosyvoice-test\n"
                "ALIYUN_TTS_VOICE=test_voice\n",
                encoding="utf-8",
            )
            load_env_file(env_path, override=True)
            cfg = Config(
                cloud_asr_model=first_env("ALIYUN_ASR_MODEL") or Config.cloud_asr_model,
                cloud_tts_model=first_env("ALIYUN_TTS_MODEL") or Config.cloud_tts_model,
                cloud_tts_voice=first_env("ALIYUN_TTS_VOICE") or Config.cloud_tts_voice,
            )
            assert cfg.cloud_asr_model == "paraformer-test"
            assert cfg.cloud_tts_model == "cosyvoice-test"
            assert cfg.cloud_tts_voice == "test_voice"

    def test_tts_suffix(self):
        from carvoice_bench.online.aliyun import tts_suffix
        assert tts_suffix("mp3") == ".mp3"
        assert tts_suffix("wav") == ".wav"
        assert tts_suffix("pcm_24000hz_mono_16bit") == ".pcm"


class TestSemanticParser:
    """语义解析模块测试"""

    def test_rule_parser_climate(self):
        from carvoice_bench.semantic.rule_parser import RuleSemanticParser
        parser = RuleSemanticParser()
        result = parser.parse("打开主驾空调到26度")
        assert result["intent"] == "set_climate"
        assert result["slots"]["zone"] == "driver"
        assert result["slots"]["temperature"] == 26

    def test_rule_parser_window(self):
        from carvoice_bench.semantic.rule_parser import RuleSemanticParser
        parser = RuleSemanticParser()
        result = parser.parse("关闭所有车窗")
        assert result["intent"] == "control_window"
        assert result["slots"]["target"] == "window"
        assert result["slots"]["zone"] == "all"
        assert result["slots"]["action"] == "close"

    def test_rule_parser_navigation(self):
        from carvoice_bench.semantic.rule_parser import RuleSemanticParser
        parser = RuleSemanticParser()
        result = parser.parse("导航到公司")
        assert result["intent"] == "navigate"
        assert result["slots"]["destination"] == "公司"


class TestOrchestrator:
    """Orchestrator 集成测试"""

    def test_orchestrator_init(self):
        from carvoice_bench.orchestrator import Orchestrator
        from carvoice_bench.config import Config
        cfg = Config(output_dir="/tmp/cvb_test")
        orchestrator = Orchestrator(cfg)
        assert orchestrator is not None

    def test_simple_run(self):
        from carvoice_bench.orchestrator import Orchestrator
        from carvoice_bench.config import Config
        import tempfile
        import soundfile as sf

        # 创建测试目录和文件
        with tempfile.TemporaryDirectory() as tmpdir:
            test_plan = {
                "test_cases": [
                    {
                        "id": "ut-001",
                        "utterance": "测试指令",
                        "expected_asr": "测试指令",
                        "timeout_ms": 5000,
                    }
                ]
            }
            audio_dir = Path(tmpdir) / "audio"
            audio_dir.mkdir()
            sr = 16000
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)) * 0.3
            sf.write(str(audio_dir / "ut-001.wav"), audio, sr)

            cfg = Config(output_dir=str(Path(tmpdir) / "report"), verbose=True)
            orchestrator = Orchestrator(cfg)
            result = orchestrator.run(audio_dir=str(audio_dir), test_plan=test_plan)
            assert "summary" in result
            assert result["summary"]["total_cases"] == 1
