"""车载语音测试的全链路调度入口。

调度器支持两种模式：
- real mode：按需加载 ASR/TTS/CAN/UI 等真实模块，依赖可选音频和视觉库。
- mock mode：只依赖 Python 标准库，用确定性结果跑通 demo、报告和指标链路。
"""

import json
import logging
import statistics
import time
from pathlib import Path
from typing import Optional

from carvoice_bench import Config
from carvoice_bench.orchestrator.metrics import (
    dialogue_metrics,
    find_matching_event,
    full_duplex_metrics,
    semantic_metrics,
    semantic_sequence_metrics,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """协调 ASR、TTS、CAN、UI 和扩展场景评价，并产出报告数据。"""

    def __init__(self, config: Config):
        self.config = config
        self._modules = {}
        self._init_modules()

    def _init_modules(self):
        """按运行模式初始化必要模块，mock 模式避免加载重依赖。"""
        if self.config.mock_mode:
            self._modules["wer_calc"] = _SimpleWERCalculator()
            self._modules["can_matcher"] = _MockCANSignalMatcher()
            self._modules["ui"] = _MockUIDetector()
            self._modules["semantic_parser"] = None
            logger.info("mock modules initialized")
            return

        if self.config.online_mode:
            from carvoice_bench.online import AliyunDashScopeClient, OnlineAudioIO

            if self.config.cloud_provider.lower() != "aliyun":
                raise ValueError(f"unsupported cloud provider: {self.config.cloud_provider}")
            self._modules["cloud"] = AliyunDashScopeClient(self.config)
            self._modules["audio_io"] = OnlineAudioIO()

        from carvoice_bench.asr_bench import ASRLatencyMeasurer, WERCalculator
        from carvoice_bench.can_parser import CANLogParser, CANSignalMatcher, DBCParser
        from carvoice_bench.semantic import RuleSemanticParser
        from carvoice_bench.tts_analyzer import MOSPredictor, ProsodyAnalyzer, EmotionMatcher
        from carvoice_bench.ui_verifier import UIDetector

        cfg = self.config
        if not self.config.online_mode:
            from carvoice_bench.asr_bench import ASREngine

            self._modules["asr_engine"] = ASREngine(
                model_name=cfg.asr_model,
                device=cfg.asr_device,
                sample_rate=cfg.asr_sample_rate,
                language=cfg.asr_language,
            )
        self._modules["asr_latency"] = ASRLatencyMeasurer()
        self._modules["wer_calc"] = WERCalculator()
        self._modules["mos"] = MOSPredictor(device=cfg.asr_device)
        self._modules["prosody"] = ProsodyAnalyzer()
        self._modules["emotion"] = EmotionMatcher()
        if cfg.semantic_parser == "rule":
            self._modules["semantic_parser"] = RuleSemanticParser(cfg.semantic_rules_path)
        elif cfg.semantic_parser == "none":
            self._modules["semantic_parser"] = None
        elif cfg.semantic_parser == "cloud":
            from carvoice_bench.semantic.cloud_parser import CloudSemanticParser

            self._modules["semantic_parser"] = CloudSemanticParser(cfg)
        else:
            raise ValueError(f"unsupported semantic parser: {cfg.semantic_parser}")
        self._modules["can_parser"] = CANLogParser()
        dbc = DBCParser(cfg.can_db_path)
        self._modules["dbc"] = dbc
        self._modules["can_matcher"] = CANSignalMatcher(dbc)
        self._modules["ui"] = UIDetector(
            confidence_threshold=cfg.ui_confidence_threshold,
            elements_config=cfg.ui_template_path,
            yolo_model_path=cfg.ui_yolo_model_path,
        )
        logger.info("modules initialized")

    def run(
        self,
        audio_dir: str,
        can_log_path: Optional[str] = None,
        ui_before_path: Optional[str] = None,
        ui_after_path: Optional[str] = None,
        test_plan: Optional[dict] = None,
    ) -> dict:
        """Run benchmark cases and persist ``report_data.json``."""
        audio_dir_path = Path(audio_dir)
        test_cases = []
        if test_plan:
            test_cases = test_plan.get("test_cases", test_plan.get("testcases", []))

        logger.info("starting evaluation: %d cases", len(test_cases))

        can_frames = []
        if can_log_path and not self.config.mock_mode:
            can_frames = self._modules["can_parser"].parse(can_log_path)
        elif self.config.mock_mode:
            can_frames = [{"timestamp_ms": 320, "frame_id": 0x2A1, "signals": {}}]

        ui_before = str(ui_before_path) if ui_before_path else None
        ui_after = str(ui_after_path) if ui_after_path else None

        asr_results = []
        tts_results = []
        can_results = []
        ui_results = []
        semantic_results = []
        duplex_results = []
        dialogue_results = []
        case_results = []

        for idx, case in enumerate(test_cases):
            case_id = case.get("id", f"tc-{idx + 1:03d}")
            utterance = case.get("utterance", "")
            expected_asr = case.get("expected_asr", utterance)
            expected_can = case.get("expected_can_signals", [])
            expected_ui = case.get("expected_ui_changes", [])
            # 三类扩展场景都走可选字段：没有配置时不会影响传统 ASR/TTS/CAN/UI 测试。
            expected_semantics = case.get("expected_semantics", {})
            expected_semantic_sequence = case.get("expected_semantic_sequence", [])
            full_duplex_spec = case.get("full_duplex", case.get("expected_full_duplex", {}))
            dialogue_spec = case.get("dialogue", case.get("multi_turn", {}))
            timeout_ms = case.get("timeout_ms", self.config.timeout_ms)

            logger.info("[%s] evaluating: %s", case_id, utterance)
            audio_path = self._find_audio(
                audio_dir_path,
                case_id,
                utterance,
                allow_generic_fallback=not self.config.online_mode,
            )
            case_timeline = {"case_id": case_id, "utterance": utterance, "events": []}
            case_asr = {}
            case_tts = {}
            case_can = {}
            case_ui = {}
            case_semantics = {}
            case_duplex = {}
            case_dialogue = {}

            if self.config.mock_mode:
                # mock 音频链路固定生成 ASR/TTS 指标，确保最小 demo 无需真实 wav 也能跑通。
                case_asr, case_tts = self._run_mock_audio_case(case, expected_asr)
                asr_results.append(case_asr)
                tts_results.append(case_tts)
                self._append_audio_timeline(case_timeline, utterance, case_asr, case_tts)
            elif audio_path:
                logger.info("[%s] using audio: %s", case_id, audio_path)
                if self.config.online_mode:
                    case_asr, case_tts = self._run_online_audio_case(audio_path, utterance, expected_asr, case_id)
                else:
                    case_asr, case_tts = self._run_real_audio_case(audio_path, utterance, expected_asr)
                asr_results.append(case_asr)
                tts_results.append(case_tts)
                self._append_audio_timeline(case_timeline, utterance, case_asr, case_tts)
            elif self.config.online_mode:
                from carvoice_bench.online.aliyun import tts_suffix

                generated_path = audio_dir_path / f"{case_id}{tts_suffix(self.config.cloud_tts_format)}"
                logger.info("[%s] audio not found, generating TTS: %s", case_id, generated_path)
                tts_meta = self._modules["cloud"].synthesize_to_file(utterance, generated_path)
                case_asr, case_tts = self._run_online_audio_case(generated_path, utterance, expected_asr, case_id, tts_meta)
                asr_results.append(case_asr)
                tts_results.append(case_tts)
                self._append_audio_timeline(case_timeline, utterance, case_asr, case_tts)
            else:
                logger.warning("[%s] audio file not found", case_id)

            start_ms = case_timeline["events"][-1].get("time_ms", 0) if case_timeline["events"] else 0

            if (can_frames or self.config.mock_mode) and expected_can:
                case_can = self._modules["can_matcher"].verify_signals(
                    can_frames, expected_can, timeout_ms, start_ms
                )
                can_results.append(case_can)
                case_timeline["events"].append({
                    "phase": "can_execution",
                    "time_ms": start_ms + 200,
                    "detail": f"CAN match rate: {case_can.get('match_rate', 0):.0%}",
                })

            if (ui_before and ui_after and expected_ui) or (self.config.mock_mode and expected_ui):
                case_ui = self._modules["ui"].verify_changes(ui_before or "", ui_after or "", expected_ui)
                ui_results.append(case_ui)
                case_timeline["events"].append({
                    "phase": "ui_change",
                    "time_ms": start_ms + 800,
                    "detail": f"UI match rate: {case_ui.get('match_rate', 0):.0%}",
                })

            if expected_semantics or expected_semantic_sequence:
                # 语义理解只比较结构化 intent/slots，不绑定具体 NLU 实现。
                case_semantics = self._verify_semantics(case, case_asr, expected_semantics, expected_semantic_sequence)
                semantic_results.append(case_semantics)
                case_timeline["events"].append({
                    "phase": "semantic_understanding",
                    "time_ms": start_ms + 120,
                    "detail": f"semantic match rate: {case_semantics.get('match_rate', 0):.0%}",
                })

            if full_duplex_spec:
                # 语音全双工评价复用 Full-Duplex-Bench 风格的事件和行为指标。
                case_duplex = self._verify_full_duplex(case, full_duplex_spec)
                duplex_results.append(case_duplex)
                case_timeline["events"].append({
                    "phase": "full_duplex",
                    "time_ms": start_ms + 160,
                    "detail": f"full-duplex match rate: {case_duplex.get('match_rate', 0):.0%}",
                })

            if dialogue_spec:
                # 多轮会话评价关注最终状态、上下文继承和轮次结构。
                case_dialogue = self._verify_dialogue(case, dialogue_spec)
                dialogue_results.append(case_dialogue)
                case_timeline["events"].append({
                    "phase": "multi_turn_dialogue",
                    "time_ms": start_ms + 180,
                    "detail": f"dialogue match rate: {case_dialogue.get('match_rate', 0):.0%}",
                })

            passed, fail_reasons = self._judge_case(
                case_asr, case_can, case_ui, timeout_ms,
                case_semantics=case_semantics,
                case_duplex=case_duplex,
                case_dialogue=case_dialogue,
            )
            case_results.append({
                "case_id": case_id,
                "utterance": utterance,
                "passed": passed,
                "fail_reasons": fail_reasons,
                "asr": case_asr,
                "tts": case_tts,
                "can": case_can,
                "ui": case_ui,
                "semantics": case_semantics,
                "full_duplex": case_duplex,
                "dialogue": case_dialogue,
                "timeline": case_timeline,
            })

            if self.config.online_mode and self.config.online_case_pause_seconds > 0 and idx < len(test_cases) - 1:
                logger.info(
                    "[%s] pausing %.1fs before next case",
                    case_id,
                    self.config.online_case_pause_seconds,
                )
                time.sleep(self.config.online_case_pause_seconds)

        summary = self._compute_summary(case_results)
        report_data = {
            "metadata": {
                "framework": "carvoice-bench",
                "version": "0.4.2",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config.to_dict(),
                "test_plan": test_plan,
                "enable_scoring": self.config.enable_scoring,
                "mock_mode": self.config.mock_mode,
            },
            "summary": summary,
            "cases": case_results,
            "asr": self._aggregate_asr(asr_results),
            "tts": self._aggregate_tts(tts_results),
            "can": self._aggregate_can(can_results),
            "ui": self._aggregate_ui(ui_results),
            "semantics": self._aggregate_match_results(semantic_results),
            "full_duplex": self._aggregate_match_results(duplex_results),
            "dialogue": self._aggregate_match_results(dialogue_results),
            "timeline": self._build_master_timeline(case_results),
            "capture": self._aggregate_capture(case_results),
        }

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "report_data.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("evaluation complete: %d/%d passed", summary["passed"], summary["total_cases"])
        return report_data

    def _run_mock_audio_case(self, case: dict, expected_asr: str) -> tuple[dict, dict]:
        """生成一组稳定的 ASR/TTS mock 指标，用于无模型环境下的冒烟测试。"""
        asr_text = case.get("mock_asr_result", expected_asr)
        latency_ms = float(case.get("mock_latency_ms", 280))
        wer_result = self._modules["wer_calc"].compute_wer(expected_asr, asr_text)
        wer_c_result = self._modules["wer_calc"].compute_wer_c(expected_asr, asr_text)
        cer_result = self._modules["wer_calc"].compute_cer(expected_asr, asr_text)

        case_asr = {
            "utterance": case.get("utterance", ""),
            "expected_asr": expected_asr,
            "asr_result": asr_text,
            "confidence": float(case.get("mock_confidence", 0.96)),
            "e2e_latency_ms": latency_ms,
            "processing_latency_ms": max(1.0, latency_ms - 80),
            "wer": wer_result["wer"],
            "wer_c": wer_c_result["wer_c"],
            "cer": cer_result["cer"],
            "wer_details": wer_result,
            "audio_duration_ms": float(case.get("mock_audio_duration_ms", 1100)),
        }
        mos = float(case.get("mock_mos", 4.35))
        case_tts = {
            "mos": mos,
            "naturalness": round(min(5.0, mos + 0.05), 2),
            "smoothness": round(max(1.0, mos - 0.08), 2),
            "mos_method": "mock",
            "prosody": {
                "duration_sec": round(case_asr["audio_duration_ms"] / 1000, 3),
                "speed_syllables_per_sec": float(case.get("mock_speed_sps", 4.6)),
                "pitch_mean_hz": float(case.get("mock_pitch_hz", 205)),
                "pause_count": int(case.get("mock_pause_count", 1)),
            },
        }
        return case_asr, case_tts

    def _run_real_audio_case(self, audio_path: Path, utterance: str, expected_asr: str) -> tuple[dict, dict]:
        """调用真实 ASR/TTS 模块计算识别文本、延迟、错误率、MOS 和韵律指标。"""
        import soundfile as sf

        audio, sr = sf.read(str(audio_path))
        asr_lat = self._modules["asr_latency"].measure_injection(audio, sr, self._modules["asr_engine"])
        asr_text = asr_lat["text"]
        wer_result = self._modules["wer_calc"].compute_wer(expected_asr, asr_text)
        wer_c_result = self._modules["wer_calc"].compute_wer_c(expected_asr, asr_text)
        cer_result = self._modules["wer_calc"].compute_cer(expected_asr, asr_text)
        mos_result = self._modules["mos"].predict_file(str(audio_path))
        prosody_result = self._modules["prosody"].analyze(audio, utterance, sample_rate=sr)

        case_asr = {
            "utterance": utterance,
            "expected_asr": expected_asr,
            "asr_result": asr_text,
            "confidence": asr_lat["confidence"],
            "e2e_latency_ms": asr_lat["e2e_latency_ms"],
            "processing_latency_ms": asr_lat["processing_latency_ms"],
            "wer": wer_result["wer"],
            "wer_c": wer_c_result["wer_c"],
            "cer": cer_result["cer"],
            "wer_details": wer_result,
            "audio_duration_ms": asr_lat["audio_duration_ms"],
        }
        case_tts = {
            "mos": mos_result["mos"],
            "naturalness": mos_result.get("naturalness", mos_result["mos"]),
            "smoothness": mos_result.get("smoothness", mos_result["mos"]),
            "mos_method": mos_result.get("method", "unknown"),
            "prosody": prosody_result,
        }
        return case_asr, case_tts

    def _run_online_audio_case(
        self,
        audio_path: Path,
        utterance: str,
        expected_asr: str,
        case_id: str,
        tts_meta: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        """Run a case through optional local I/O and Aliyun cloud ASR."""
        import soundfile as sf

        source_audio_path = Path(audio_path)
        playback_meta = {}
        record_meta = {}
        video_meta = {}
        asr_audio_path = source_audio_path

        should_record_audio = self.config.online_record_seconds > 0
        should_record_video = self.config.online_record_ui_video or self.config.online_record_cabin_video

        if self.config.online_play_audio and (should_record_audio or should_record_video):
            record_path = Path(self.config.output_dir) / "recordings" / f"{case_id}.wav"
            video_path = (
                Path(self.config.output_dir) / "videos" / f"{case_id}.mp4"
                if should_record_video else None
            )
            capture = self._modules["audio_io"].capture_case(
                source_audio_path,
                mic_output_path=record_path if should_record_audio else None,
                seconds=self.config.online_record_seconds,
                sample_rate=self.config.online_record_sample_rate,
                video_output_path=video_path,
                camera_index=self.config.online_camera_index,
                video_fps=self.config.online_video_fps,
            )
            playback_meta = capture.get("playback", {})
            record_meta = capture.get("recording", {})
            video_meta = capture.get("video", {})
            if should_record_audio:
                asr_audio_path = record_path
        elif self.config.online_play_audio:
            playback_meta = self._modules["audio_io"].play(source_audio_path)
        elif should_record_audio:
            record_path = Path(self.config.output_dir) / "recordings" / f"{case_id}.wav"
            record_meta = self._modules["audio_io"].record(
                record_path,
                seconds=self.config.online_record_seconds,
                sample_rate=self.config.online_record_sample_rate,
            )
            asr_audio_path = record_path
        elif should_record_video:
            video_meta = self._video_capture_placeholder(case_id)

        cloud_asr = self._modules["cloud"].transcribe_file(asr_audio_path)
        asr_text = cloud_asr.get("text", "")
        wer_result = self._modules["wer_calc"].compute_wer(expected_asr, asr_text)
        wer_c_result = self._modules["wer_calc"].compute_wer_c(expected_asr, asr_text)
        cer_result = self._modules["wer_calc"].compute_cer(expected_asr, asr_text)

        audio, sr = sf.read(str(asr_audio_path))
        mos_result = self._modules["mos"].predict_file(str(source_audio_path))
        prosody_result = self._modules["prosody"].analyze(audio, utterance, sample_rate=sr)
        duration_ms = len(audio) / sr * 1000 if sr else 0

        case_asr = {
            "utterance": utterance,
            "expected_asr": expected_asr,
            "asr_result": asr_text,
            "confidence": cloud_asr.get("confidence", 0.0),
            "e2e_latency_ms": cloud_asr.get("processing_latency_ms", 0.0),
            "processing_latency_ms": cloud_asr.get("processing_latency_ms", 0.0),
            "wer": wer_result["wer"],
            "wer_c": wer_c_result["wer_c"],
            "cer": cer_result["cer"],
            "wer_details": wer_result,
            "audio_duration_ms": round(duration_ms, 2),
            "cloud": cloud_asr,
            "playback": playback_meta,
            "recording": record_meta,
            "video": video_meta,
        }
        case_tts = {
            "mos": mos_result["mos"],
            "naturalness": mos_result.get("naturalness", mos_result["mos"]),
            "smoothness": mos_result.get("smoothness", mos_result["mos"]),
            "mos_method": mos_result.get("method", "unknown"),
            "prosody": prosody_result,
            "cloud_tts": tts_meta or {},
        }
        return case_asr, case_tts

    def _video_capture_placeholder(self, case_id: str) -> dict:
        """Record requested video capture settings until device-specific adapters are added."""
        requested = []
        if self.config.online_record_ui_video:
            requested.append("ui_video")
        if self.config.online_record_cabin_video:
            requested.append("cabin_video")
        return {
            "case_id": case_id,
            "requested": requested,
            "captured": False,
            "reason": "video capture adapter is not configured yet",
        }

    def _append_audio_timeline(self, case_timeline: dict, utterance: str, case_asr: dict, case_tts: dict):
        """把 ASR/TTS 关键节点追加到单条用例时间线，供 HTML 报告展示。"""
        latency_ms = case_asr.get("e2e_latency_ms", 0)
        case_timeline["events"].extend([
            {"phase": "user_speech", "time_ms": 0, "detail": f"user speech: {utterance}"},
            {"phase": "asr_start", "time_ms": 0, "detail": "ASR processing started"},
            {
                "phase": "asr_end",
                "time_ms": latency_ms,
                "detail": f"ASR result: {case_asr.get('asr_result', '')} "
                          f"(confidence: {case_asr.get('confidence', 0):.2f})",
            },
            {
                "phase": "tts",
                "time_ms": latency_ms + 50,
                "detail": f"TTS MOS: {case_tts.get('mos', 0)} | "
                          f"speed: {case_tts.get('prosody', {}).get('speed_syllables_per_sec', 0):.1f} syl/s",
            },
        ])

    def _verify_semantics(
        self,
        case: dict,
        case_asr: dict,
        expected: dict,
        expected_sequence: Optional[list[dict]] = None,
    ) -> dict:
        """校验语义理解结果；真实模式可从 actual_semantics 接入外部 NLU 输出。"""
        if expected_sequence:
            actual_sequence = case.get("mock_semantic_sequence", case.get("actual_semantic_sequence"))
            if actual_sequence is None and self.config.mock_mode:
                actual_sequence = expected_sequence
            if actual_sequence is None and self._modules.get("semantic_parser") is not None:
                source_text = case_asr.get("asr_result", case.get("utterance", ""))
                parser = self._modules["semantic_parser"]
                actual_sequence = parser.parse_many(source_text, case) if hasattr(parser, "parse_many") else [parser.parse(source_text, case)]
            actual_sequence = actual_sequence or []
            result = semantic_sequence_metrics(expected_sequence, actual_sequence)
            result.update({
                "expected": expected_sequence,
                "actual": actual_sequence,
                "source_text": case_asr.get("asr_result", case.get("utterance", "")),
            })
            return result
        actual = case.get("mock_semantics", case.get("actual_semantics"))
        if actual is None and self.config.mock_mode:
            actual = expected
        if actual is None and self._modules.get("semantic_parser") is not None:
            source_text = case_asr.get("asr_result", case.get("utterance", ""))
            actual = self._modules["semantic_parser"].parse(source_text, case)
        if actual is None:
            actual = {}
        result = semantic_metrics(expected, actual)
        result.update({
            "expected": expected,
            "actual": actual,
            "source_text": case_asr.get("asr_result", case.get("utterance", "")),
        })
        return result

    def _verify_full_duplex(self, case: dict, spec: dict) -> dict:
        """校验语音全双工结果；mock 模式会补齐可计算的实际事件和行为字段。"""
        actual = case.get("mock_full_duplex", case.get("actual_full_duplex", spec if self.config.mock_mode else {}))
        if self.config.mock_mode:
            actual = _mock_full_duplex_actual(spec, actual)
        result = full_duplex_metrics(spec, actual)
        result.update({"expected": spec, "actual": actual})
        return result

    def _verify_dialogue(self, case: dict, spec: dict) -> dict:
        """校验多轮会话结果；真实模式可从 actual_dialogue 接入对话管理器输出。"""
        actual = case.get("mock_dialogue", case.get("actual_dialogue", spec if self.config.mock_mode else {}))
        if self.config.mock_mode:
            actual = _mock_dialogue_actual(spec, actual)
        result = dialogue_metrics(spec, actual)
        result.update({"expected": spec, "actual": actual})
        return result

    def _judge_case(
        self,
        case_asr: dict,
        case_can: dict,
        case_ui: dict,
        timeout_ms: float,
        case_semantics: Optional[dict] = None,
        case_duplex: Optional[dict] = None,
        case_dialogue: Optional[dict] = None,
    ) -> tuple[bool, list[str]]:
        """统一判定单条用例是否通过，并汇总失败原因。"""
        passed = True
        fail_reasons = []

        if case_asr:
            if case_asr.get("wer", 1.0) > 0.3:
                passed = False
                fail_reasons.append(f"WER too high ({case_asr['wer']:.1%})")
            if case_asr.get("e2e_latency_ms", 9999) > timeout_ms:
                passed = False
                fail_reasons.append(
                    f"ASR latency timeout ({case_asr['e2e_latency_ms']:.0f}ms > {timeout_ms}ms)"
                )
        else:
            passed = False
            fail_reasons.append("ASR result missing")

        if case_can and not case_can.get("matched", False):
            passed = False
            fail_reasons.append(f"CAN mismatch ({case_can.get('match_rate', 0):.0%})")

        if case_ui and not case_ui.get("all_passed", False):
            passed = False
            fail_reasons.append(f"UI mismatch ({case_ui.get('match_rate', 0):.0%})")

        if case_semantics and not case_semantics.get("matched", False):
            passed = False
            fail_reasons.append(f"semantic mismatch ({case_semantics.get('match_rate', 0):.0%})")

        if case_duplex and not case_duplex.get("matched", False):
            passed = False
            fail_reasons.append(f"full-duplex mismatch ({case_duplex.get('match_rate', 0):.0%})")

        if case_dialogue and not case_dialogue.get("matched", False):
            passed = False
            fail_reasons.append(f"dialogue mismatch ({case_dialogue.get('match_rate', 0):.0%})")

        return passed, fail_reasons

    def _find_audio(
        self,
        audio_dir: Path,
        case_id: str,
        utterance: str,
        allow_generic_fallback: bool = True,
    ) -> Optional[Path]:
        for ext in [".wav", ".mp3", ".m4a", ".flac"]:
            p = audio_dir / f"{case_id}{ext}"
            if p.exists():
                return p
        if utterance:
            keyword = utterance[:4]
            for p in audio_dir.glob(f"*{keyword}*"):
                if p.suffix in [".wav", ".mp3", ".m4a", ".flac"]:
                    return p
        if not allow_generic_fallback:
            return None
        for p in sorted(audio_dir.glob("*.wav")):
            return p
        return None

    def _compute_summary(self, case_results: list) -> dict:
        total = len(case_results)
        passed = sum(1 for c in case_results if c["passed"])
        failed = total - passed

        asr_latencies = [c["asr"].get("e2e_latency_ms", 0) for c in case_results if c.get("asr")]
        asr_wers = [c["asr"].get("wer", 1.0) for c in case_results if c.get("asr")]
        tts_mos = [c["tts"].get("mos", 0) for c in case_results if c.get("tts")]
        can_rates = [c["can"].get("match_rate", 0) for c in case_results if c.get("can")]
        ui_rates = [c["ui"].get("match_rate", 0) for c in case_results if c.get("ui")]
        semantic_rates = [c["semantics"].get("match_rate", 0) for c in case_results if c.get("semantics")]
        duplex_rates = [c["full_duplex"].get("match_rate", 0) for c in case_results if c.get("full_duplex")]
        dialogue_rates = [c["dialogue"].get("match_rate", 0) for c in case_results if c.get("dialogue")]

        return {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(total, 1), 4),
            "avg_asr_latency_ms": round(_mean(asr_latencies), 2),
            "max_asr_latency_ms": round(max(asr_latencies), 2) if asr_latencies else 0,
            "p95_asr_latency_ms": round(_percentile(asr_latencies, 95), 2),
            "avg_wer": round(_mean(asr_wers), 4),
            "avg_mos": round(_mean(tts_mos), 2),
            "avg_can_match_rate": round(_mean(can_rates), 4),
            "avg_ui_match_rate": round(_mean(ui_rates), 4),
            "avg_semantic_match_rate": round(_mean(semantic_rates), 4),
            "avg_full_duplex_match_rate": round(_mean(duplex_rates), 4),
            "avg_dialogue_match_rate": round(_mean(dialogue_rates), 4),
        }

    def _aggregate_asr(self, results: list) -> dict:
        if not results:
            return {}
        latencies = [r.get("e2e_latency_ms", 0) for r in results]
        wers = [r.get("wer", 1.0) for r in results]
        werc = [r.get("wer_c", 1.0) for r in results]
        return {
            "avg_latency_ms": round(_mean(latencies), 2),
            "avg_wer": round(_mean(wers), 4),
            "avg_wer_c": round(_mean(werc), 4),
            "total_utterances": len(results),
        }

    def _aggregate_tts(self, results: list) -> dict:
        if not results:
            return {}
        mos_scores = [r.get("mos", 0) for r in results]
        return {
            "avg_mos": round(_mean(mos_scores), 2),
            "mos_distribution": {
                "excellent_4.5+": sum(1 for m in mos_scores if m >= 4.5),
                "good_3.5_4.5": sum(1 for m in mos_scores if 3.5 <= m < 4.5),
                "fair_2.5_3.5": sum(1 for m in mos_scores if 2.5 <= m < 3.5),
                "poor_below_2.5": sum(1 for m in mos_scores if m < 2.5),
            },
            "total_tts": len(results),
        }

    def _aggregate_can(self, results: list) -> dict:
        if not results:
            return {}
        rates = [r.get("match_rate", 0) for r in results]
        return {
            "avg_match_rate": round(_mean(rates), 4),
            "match_rate": round(_mean(rates), 4),
            "total_checks": len(results),
        }

    def _aggregate_ui(self, results: list) -> dict:
        if not results:
            return {}
        rates = [r.get("match_rate", 0) for r in results]
        pass_counts = [r.get("passed", 0) for r in results]
        total_counts = [r.get("total", 0) for r in results]
        return {
            "avg_match_rate": round(_mean(rates), 4),
            "total_passed": sum(pass_counts),
            "total_checks": sum(total_counts),
            "total_cases": len(results),
        }

    def _aggregate_match_results(self, results: list) -> dict:
        """聚合语义、全双工、多轮会话这类 match_rate 型指标。"""
        if not results:
            return {}
        rates = [r.get("match_rate", 0) for r in results]
        aggregate = {
            "avg_match_rate": round(_mean(rates), 4),
            "total_cases": len(results),
            "passed_cases": sum(1 for r in results if r.get("matched", False)),
        }
        metric_keys = sorted({
            key for result in results for key, value in result.items()
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and key not in {"matched", "match_rate", "total", "matched_items"}
            )
        })
        for key in metric_keys:
            values = [r.get(key) for r in results if isinstance(r.get(key), (int, float))]
            if values:
                aggregate[f"avg_{key}"] = round(_mean(values), 4)
        return aggregate

    def _aggregate_capture(self, case_results: list) -> dict:
        playback = [c.get("asr", {}).get("playback", {}) for c in case_results]
        recordings = [c.get("asr", {}).get("recording", {}) for c in case_results]
        videos = [c.get("asr", {}).get("video", {}) for c in case_results]
        return {
            "played_cases": sum(1 for item in playback if item.get("played")),
            "recorded_audio_cases": sum(1 for item in recordings if item.get("recorded")),
            "video_requested_cases": sum(1 for item in videos if item.get("requested")),
            "video_captured_cases": sum(1 for item in videos if item.get("captured")),
        }

    def _build_master_timeline(self, case_results: list) -> dict:
        all_events = []
        for case in case_results:
            tl = case.get("timeline", {})
            for event in tl.get("events", []):
                all_events.append({
                    "case_id": tl.get("case_id", ""),
                    "utterance": tl.get("utterance", ""),
                    **event,
                })
        return {"events": all_events, "total_events": len(all_events)}


class _SimpleWERCalculator:
    """mock 模式使用的轻量 WER/CER 计算器，只依赖标准库编辑距离。"""

    def compute_wer(self, reference: str, hypothesis: str) -> dict:
        ref_words = _tokenize(reference)
        hyp_words = _tokenize(hypothesis)
        distance = _edit_distance(ref_words, hyp_words)
        return {
            "wer": round(distance / max(len(ref_words), 1), 4),
            "insertions": 0,
            "deletions": 0,
            "substitutions": distance,
            "reference_words": len(ref_words),
            "hypothesis_words": len(hyp_words),
            "word_details": [],
        }

    def compute_wer_c(self, reference: str, hypothesis: str) -> dict:
        result = self.compute_wer(reference, hypothesis)
        return {
            "wer_c": result["wer"],
            "weighted_errors": float(result["substitutions"]),
            "weighted_total": float(max(result["reference_words"], 1)),
            "reference_words": result["reference_words"],
        }

    def compute_cer(self, reference: str, hypothesis: str) -> dict:
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        distance = _edit_distance(ref_chars, hyp_chars)
        return {
            "cer": round(distance / max(len(ref_chars), 1), 4),
            "total_errors": distance,
            "reference_chars": len(ref_chars),
        }


class _MockCANSignalMatcher:
    """CAN mock 校验器：只要配置了期望信号，就按全部命中返回。"""

    def verify_signals(self, frames: list[dict], expected_signals: list[dict], timeout_ms: float, start_time_ms: float = 0) -> dict:
        total = sum(len(item.get("signals", {})) for item in expected_signals)
        details = [
            {
                "frame_id": item.get("frame_id"),
                "expected_signals": item.get("signals", {}),
                "matched": True,
                "found_at_ms": start_time_ms + 200,
                "matched_values": item.get("signals", {}),
            }
            for item in expected_signals
        ]
        return {
            "matched": total > 0,
            "match_rate": 1.0 if total else 0,
            "total_signals": total,
            "matched_signals": total,
            "details": details,
        }


class _MockUIDetector:
    """UI mock 校验器：用于没有截图或视觉依赖时跑通报告链路。"""

    def verify_changes(self, before_path: str, after_path: str, expected_changes: list[dict]) -> dict:
        details = [
            {
                "element": item.get("element"),
                "passed": True,
                "state_match": True,
                "value_match": True,
                "expected": item,
                "actual": {
                    "present": item.get("state", "visible") != "invisible",
                    "detected_value": item.get("value", ""),
                    "method": "mock",
                },
            }
            for item in expected_changes
        ]
        return {
            "all_passed": bool(expected_changes),
            "match_rate": 1.0 if expected_changes else 0,
            "passed": len(expected_changes),
            "total": len(expected_changes),
            "details": details,
        }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _mock_full_duplex_actual(spec: dict, actual: dict) -> dict:
    """根据期望全双工配置补齐 mock 实测字段，保证所有指标都可计算。"""
    expected_events = spec.get("expected_events", spec.get("events", [])) or []
    expected_behavior = spec.get("expected_behavior", {}) or {}
    merged = dict(actual or {})
    merged.setdefault("events", expected_events)
    merged.setdefault("behavior", expected_behavior)
    merged.setdefault("took_turn", True)
    merged.setdefault("response_latency_ms", 260)
    merged.setdefault("stop_latency_ms", 180)
    merged.setdefault("overlap_duration_ms", 120)
    merged.setdefault("false_interruption_rate", 0.0)
    if spec.get("scenario") == "user_backchannel":
        merged.setdefault("backchannel_distribution", spec.get("expected_backchannel_distribution", [1.0]))
        merged.setdefault("audio_duration_sec", 10.0)
    return merged


def _mock_dialogue_actual(spec: dict, actual: dict) -> dict:
    """根据期望多轮配置补齐 mock 实测状态，保证 demo 默认通过。"""
    merged = dict(actual or {})
    merged.setdefault("turns", spec.get("turns", []))
    merged.setdefault("final_state", spec.get("expected_final_state", {}))
    merged.setdefault("task_completed", True)
    merged.setdefault("context_carryover_accuracy", 1.0)
    return merged


def _tokenize(text: str) -> list[str]:
    """中文无空格时按字切分，英文或已分词文本按空格切分。"""
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return [part for part in text.split() if part]
    return list(text)


def _edit_distance(left: list[str], right: list[str]) -> int:
    """标准动态规划编辑距离，供 WER/CER mock 指标复用。"""
    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]
