"""ASR 延迟毫秒级打点"""

import time
import logging
from typing import Optional

import numpy as np

from carvoice_bench.utils.audio import find_voice_activity

logger = logging.getLogger(__name__)


class ASRLatencyMeasurer:
    """
    ASR 延迟测量器

    支持三种模式:
    1. 音频注入模式: 注入音频 -> ASR 引擎
    2. 麦克风实录模式: 录音时间戳 -> ASR 引擎
    3. 日志回放模式: 从 CAN/日志中提取时间戳

    延迟分段:
    - E2E Latency: 用户说完 -> ASR 给出最终结果
    - Processing Latency: ASR 引擎内部计算时间
    - First-word Latency: 用户说完 -> 第一个词输出
    """

    def __init__(self):
        self._timestamps: list[dict] = []

    def measure_injection(
        self,
        audio: np.ndarray,
        sample_rate: int,
        asr_engine,
    ) -> dict:
        """
        音频注入模式的延迟测量

        返回:
            {
                "e2e_latency_ms": 端到端延迟(ms),
                "processing_latency_ms": 处理延迟(ms),
                "audio_duration_ms": 音频时长(ms),
                "vad_detected": VAD检测结果
            }
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        # 检测语音活跃区间
        vad_segments = find_voice_activity(audio, sample_rate)
        speaker_end_time = 0.0
        if vad_segments:
            # 用户说完的时刻（最后一个语音帧结束）
            last_start, last_end = vad_segments[-1]
            speaker_end_time = last_end / sample_rate * 1000  # ms

        audio_duration_ms = len(audio) / sample_rate * 1000

        if asr_engine is None:
            result = {
                "e2e_latency_ms": 0.0,
                "processing_latency_ms": 0.0,
                "audio_duration_ms": round(audio_duration_ms, 2),
                "speaker_end_time_ms": round(speaker_end_time, 2),
                "first_word_latency_ms": 0.0,
                "text": "",
                "confidence": 0.0,
                "vad_segments": [
                    {"start_ms": round(s / sample_rate * 1000, 2),
                     "end_ms": round(e / sample_rate * 1000, 2)}
                    for s, e in vad_segments
                ],
                "method": "no_asr_engine",
            }
            self._timestamps.append(result)
            return result

        # 执行 ASR
        from carvoice_bench.utils.audio import convert_sample_rate

        asr_audio = convert_sample_rate(audio, sample_rate, asr_engine.sample_rate)

        t0 = time.perf_counter()
        text, confidence, elapsed = asr_engine.transcribe(asr_audio)
        t1 = time.perf_counter()

        processing_latency_ms = elapsed * 1000
        total_wall_time = (t1 - t0) * 1000

        result = {
            "e2e_latency_ms": round(total_wall_time, 2),
            "processing_latency_ms": round(processing_latency_ms, 2),
            "audio_duration_ms": round(audio_duration_ms, 2),
            "speaker_end_time_ms": round(speaker_end_time, 2),
            "first_word_latency_ms": round(processing_latency_ms, 2),
            "text": text,
            "confidence": confidence,
            "vad_segments": [
                {"start_ms": round(s / sample_rate * 1000, 2),
                 "end_ms": round(e / sample_rate * 1000, 2)}
                for s, e in vad_segments
            ],
        }
        self._timestamps.append(result)
        logger.info("  ASR延迟: E2E=%.1fms | 处理=%.1fms | 音频=%.1fms",
                     result["e2e_latency_ms"], result["processing_latency_ms"],
                     result["audio_duration_ms"])
        return result

    def measure_replay(self, utterance_timestamps: list[float],
                       asr_result_timestamps: list[float]) -> dict:
        """
        日志回放模式的延迟测量

        Args:
            utterance_timestamps: 用户发出语音的时间戳列表 (ms)
            asr_result_timestamps: ASR 返回结果的时间戳列表 (ms)

        Returns: 延迟统计
        """
        if len(utterance_timestamps) != len(asr_result_timestamps):
            logger.warning("时间戳数量不匹配: utterance=%d, asr=%d",
                           len(utterance_timestamps), len(asr_result_timestamps))

        min_len = min(len(utterance_timestamps), len(asr_result_timestamps))
        latencies = [
            asr_result_timestamps[i] - utterance_timestamps[i]
            for i in range(min_len)
        ]

        result = {
            "e2e_latency_ms": np.mean(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0,
            "sample_count": len(latencies),
        }
        self._timestamps.append(result)
        return result

    def get_all_measurements(self) -> list[dict]:
        """获取所有测量记录"""
        return self._timestamps

    def reset(self):
        """重置测量记录"""
        self._timestamps.clear()
