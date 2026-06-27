"""TTS 韵律客观指标：语速、基频、停顿和响度。"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProsodyAnalyzer:
    """分析合成语音的客观韵律指标，并在依赖缺失时给出稳定兜底。"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate

    def analyze(self, audio: np.ndarray, text: Optional[str] = None,
                sample_rate: Optional[int] = None) -> dict:
        """分析单段音频，返回报告中直接展示的韵律指标字典。"""
        sr = sample_rate or self.sample_rate
        audio = _to_mono_float32(audio)
        if audio.size == 0 or sr <= 0:
            return self._empty_result()

        pitch_data = self._analyze_pitch(audio, sr)
        pause_data = self._analyze_pauses(audio, sr)
        loudness_data = self._analyze_loudness(audio)
        speed = self._estimate_speed(audio, sr, text, pause_data)

        return {
            "duration_sec": round(len(audio) / sr, 3),
            "speed_syllables_per_sec": round(speed, 2),
            **pitch_data,
            **pause_data,
            **loudness_data,
        }

    def _analyze_pitch(self, audio: np.ndarray, sr: int) -> dict:
        """估计 F0；当音频太短或 librosa 不可用时返回零值指标。"""
        try:
            import librosa

            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr,
                fill_na=np.nan,
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                return {
                    "pitch_mean_hz": round(float(np.mean(valid_f0)), 1),
                    "pitch_std_hz": round(float(np.std(valid_f0)), 1),
                    "pitch_min_hz": round(float(np.min(valid_f0)), 1),
                    "pitch_max_hz": round(float(np.max(valid_f0)), 1),
                    "pitch_range_hz": round(float(np.max(valid_f0) - np.min(valid_f0)), 1),
                    "voiced_ratio": round(float(np.mean(~np.isnan(f0))), 3),
                }
        except Exception as exc:
            logger.warning("pitch analysis unavailable; using zero metrics (%s)", exc)

        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_min_hz": 0.0,
            "pitch_max_hz": 0.0,
            "pitch_range_hz": 0.0,
            "voiced_ratio": 0.0,
        }

    def _analyze_pauses(self, audio: np.ndarray, sr: int,
                        pause_threshold_db: float = -50.0,
                        min_pause_ms: float = 100.0) -> dict:
        """基于 RMS 能量检测静音停顿，并保留前 50 个停顿片段用于排查。"""
        hop_length = max(1, int(sr * 0.010))
        try:
            import librosa

            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        except Exception as exc:
            logger.warning("librosa RMS unavailable; using lightweight RMS fallback (%s)", exc)
            # 兜底 RMS 使用固定 10ms 帧，避免 librosa 缺失时停顿指标完全无法计算。
            frame_len = hop_length
            num_frames = max(1, int(np.ceil(len(audio) / frame_len)))
            rms = np.array([
                np.sqrt(np.mean(audio[i * frame_len:min((i + 1) * frame_len, len(audio))] ** 2))
                for i in range(num_frames)
            ])

        threshold = 10 ** (pause_threshold_db / 20)
        is_silent = rms < threshold
        min_pause_frames = max(1, int(min_pause_ms / 10))
        pauses = []
        i = 0
        while i < len(is_silent):
            if is_silent[i]:
                start = i
                while i < len(is_silent) and is_silent[i]:
                    i += 1
                end = i
                if end - start >= min_pause_frames:
                    pauses.append((start * 10, end * 10))
            else:
                i += 1

        total_pause_ms = sum(e - s for s, e in pauses)
        audio_length_ms = len(audio) / sr * 1000
        return {
            "pause_count": len(pauses),
            "pause_total_sec": round(total_pause_ms / 1000, 3),
            "pause_mean_sec": round((total_pause_ms / max(len(pauses), 1)) / 1000, 3),
            "pause_ratio": round(total_pause_ms / max(audio_length_ms, 1), 4),
            "pause_segments_ms": [(s, e) for s, e in pauses[:50]],
        }

    def _analyze_loudness(self, audio: np.ndarray) -> dict:
        """计算 RMS、峰值幅度和动态范围。"""
        if audio.size == 0:
            return {"rms_energy": 0.0, "peak_amplitude": 0.0, "dynamic_range_db": 0.0}

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        dynamic_range_db = float(20 * np.log10(peak / rms)) if rms > 1e-12 and peak > 0 else 0.0
        return {
            "rms_energy": round(rms, 6),
            "peak_amplitude": round(peak, 6),
            "dynamic_range_db": round(dynamic_range_db, 2),
        }

    def _estimate_speed(self, audio: np.ndarray, sr: int, text: Optional[str],
                        pause_data: dict) -> float:
        """估计每秒音节数；中文按汉字近似，缺少文本时按默认语速估计。"""
        if text:
            syllables = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
        else:
            syllables = int(len(audio) / sr * 3)
        speech_duration = max(0.001, len(audio) / sr - pause_data.get("pause_total_sec", 0))
        return syllables / speech_duration

    def _empty_result(self) -> dict:
        return {
            "duration_sec": 0.0,
            "speed_syllables_per_sec": 0.0,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_min_hz": 0.0,
            "pitch_max_hz": 0.0,
            "pitch_range_hz": 0.0,
            "voiced_ratio": 0.0,
            "pause_count": 0,
            "pause_total_sec": 0.0,
            "pause_mean_sec": 0.0,
            "pause_ratio": 0.0,
            "pause_segments_ms": [],
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "dynamic_range_db": 0.0,
        }


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """统一为 mono float32，避免立体声输入影响韵律统计。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio
