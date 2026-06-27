"""音频读写、重采样、能量和 VAD 等通用工具函数。"""

import logging
import wave

import numpy as np

logger = logging.getLogger(__name__)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件并返回 ``(audio_samples, sample_rate)``。"""
    try:
        import soundfile as sf

        audio, sr = sf.read(path)
        return _to_float32(audio), int(sr)
    except ImportError:
        # 没有 soundfile 时回退到标准库 wave；该路径只支持最常见的 16-bit PCM。
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sample_width != 2:
            raise RuntimeError("stdlib WAV fallback only supports 16-bit PCM")
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        return _to_float32(audio), sr


def convert_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """把音频重采样到目标采样率，供 ASR/TTS 指标统一输入。"""
    audio = _to_float32(audio)
    if orig_sr == target_sr or audio.size == 0:
        return audio
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("sample rates must be positive")

    num_samples = max(1, int(round(len(audio) * target_sr / orig_sr)))
    try:
        from scipy.signal import resample

        return resample(audio, num_samples).astype(np.float32)
    except Exception as exc:
        logger.warning("scipy resample unavailable; using linear interpolation (%s)", exc)
        # scipy 不可用时用线性插值兜底，精度较低但可以保证指标链路不中断。
        old_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        new_x = np.linspace(0.0, 1.0, num=num_samples, endpoint=False)
        return np.interp(new_x, old_x, audio).astype(np.float32)


def audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """返回音频时长，单位秒。"""
    if sample_rate <= 0:
        return 0.0
    return len(audio) / sample_rate


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0,
                 min_interval: float = 0.1) -> np.ndarray:
    """裁剪首尾静音，用于降低空白段对客观指标的影响。"""
    audio = _to_float32(audio)
    if audio.size == 0:
        return audio

    try:
        import librosa

        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=-threshold_db,
            frame_length=max(1, int(sr * 0.025)),
            hop_length=max(1, int(sr * 0.010)),
        )
        return trimmed.astype(np.float32)
    except Exception as exc:
        logger.warning("librosa trim unavailable; using amplitude trim fallback (%s)", exc)
        # librosa 不可用时按幅度阈值裁剪，保留少量 pad 避免切掉语音边缘。
        threshold = 10 ** (threshold_db / 20)
        active = np.where(np.abs(audio) > threshold)[0]
        if active.size == 0:
            return np.array([], dtype=np.float32)
        pad = max(0, int(sr * min_interval))
        start = max(0, int(active[0]) - pad)
        end = min(len(audio), int(active[-1]) + pad + 1)
        return audio[start:end]


def compute_rms(audio: np.ndarray) -> float:
    """计算 RMS 能量。"""
    audio = _to_float32(audio)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """计算信噪比，单位 dB；输入长度不一致时按较短长度对齐。"""
    clean = _to_float32(clean)
    noisy = _to_float32(noisy)
    n = min(len(clean), len(noisy))
    if n == 0:
        return 0.0
    clean = clean[:n]
    noisy = noisy[:n]
    noise = noisy - clean
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2))
    if signal_power < 1e-12:
        return 0.0
    if noise_power < 1e-12:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def find_voice_activity(audio: np.ndarray, sr: int,
                        threshold: float = 0.01) -> list[tuple[int, int]]:
    """返回简单能量 VAD 的活跃语音片段 ``[(start_sample, end_sample), ...]``。"""
    audio = _to_float32(audio)
    if audio.size == 0:
        return []
    energy = np.abs(audio)
    above = energy > threshold
    changes = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _to_float32(audio: np.ndarray) -> np.ndarray:
    """统一为 mono float32，避免不同音频库读出的 dtype/通道数影响指标。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio
