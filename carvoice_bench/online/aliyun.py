"""Alibaba Cloud DashScope adapters for online ASR/TTS tests."""

from __future__ import annotations

import logging
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from carvoice_bench.config import Config
from carvoice_bench.utils.env import first_env, load_env_file

logger = logging.getLogger(__name__)


class AliyunDashScopeClient:
    """Thin wrapper around DashScope CosyVoice TTS and Paraformer ASR."""

    def __init__(self, config: Config):
        self.config = config
        load_env_file(config.env_file, override=True)
        self.api_key = first_env("DASHSCOPE_API_KEY", "ALIYUN_API_KEY", "ALIYUN_DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Aliyun online mode requires DASHSCOPE_API_KEY in .env "
                "(ALIYUN_API_KEY and ALIYUN_DASHSCOPE_API_KEY are also accepted)."
            )
        self._dashscope = self._load_dashscope()
        self.config.cloud_asr_model = (
            self.config.cloud_asr_model
            or first_env("ALIYUN_ASR_MODEL", "DASHSCOPE_ASR_MODEL")
            or Config.cloud_asr_model
        )
        self.config.cloud_tts_model = (
            self.config.cloud_tts_model
            or first_env("ALIYUN_TTS_MODEL", "DASHSCOPE_TTS_MODEL")
            or Config.cloud_tts_model
        )
        self.config.cloud_tts_voice = (
            self.config.cloud_tts_voice
            or first_env("ALIYUN_TTS_VOICE", "DASHSCOPE_TTS_VOICE")
            or Config.cloud_tts_voice
        )
        self.config.cloud_tts_format = (
            self.config.cloud_tts_format
            or first_env("ALIYUN_TTS_FORMAT", "DASHSCOPE_TTS_FORMAT")
            or Config.cloud_tts_format
        )
        self._dashscope.api_key = self.api_key
        websocket_url = first_env("DASHSCOPE_WEBSOCKET_URL", "ALIYUN_DASHSCOPE_WEBSOCKET_URL")
        if websocket_url:
            self._dashscope.base_websocket_api_url = websocket_url

    @staticmethod
    def _load_dashscope():
        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError(
                "DashScope SDK is not installed. Install it with "
                '`pip install -e ".[online]"` or `pip install dashscope`.'
            ) from exc
        return dashscope

    def synthesize_to_file(self, text: str, output_path: str | Path) -> dict[str, Any]:
        """Generate speech from text and save it to ``output_path``."""
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        audio_format = _dashscope_tts_format(AudioFormat, self.config.cloud_tts_format)
        kwargs = {
            "model": self.config.cloud_tts_model,
            "voice": self.config.cloud_tts_voice,
        }
        if audio_format is not None:
            kwargs["format"] = audio_format
        synthesizer = SpeechSynthesizer(**kwargs)
        try:
            audio = synthesizer.call(text)
        except Exception as exc:
            raise RuntimeError(_format_aliyun_error("TTS", exc)) from exc
        if not audio:
            raise RuntimeError("Aliyun TTS returned empty audio.")
        output_path.write_bytes(audio)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Aliyun TTS generated %s in %.1fms", output_path, elapsed_ms)
        return {
            "provider": "aliyun_dashscope",
            "model": self.config.cloud_tts_model,
            "voice": self.config.cloud_tts_voice,
            "format": self.config.cloud_tts_format,
            "text": text,
            "audio_path": str(output_path),
            "latency_ms": round(elapsed_ms, 2),
            "request_id": _safe_call(synthesizer, "get_last_request_id"),
            "first_package_delay_ms": _safe_call(synthesizer, "get_first_package_delay"),
            "bytes": len(audio),
        }

    def transcribe_file(self, audio_path: str | Path) -> dict[str, Any]:
        """Recognize a local audio file with DashScope Paraformer realtime ASR."""
        from dashscope.audio.asr import Recognition
        import soundfile as sf

        # 读取音频实际采样率，确保与声明的 sample_rate 一致
        try:
            _, actual_sr = sf.read(str(audio_path), stop=0)
        except Exception:
            actual_sr = self.config.online_record_sample_rate

        start = time.perf_counter()
        recognizer = Recognition(
            model=self.config.cloud_asr_model,
            format=_audio_format(audio_path),
            sample_rate=actual_sr,
            callback=None,
        )
        try:
            result = recognizer.call(str(audio_path))
        except Exception as exc:
            raise RuntimeError(_format_aliyun_error("ASR", exc)) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        if getattr(result, "status_code", HTTPStatus.OK) != HTTPStatus.OK:
            raise RuntimeError(getattr(result, "message", "Aliyun ASR failed."))

        sentence = result.get_sentence() if hasattr(result, "get_sentence") else {}
        text = _merge_sentences([sentence])
        return {
            "provider": "aliyun_dashscope",
            "model": self.config.cloud_asr_model,
            "audio_path": str(audio_path),
            "text": text,
            "confidence": 0.0,
            "processing_latency_ms": round(elapsed_ms, 2),
            "events": [sentence] if sentence else [],
        }


def _audio_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"wav", "mp3", "pcm", "opus", "speex", "aac", "amr"}:
        return suffix
    return "wav"


def tts_suffix(tts_format: str | None) -> str:
    fmt = (tts_format or Config.cloud_tts_format).lower()
    if fmt.startswith("wav"):
        return ".wav"
    if fmt.startswith("pcm"):
        return ".pcm"
    return ".mp3"


def _dashscope_tts_format(audio_format_cls, value: str | None):
    fmt = (value or "").lower()
    if not fmt or fmt == "mp3":
        return None
    if fmt in {"wav", "wav_24000hz_mono_16bit"}:
        return getattr(audio_format_cls, "WAV_24000HZ_MONO_16BIT", None)
    if fmt == "pcm_24000hz_mono_16bit":
        return getattr(audio_format_cls, "PCM_24000HZ_MONO_16BIT", None)
    return None


def _safe_call(obj, method_name: str):
    method = getattr(obj, method_name, None)
    if not callable(method):
        return None
    try:
        return method()
    except Exception:
        return None


def _merge_sentences(sentences: list[Any]) -> str:
    texts = []
    for sentence in sentences:
        if isinstance(sentence, dict):
            text = sentence.get("text") or sentence.get("sentence") or ""
        else:
            text = str(sentence)
        if text:
            texts.append(text.strip())
    return "".join(texts).strip()


def _format_aliyun_error(service: str, exc: Exception) -> str:
    message = str(exc)
    if "InvalidApiKey" in message or "Invalid API-key" in message or "401" in message:
        return (
            f"Aliyun {service} authentication failed: invalid API key. "
            "Check DASHSCOPE_API_KEY in .env, make sure the key belongs to Alibaba Cloud Model Studio, "
            "and rerun with --env-file .env."
        )
    return f"Aliyun {service} request failed: {message}"
