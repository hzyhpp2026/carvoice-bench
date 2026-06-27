"""基于 UTMOSv2 的 TTS MOS 预测器。

优先使用 UTMOSv2 开源模型计算 MOS；如果可选依赖或模型权重不可用，
会回退到本地轻量启发式估计，保证评测主流程仍可运行。
"""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MOSPredictor:
    """为合成语音预测 MOS、自然度和平滑度。"""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self._utmos_model = None
        self._load_models()

    def _load_models(self):
        """在环境可用时加载 UTMOSv2 预训练模型。"""
        try:
            import utmosv2

            self._utmos_model = utmosv2.create_model(pretrained=True)
            if hasattr(self._utmos_model, "to"):
                self._utmos_model.to(self.device)
            if hasattr(self._utmos_model, "eval"):
                self._utmos_model.eval()
            logger.info("UTMOSv2 MOS model loaded")
        except Exception as exc:
            self._utmos_model = None
            logger.warning("UTMOSv2 unavailable (%s); using heuristic MOS fallback", exc)

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict:
        """预测单段内存音频的 MOS。

        返回字段包括 mos、naturalness、smoothness、confidence_interval 和 method。
        """
        if self._utmos_model is not None:
            try:
                return self._predict_utmos_array(audio, sample_rate)
            except Exception as exc:
                logger.warning("UTMOSv2 array prediction failed (%s); using fallback", exc)
        return self._predict_heuristic(audio, sample_rate)

    def predict_file(self, audio_path: str) -> dict:
        """从音频文件路径预测 MOS，这是 UTMOSv2 当前最稳定的调用方式。"""
        if self._utmos_model is not None:
            try:
                score = self._utmos_model.predict(input_path=str(audio_path))
                return self._format_utmos_result(score)
            except Exception as exc:
                logger.warning("UTMOSv2 file prediction failed (%s); using fallback", exc)

        try:
            import soundfile as sf

            audio, sample_rate = sf.read(str(audio_path))
            return self._predict_heuristic(audio, sample_rate)
        except Exception as exc:
            logger.warning("MOS fallback failed for %s (%s)", audio_path, exc)
            return self._format_score(0.0, method="unavailable")

    def predict_batch(self, audio_list: list[tuple[np.ndarray, int]]) -> list[dict]:
        """批量预测多段音频的 MOS。"""
        return [self.predict(audio, sr) for audio, sr in audio_list]

    def _predict_utmos_array(self, audio: np.ndarray, sample_rate: int) -> dict:
        """把内存音频临时写成 WAV 后交给 UTMOSv2 推理。"""
        audio = _to_mono_float32(audio)

        # UTMOSv2 不同版本的数据接口不完全一致，临时 WAV 能让封装在多数版本下保持稳定。
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for UTMOSv2 array prediction") from exc

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(str(tmp_path), audio, sample_rate)
            score = self._utmos_model.predict(input_path=str(tmp_path))
            return self._format_utmos_result(score)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _format_utmos_result(self, raw_score) -> dict:
        score = _scalar(raw_score)
        return self._format_score(score, method="utmosv2")

    def _format_score(self, mos: float, method: str) -> dict:
        if not np.isfinite(mos):
            mos = 0.0
        mos = max(1.0, min(5.0, float(mos))) if mos > 0 else 0.0
        if mos == 0.0:
            naturalness = 0.0
            smoothness = 0.0
            ci = (0.0, 0.0)
        else:
            naturalness = mos
            smoothness = mos
            ci = (round(max(1.0, mos - 0.3), 2), round(min(5.0, mos + 0.3), 2))

        return {
            "mos": round(mos, 2),
            "naturalness": round(naturalness, 2),
            "smoothness": round(smoothness, 2),
            "confidence_interval": ci,
            "method": method,
        }

    def _predict_heuristic(self, audio: np.ndarray, sr: int) -> dict:
        """基于 RMS、峰值和过零率的兜底 MOS 估计。"""
        audio = _to_mono_float32(audio)
        if audio.size == 0:
            return self._format_score(0.0, method="heuristic_fallback")

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        zero_crossing = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2) if audio.size > 1 else 0.0

        rms_score = min(5.0, rms * 10 + 3.0)
        zcr_score = max(1.0, 5.0 - zero_crossing * 50)
        peak_score = min(5.0, peak * 5 + 2.0)
        mos = max(1.0, min(5.0, rms_score * 0.4 + zcr_score * 0.3 + peak_score * 0.3))

        result = self._format_score(mos, method="heuristic_fallback")
        result["naturalness"] = round(result["mos"] * 0.95, 2)
        result["smoothness"] = round(result["mos"] * 0.90, 2)
        result["confidence_interval"] = (
            round(max(1.0, result["mos"] - 0.5), 2),
            round(min(5.0, result["mos"] + 0.5), 2),
        )
        return result


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """统一把音频转为 mono float32，避免模型和指标函数收到异常形状。"""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio


def _scalar(value) -> float:
    """把模型输出的标量、tensor 或嵌套 list 规整成 float。"""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        while isinstance(value, list) and value:
            value = value[0]
        return float(value)
    return float(value)
