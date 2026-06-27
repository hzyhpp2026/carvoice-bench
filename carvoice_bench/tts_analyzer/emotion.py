"""情感风格匹配度分析"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmotionMatcher:
    """
    情感风格匹配度分析

    分析 TTS 合成语音的情感表达与预期情感标签的匹配程度。
    支持的情感类别: neutral, happy, sad, angry, surprised, whisper (私语/安静)
    """

    EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "surprised", "whisper"]

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    def analyze(self, audio: np.ndarray, sample_rate: int,
                expected_emotion: str = "neutral") -> dict:
        """
        分析语音情感 & 匹配度

        Returns:
            {
                "predicted_emotion": str,
                "emotion_scores": {标签: 概率},
                "match_score": float (0~1),
                "is_matched": bool,
            }
        """
        if expected_emotion not in self.EMOTION_LABELS:
            expected_emotion = "neutral"

        audio = _to_mono_float32(audio)
        emotion_scores = self._predict_emotion(audio, sample_rate)
        predicted = max(emotion_scores, key=emotion_scores.get)
        match_score = emotion_scores.get(expected_emotion, 0.0)

        return {
            "predicted_emotion": predicted,
            "emotion_probs": emotion_scores,
            "expected_emotion": expected_emotion,
            "match_score": round(float(match_score), 4),
            "is_matched": predicted == expected_emotion,
        }

    def _predict_emotion(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        预测情感概率分布

        使用音频的声学特征（基频、能量、语速、频谱特征）进行
        轻型情感分类，不依赖深度学习模型。
        """
        # 提取声学特征
        features = self._extract_acoustic_features(audio, sr)

        # 简单规则分类（实际项目中替换为训练好的分类器）
        scores = self._rule_based_classify(features)

        return scores

    def _extract_acoustic_features(self, audio: np.ndarray, sr: int) -> dict:
        """提取声学特征"""
        if audio.size == 0 or sr <= 0:
            return self._fallback_features()

        try:
            import librosa

            # 基频统计
            f0, _, _ = librosa.pyin(audio, fmin=65, fmax=2093, sr=sr, fill_na=np.nan)
            valid_f0 = f0[~np.isnan(f0)]

            # 能量
            rms = librosa.feature.rms(y=audio)[0]

            # 零交叉率
            zcr = librosa.feature.zero_crossing_rate(audio)[0]

            # MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

            features = {
                "f0_mean": float(np.nanmean(valid_f0)) if len(valid_f0) > 0 else 0,
                "f0_std": float(np.nanstd(valid_f0)) if len(valid_f0) > 0 else 0,
                "f0_range": float(np.nanmax(valid_f0) - np.nanmin(valid_f0)) if len(valid_f0) > 0 else 0,
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms)),
                "zcr_mean": float(np.mean(zcr)),
                "mfcc_means": [float(np.mean(mfcc[i])) for i in range(min(13, mfcc.shape[0]))],
            }
        except Exception as exc:
            logger.warning("情感声学特征提取不可用，使用回退特征 (%s)", exc)
            features = self._fallback_features()

        return features

    def _fallback_features(self) -> dict:
        return {
            "f0_mean": 200.0, "f0_std": 30.0, "f0_range": 100.0,
            "rms_mean": 0.1, "rms_std": 0.05, "zcr_mean": 0.05,
            "mfcc_means": [0.0] * 13,
        }

    def _rule_based_classify(self, features: dict) -> dict[str, float]:
        """
        基于规则的简单情感分类

        规则:
        - angry: 高基频、高能量、高ZCR
        - happy: 中高基频、高能量、基频范围大
        - sad: 低基频、低能量、低ZCR
        - surprised: 高基频、基频范围大、高ZCR
        - whisper: 低能量、低基频、波动小
        - neutral: 其他情况
        """
        f0_mean = features["f0_mean"]
        f0_std = features["f0_std"]
        f0_range = features["f0_range"]
        rms_mean = features["rms_mean"]
        rms_std = features["rms_std"]
        zcr_mean = features["zcr_mean"]

        # 归一化评分
        scores = {
            "neutral": 0.5,
            "happy": 0.2 + 0.3 * min(1.0, f0_range / 200),
            "sad": 0.2 + 0.3 * (1.0 - min(1.0, f0_mean / 200)),
            "angry": 0.2 + 0.3 * min(1.0, max(0, (f0_mean - 150) / 200)) + 0.2 * min(1.0, rms_mean * 10),
            "surprised": 0.1 + 0.3 * min(1.0, f0_range / 250) + 0.2 * min(1.0, zcr_mean * 20),
            "whisper": 0.2 + 0.3 * (1.0 - min(1.0, rms_mean * 10)) + 0.2 * (1.0 - min(1.0, f0_std / 50)),
        }

        # softmax 归一化
        exp_scores = {k: np.exp(v) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: round(float(v / total), 4) for k, v in exp_scores.items()}


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio
