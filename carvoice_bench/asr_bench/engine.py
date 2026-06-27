"""ASR 引擎封装 — 支持 Whisper / Paraformer / Kaldi 三种后端"""

import time
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASREngine:
    """ASR 引擎统一接口"""

    def __init__(self, model_name: str = "whisper-base-zh", device: str = "cpu",
                 sample_rate: int = 16000, language: str = "zh"):
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.language = language
        self._model = None
        self._processor = None
        self._backend_available = True
        self._load_model()

    def _load_model(self):
        """加载 ASR 模型"""
        logger.info("加载 ASR 模型: %s (device=%s)", self.model_name, self.device)
        if "whisper" in self.model_name.lower():
            self._load_whisper()
        elif "paraformer" in self.model_name.lower():
            self._load_paraformer()
        elif "kaldi" in self.model_name.lower():
            self._load_kaldi()
        else:
            raise ValueError(f"不支持的 ASR 模型: {self.model_name}")

    def _load_whisper(self):
        """加载 Whisper 模型"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
        except ImportError as exc:
            self._backend_available = False
            logger.warning("transformers 未安装，ASR 引擎降级为空识别后端: %s", exc)
            return

        model_id = {
            "whisper-tiny-zh": "openai/whisper-tiny",
            "whisper-base-zh": "openai/whisper-base",
            "whisper-small-zh": "openai/whisper-small",
        }.get(self.model_name, self.model_name)

        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self._model.to(self.device)
        self._model.eval()
        logger.info("  Whisper 模型加载完成: %s", model_id)

    def _load_paraformer(self):
        """加载 Paraformer 模型 (通过 FunASR)"""
        try:
            from funasr import AutoModel
            self._model = AutoModel(
                model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                device=self.device,
            )
            logger.info("  Paraformer 模型加载完成")
        except ImportError:
            logger.warning("FunASR 未安装，回退到 Whisper 模式")
            self._load_whisper()

    def _load_kaldi(self):
        """加载 Kaldi 模型 (通过 sherpa-onnx)"""
        try:
            import sherpa_onnx
            # 示例配置，实际使用需替换为具体模型路径
            self._model = sherpa_onnx.OfflineRecognizer.from_zipformer(
                encoder="model.int8.onnx",
                decoder="decoder.int8.onnx",
                joiner="joiner.int8.onnx",
                tokens="tokens.txt",
            )
            logger.info("  Kaldi/Sherpa-ONNX 模型加载完成")
        except ImportError:
            logger.warning("sherpa_onnx 未安装，回退到 Whisper 模式")
            self._load_whisper()

    def transcribe(self, audio: np.ndarray, sr: Optional[int] = None) -> tuple[str, float, float]:
        """
        语音识别

        Returns:
            (text, confidence, elapsed_seconds)
        """
        if sr is None:
            sr = self.sample_rate

        # 重采样
        if sr != self.sample_rate:
            from scipy.signal import resample
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = resample(audio, num_samples)

        start = time.perf_counter()
        if not self._backend_available or self._model is None:
            elapsed = time.perf_counter() - start
            return "", 0.0, elapsed

        if "whisper" in self.model_name.lower():
            text, conf = self._transcribe_whisper(audio)
        elif "paraformer" in self.model_name.lower():
            text, conf = self._transcribe_paraformer(audio)
        elif "kaldi" in self.model_name.lower():
            text, conf = self._transcribe_kaldi(audio)
        else:
            text, conf = "", 0.0

        elapsed = time.perf_counter() - start
        return text, conf, elapsed

    def _transcribe_whisper(self, audio: np.ndarray) -> tuple[str, float]:
        """Whisper 推理"""
        import torch

        input_features = self._processor(
            audio, sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features)
        transcription = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        # 计算置信度（平均 logits 的 softmax 概率）
        conf = 0.85  # 实际项目中通过 logits 计算
        return transcription.strip(), conf

    def _transcribe_paraformer(self, audio: np.ndarray) -> tuple[str, float]:
        """Paraformer 推理"""
        result = self._model.generate(input=audio, cache={})
        text = result[0]["text"] if result else ""
        conf = result[0].get("confidence", 0.8) if result else 0.0
        return text.strip(), float(conf)

    def _transcribe_kaldi(self, audio: np.ndarray) -> tuple[str, float]:
        """Kaldi 推理"""
        stream = self._model.create_stream()
        stream.accept_waveform(self.sample_rate, (audio * 32767).astype(np.int16).tobytes())
        stream.input_finished()
        self._model.decode_stream(stream)
        text = stream.result.text
        conf = stream.result.confidence if hasattr(stream.result, "confidence") else 0.8
        return text.strip(), float(conf)
