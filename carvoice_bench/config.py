"""全局配置模型"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """CarVoice Bench 全局配置"""

    # ASR 配置
    asr_model: str = "whisper-base-zh"
    asr_device: str = "cpu"
    asr_sample_rate: int = 16000
    asr_language: str = "zh"

    # TTS 配置
    tts_mos_model_path: Optional[str] = None
    tts_sample_rate: int = 24000

    # CAN 配置
    can_db_path: Optional[str] = None
    can_bus_type: str = "socketcan"
    can_channel: str = "can0"

    # UI 配置
    ui_confidence_threshold: float = 0.45
    ui_template_path: Optional[str] = None
    ui_yolo_model_path: Optional[str] = None

    # 对齐引擎
    clock_ntp_server: str = "ntp.aliyun.com"
    audio_pts_interpolation: str = "linear"

    # 报告
    report_title: str = "CarVoice Bench 评测报告"
    report_company: str = ""
    report_language: str = "zh-CN"
    enable_scoring: bool = True

    # 通用
    timeout_ms: int = 5000
    output_dir: str = "./carvoice_report"
    verbose: bool = False
    debug: bool = False
    seed: int = 42
    mock_mode: bool = False

    # 在线测试 / 云端服务
    online_mode: bool = False
    cloud_provider: str = "aliyun"
    env_file: str = ".env"
    cloud_asr_model: str = "paraformer-realtime-v2"
    cloud_tts_model: str = "cosyvoice-v3-flash"
    cloud_tts_voice: str = "longxiaochun_v2"
    cloud_tts_format: str = "mp3"
    online_play_audio: bool = False
    online_record_seconds: float = 0.0
    online_record_sample_rate: int = 16000
    online_record_ui_video: bool = False
    online_record_cabin_video: bool = False
    online_case_pause_seconds: float = 0.0
    online_camera_index: int = 0
    online_video_fps: float = 20.0

    # 语义解析
    semantic_parser: str = "rule"
    semantic_rules_path: Optional[str] = None
    semantic_cloud_model: str = "qwen-plus"

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}  # type: ignore
