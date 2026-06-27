"""CAN 日志解析器 — 支持 ASC / BLF / CSV 格式"""

import re
import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CANLogParser:
    """
    CAN 总线日志解析器

    支持的格式:
    - .asc: Vector ASC 日志格式 (CANalyzer/CANoe)
    - .blf: Vector BLF 二进制日志格式
    - .csv: 通用 CSV 格式 (时间戳, ID, 数据)
    - .log: Vector CANoe 文本日志 (向前兼容)
    """

    def __init__(self, bus_type: str = "vector", channel: str = "can0"):
        self.bus_type = bus_type
        self.channel = channel

    def parse(self, path: str) -> list[dict]:
        """
        解析 CAN 日志文件

        Returns:
            [{
                "timestamp_s": float,      # 时间戳(秒)
                "timestamp_ms": float,     # 时间戳(毫秒)
                "frame_id": int,           # CAN 帧ID (十进制)
                "frame_id_hex": str,       # CAN 帧ID (十六进制)
                "data": bytes,             # 数据字节
                "data_hex": str,           # 数据十六进制字符串
                "dlc": int,                # 数据长度
                "channel": str,            # CAN 通道
                "flags": str,              # 帧标志
                "is_extended": bool,       # 是否扩展帧
            }, ...]
        """
        path = str(path)
        if path.endswith(".asc"):
            return self._parse_asc(path)
        elif path.endswith(".blf"):
            return self._parse_blf(path)
        elif path.endswith(".csv"):
            return self._parse_csv(path)
        elif path.endswith(".log"):
            return self._parse_log(path)
        else:
            raise ValueError(f"不支持的 CAN 日志格式: {path}")

    def _parse_asc(self, path: str) -> list[dict]:
        """解析 ASC 格式日志"""
        frames = []
        if not Path(path).exists():
            logger.warning("ASC 文件不存在: %s", path)
            return frames
        # ASC 行格式:
        # 时间戳  通道  ID  Tx/Rx  数据长度  数据字节
        # e.g. 1.234567 1 2A1 Rx d 8 02 1A 00 00 00 00 00 00
        pattern = re.compile(
            r"^(\d+\.\d+)\s+"        # 时间戳
            r"(\d+)\s+"              # 通道
            r"([0-9A-Fa-f]+)\s+"     # 帧ID
            r"(Tx|Rx)\s+"            # 方向
            r"[dr]\s+"               # 数据帧类型标记
            r"(\d+)"                 # DLC
            r"((?:\s+[0-9A-Fa-f]{2})+)"  # 数据字节
        )

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                match = pattern.match(line)
                if match:
                    try:
                        timestamp_s = float(match.group(1))
                        channel = int(match.group(2))
                        frame_id_hex = match.group(3)
                        direction = match.group(4)
                        dlc = int(match.group(5))
                        data_str = match.group(6).strip()
                        data_bytes = bytes.fromhex(data_str)

                        frames.append({
                            "timestamp_s": timestamp_s,
                            "timestamp_ms": timestamp_s * 1000,
                            "frame_id": int(frame_id_hex, 16),
                            "frame_id_hex": frame_id_hex.upper(),
                            "data": data_bytes,
                            "data_hex": data_bytes.hex().upper(),
                            "dlc": dlc,
                            "channel": str(channel),
                            "direction": direction,
                            "flags": "",
                            "is_extended": len(frame_id_hex) > 3,
                        })
                    except (ValueError, IndexError):
                        continue

        logger.info("解析 ASC 文件: %d 帧 (%s)", len(frames), path)
        return frames

    def _parse_blf(self, path: str) -> list[dict]:
        """解析 BLF 二进制格式"""
        frames = []
        try:
            import can
            from can.io.blf import BLFReader

            with BLFReader(path) as reader:
                for msg in reader:
                    frames.append({
                        "timestamp_s": msg.timestamp,
                        "timestamp_ms": msg.timestamp * 1000,
                        "frame_id": msg.arbitration_id,
                        "frame_id_hex": f"{msg.arbitration_id:X}",
                        "data": msg.data,
                        "data_hex": msg.data.hex().upper(),
                        "dlc": msg.dlc,
                        "channel": str(msg.channel) if msg.channel else "",
                        "direction": "Rx",
                        "flags": "",
                        "is_extended": msg.is_extended_id,
                    })
        except ImportError:
            logger.warning("python-can 未安装，BLF 解析回退到原始字节模式")
            frames = self._parse_blf_raw(path)
        except Exception as e:
            logger.error("BLF 解析失败: %s", e)

        logger.info("解析 BLF 文件: %d 帧 (%s)", len(frames), path)
        return frames

    def _parse_blf_raw(self, path: str) -> list[dict]:
        """回退：解析 BLF 原始字节"""
        frames = []
        with open(path, "rb") as f:
            raw = f.read()
        logger.info("BLF 文件大小: %d 字节", len(raw))
        return frames  # 实际项目中应实现完整 BLF 协议解析

    def _parse_csv(self, path: str) -> list[dict]:
        """解析 CSV 格式"""
        frames = []
        if not Path(path).exists():
            logger.warning("CSV 文件不存在: %s", path)
            return frames
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row.get("timestamp_ms", row.get("time", 0)))
                    fid = int(row.get("frame_id", row.get("id", 0)), 16)
                    data_hex = row.get("data", row.get("data_hex", ""))
                    data_bytes = bytes.fromhex(data_hex) if data_hex else b""

                    frames.append({
                        "timestamp_s": ts / 1000.0,
                        "timestamp_ms": ts,
                        "frame_id": fid,
                        "frame_id_hex": f"{fid:X}",
                        "data": data_bytes,
                        "data_hex": data_bytes.hex().upper(),
                        "dlc": len(data_bytes),
                        "channel": row.get("channel", "0"),
                        "direction": "Rx",
                        "flags": "",
                        "is_extended": len(f"{fid:X}") > 3,
                    })
                except (ValueError, KeyError):
                    continue

        logger.info("解析 CSV 文件: %d 帧 (%s)", len(frames), path)
        return frames

    def _parse_log(self, path: str) -> list[dict]:
        """解析 CANoe LOG 格式 (向前兼容)"""
        frames = []
        pattern = re.compile(
            r"^.*?(\d+\.\d{6})\s+"   # 时间戳
            r"(\d+)\s+"              # 通道
            r"([0-9A-Fa-f]+)\s+"     # 帧ID
            r"(?:\w+\s+)"            # 报文名称
            r"(\w+)\s+"              # 方向
            r"\d+\s+"                # DLC
            r"((?:[0-9A-Fa-f]{2}\s*)+)"  # 数据
        )

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    try:
                        frames.append({
                            "timestamp_s": float(match.group(1)),
                            "timestamp_ms": float(match.group(1)) * 1000,
                            "frame_id": int(match.group(3), 16),
                            "frame_id_hex": match.group(3).upper(),
                            "data": bytes.fromhex(match.group(5).replace(" ", "")),
                            "dlc": len(match.group(5).split()),
                            "channel": match.group(2),
                            "direction": match.group(4),
                        })
                    except ValueError:
                        continue

        logger.info("解析 LOG 文件: %d 帧 (%s)", len(frames), path)
        return frames

    def filter_by_frame_id(self, frames: list[dict], target_ids: list[int]) -> list[dict]:
        """按帧ID过滤"""
        return [f for f in frames if f["frame_id"] in target_ids]

    def filter_by_time_range(self, frames: list[dict], start_ms: float, end_ms: float) -> list[dict]:
        """按时间范围过滤"""
        return [f for f in frames if start_ms <= f["timestamp_ms"] <= end_ms]

    def extract_signals(self, frames: list[dict], dbc_parser) -> list[dict]:
        """使用 DBC 解析器提取信号"""
        return [dbc_parser.decode(f) for f in frames]
