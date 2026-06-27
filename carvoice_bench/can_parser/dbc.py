"""DBC 信号解析器 — 将 CAN 数据帧解码为物理信号值"""

import logging
import struct
from typing import Optional

logger = logging.getLogger(__name__)


class DBCParser:
    """
    DBC (CAN Database) 解析器

    将 CAN 帧的原始数据字节解码为物理信号值。
    支持:
    - Intel / Motorola 字节序
    - 有符号/无符号整型
    - IEEE 754 浮点数
    - 值表枚举转换
    - 缩放因子和偏移量
    """

    def __init__(self, dbc_path: Optional[str] = None):
        self.signals: dict[int, list[dict]] = {}  # frame_id -> [signal_def, ...]
        self.frame_names: dict[int, str] = {}
        self.value_tables: dict[str, dict[int, str]] = {}
        if dbc_path:
            self.load(dbc_path)

    def load(self, path: str):
        """加载 DBC 文件"""
        try:
            import cantools
            db = cantools.database.load_file(str(path))
            for msg in db.messages:
                self.frame_names[msg.frame_id] = msg.name
                sig_list = []
                for sig in msg.signals:
                    sig_def = {
                        "name": sig.name,
                        "start_bit": sig.start_bit,
                        "length": sig.length,
                        "byte_order": "intel" if sig.byte_order == "little_endian" else "motorola",
                        "is_signed": sig.is_signed,
                        "is_float": getattr(sig, "is_float", False),
                        "scale": sig.scale,
                        "offset": sig.offset,
                        "unit": sig.unit or "",
                        "minimum": sig.minimum,
                        "maximum": sig.maximum,
                        "choices": sig.choices or {},
                        "comment": sig.comment or "",
                    }
                    sig_list.append(sig_def)
                self.signals[msg.frame_id] = sig_list
            logger.info("DBC 加载完成: %d 条报文, %s", len(self.signals), path)
        except ImportError:
            logger.warning("cantools 未安装，使用简化的 DBC 解析")
            self._load_simple_dbc(path)
        except Exception as e:
            logger.warning("DBC 解析失败 (%s)，使用空配置", e)

    def _load_simple_dbc(self, path: str):
        """纯 Python DBC 解析（无 cantools 依赖）"""
        import re

        bo_pattern = re.compile(r"BO_\s+(\d+)\s+(\w+):\s+\d+\s+\w+")
        sg_pattern = re.compile(
            r'SG_\s+(\w+)(?:\s+\w+)?\s*:\s*(\d+)\|(\d+)@(\d)([+-])\s+'
            r'\(([^,]+),([^)]+)\)\s+\[([^\]]*)\]\s+"([^"]*)".*'
        )
        current_frame_id = None

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                bo_match = bo_pattern.match(line)
                if bo_match:
                    current_frame_id = int(bo_match.group(1))
                    self.frame_names[current_frame_id] = bo_match.group(2)
                    self.signals.setdefault(current_frame_id, [])
                    continue

                sig_match = sg_pattern.match(line)
                if sig_match and current_frame_id is not None:
                    self.signals[current_frame_id].append({
                        "name": sig_match.group(1),
                        "start_bit": int(sig_match.group(2)),
                        "length": int(sig_match.group(3)),
                        "byte_order": "motorola" if sig_match.group(4) == "0" else "intel",
                        "is_signed": sig_match.group(5) == "-",
                        "is_float": False,
                        "scale": float(sig_match.group(6).strip()),
                        "offset": float(sig_match.group(7).strip()),
                        "unit": sig_match.group(9),
                        "minimum": None,
                        "maximum": None,
                        "choices": {},
                        "comment": "",
                    })

        logger.info("简化 DBC 解析完成: %d 条报文 (%s)", len(self.frame_names), path)

    def decode(self, frame: dict) -> dict:
        """
        解码单帧 CAN 数据

        Returns: {
            "timestamp_ms": float,
            "frame_id": int,
            "frame_name": str,
            "signals": {信号名: 物理值, ...},
            "raw_data": str,
        }
        """
        frame_id = frame["frame_id"]
        data = frame["data"]
        sig_defs = self.signals.get(frame_id, [])

        decoded = {"timestamp_ms": frame.get("timestamp_ms", 0),
                   "frame_id": frame_id,
                   "frame_name": self.frame_names.get(frame_id, f"0x{frame_id:X}"),
                   "signals": {},
                   "raw_data": frame.get("data_hex", "")}

        for sig in sig_defs:
            try:
                value = self._extract_signal(
                    data, sig["start_bit"], sig["length"],
                    sig["byte_order"], sig["is_signed"], sig["is_float"]
                )
                phys_value = value * sig["scale"] + sig["offset"]
                # 值表映射
                if sig["choices"] and int(value) in sig["choices"]:
                    decoded["signals"][sig["name"]] = {
                        "raw": value,
                        "physical": phys_value,
                        "enum": sig["choices"][int(value)],
                        "unit": sig["unit"],
                    }
                else:
                    decoded["signals"][sig["name"]] = {
                        "raw": value,
                        "physical": round(phys_value, 4),
                        "unit": sig["unit"],
                    }
            except Exception as e:
                logger.warning("信号 %s 解码失败: %s", sig.get("name", "?"), e)

        return decoded

    def _extract_signal(self, data: bytes, start_bit: int, length: int,
                        byte_order: str, is_signed: bool, is_float: bool):
        """从原始字节中提取信号值"""
        if length <= 64 and not is_float:
            return self._extract_integer(data, start_bit, length, byte_order, is_signed)
        elif is_float:
            if length == 32:
                return struct.unpack("<f", data[:4])[0]
            elif length == 64:
                return struct.unpack("<d", data[:8])[0]
        return 0

    def _extract_integer(self, data: bytes, start_bit: int, length: int,
                         byte_order: str, is_signed: bool) -> int:
        """提取整型信号"""
        if byte_order == "intel":
            return self._extract_intel(data, start_bit, length, is_signed)
        else:
            return self._extract_motorola(data, start_bit, length, is_signed)

    def _extract_intel(self, data: bytes, start_bit: int, length: int,
                       is_signed: bool) -> int:
        """Intel 字节序提取"""
        total_bits = len(data) * 8
        value = 0
        for i in range(length):
            bit_pos = start_bit + i
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx >= len(data):
                break
            bit = (data[byte_idx] >> bit_in_byte) & 1
            value |= (bit << i)

        if is_signed and (value >> (length - 1)):
            value -= (1 << length)
        return value

    def _extract_motorola(self, data: bytes, start_bit: int, length: int,
                          is_signed: bool) -> int:
        """Motorola 字节序提取 (big-endian)"""
        total_bits = len(data) * 8
        value = 0
        for i in range(length):
            bit_pos = start_bit - i
            if bit_pos < 0:
                break
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx >= len(data):
                break
            bit = (data[byte_idx] >> bit_in_byte) & 1
            value |= (bit << (length - 1 - i))

        if is_signed and (value >> (length - 1)):
            value -= (1 << length)
        return value

    def lookup_signal(self, frame_id: int, signal_name: str) -> Optional[dict]:
        """查找信号定义"""
        sig_defs = self.signals.get(frame_id, [])
        for sig in sig_defs:
            if sig["name"] == signal_name:
                return sig
        return None
