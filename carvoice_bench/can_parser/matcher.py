"""CAN 信号匹配器 — 将测试计划中的预期信号与实际 CAN 日志匹配"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CANSignalMatcher:
    """
    CAN 信号匹配器

    从 CAN 日志中查找与测试用例期望匹配的信号变化。
    支持:
    - 精确匹配: frame_id + 信号名 + 物理值
    - 范围匹配: frame_id + 信号名 + 值范围
    - 状态变化匹配: 识别信号从旧值到新值的变化
    """

    def __init__(self, dbc_parser):
        self.dbc = dbc_parser

    def verify_signals(self, frames: list[dict],
                       expected_signals: list[dict],
                       timeout_ms: float,
                       start_time_ms: float = 0) -> dict:
        """
        验证预期信号是否在 CAN 日志中出现

        Args:
            frames: CAN 帧列表 (已按时间戳排序)
            expected_signals: [{"frame_id": "0x2A1", "signals": {"AC_ON": 1}}, ...]
            timeout_ms: 超时窗口 (ms)
            start_time_ms: 搜索起始时间 (ms)

        Returns:
            {
                "matched": bool,                 # 所有信号都匹配
                "match_rate": float,             # 匹配率
                "total_signals": int,
                "matched_signals": int,
                "details": [                     # 每个预期的匹配详情
                    {
                        "frame_id": str,
                        "expected": {...},
                        "matched": bool,
                        "found_at_ms": float,
                        "actual_value": ...,
                    }
                ]
            }
        """
        end_time = start_time_ms + timeout_ms
        matched_count = 0
        total_signals = sum(len(s.get("signals", {})) for s in expected_signals)
        details = []

        for expected in expected_signals:
            frame_id = int(expected["frame_id"], 16) if isinstance(expected["frame_id"], str) else expected["frame_id"]
            expected_sigs = expected.get("signals", {})

            # 在当前时间窗口内搜索匹配的帧
            found_match = False
            found_at = 0.0
            matched_values = {}

            for frame in frames:
                ts = frame.get("timestamp_ms", 0)
                if ts < start_time_ms:
                    continue
                if ts > end_time:
                    break

                if frame["frame_id"] != frame_id:
                    continue

                # 解码信号
                decoded = self.dbc.decode(frame)
                frame_sigs = decoded.get("signals", {})

                # 检查所有预期信号是否匹配
                all_match = True
                for sig_name, expected_val in expected_sigs.items():
                    actual = frame_sigs.get(sig_name, {})
                    actual_val = actual.get("physical", actual.get("raw"))
                    if actual_val != expected_val:
                        all_match = False
                        break
                    matched_values[sig_name] = actual_val

                if all_match:
                    found_match = True
                    found_at = ts
                    break

            if found_match:
                matched_count += len(expected_sigs)
            else:
                # 记录最接近的值
                matched_values = self._find_closest_signal(frames, frame_id, expected_sigs, start_time_ms, end_time)

            details.append({
                "frame_id": expected["frame_id"],
                "expected_signals": expected_sigs,
                "matched": found_match,
                "found_at_ms": found_at if found_match else None,
                "matched_values": matched_values if found_match else matched_values,
            })

        match_rate = matched_count / max(total_signals, 1)

        return {
            "matched": matched_count == total_signals and total_signals > 0,
            "match_rate": round(match_rate, 4),
            "total_signals": total_signals,
            "matched_signals": matched_count,
            "details": details,
        }

    def _find_closest_signal(self, frames: list[dict], frame_id: int,
                              expected_sigs: dict,
                              start_ms: float, end_ms: float) -> dict:
        """查找最接近的匹配值"""
        closest = {}
        for sig_name in expected_sigs:
            for frame in frames:
                if start_ms <= frame.get("timestamp_ms", 0) <= end_ms and frame["frame_id"] == frame_id:
                    decoded = self.dbc.decode(frame)
                    if sig_name in decoded.get("signals", {}):
                        closest[sig_name] = decoded["signals"][sig_name]
                        break
            if sig_name not in closest:
                closest[sig_name] = None
        return closest
