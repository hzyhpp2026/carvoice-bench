"""Rule and dictionary based semantic parser for common car voice commands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class RuleSemanticParser:
    """Extract intent and slots from ASR text with small deterministic rules."""

    def __init__(self, rules_path: str | None = None):
        self.custom_rules = _load_custom_rules(rules_path)

    def parse(self, text: str, case: dict | None = None) -> dict:
        text = normalize_text(text)
        if not text:
            return self._result("unknown", {}, text, 0.0, "empty")

        custom = self._parse_custom_rules(text)
        if custom:
            return custom

        parsers = [
            self._parse_climate,
            self._parse_window,
            self._parse_navigation,
            self._parse_media,
            self._parse_volume,
            self._parse_seat,
            self._parse_fuzzy,
        ]
        for parser in parsers:
            result = parser(text)
            if result:
                return result
        return self._result("unknown", {}, text, 0.2, "rule")

    def parse_many(self, text: str, case: dict | None = None) -> list[dict]:
        """Parse coordinated commands into ordered sub-intents when possible."""
        normalized = normalize_text(text)
        clauses = [item for item in re.split(r"(?:，|,)?(?:并且|然后|顺便|再|并)", normalized) if item]
        if len(clauses) <= 1:
            return [self.parse(normalized, case)]
        return [self.parse(clause, case) for clause in clauses]

    def _parse_fuzzy(self, text: str) -> dict | None:
        fuzzy_cases = [
            ("我有点冷", "fuzzy_comfort", {"condition": "cold"}),
            ("车里好闷", "fuzzy_cabin_air", {"condition": "stuffy"}),
            ("我有点累", "fuzzy_driver_state", {"condition": "tired"}),
            ("我饿", "fuzzy_lifestyle", {"need": "dining"}),
            ("有点无聊", "fuzzy_lifestyle", {"need": "entertainment"}),
        ]
        for phrase, intent, slots in fuzzy_cases:
            if phrase in text:
                return self._result(intent, slots, text, 0.65, "rule_fuzzy")
        return None

    def _parse_custom_rules(self, text: str) -> dict | None:
        for rule in self.custom_rules:
            patterns = rule.get("patterns", [])
            if not any(re.search(pattern, text) for pattern in patterns):
                continue
            slots = dict(rule.get("slots", {}))
            for slot_name, pattern in rule.get("slot_patterns", {}).items():
                match = re.search(pattern, text)
                if match:
                    slots[slot_name] = _coerce_value(match.group(1))
            return self._result(rule.get("intent", "unknown"), slots, text, 0.9, "custom_rule")
        return None

    def _parse_climate(self, text: str) -> dict | None:
        if not any(word in text for word in ["空调", "温度", "暖风", "制冷", "制热"]):
            return None
        slots: dict[str, Any] = {}
        action = extract_action(text)
        if action:
            slots["action"] = action
        zone = extract_zone(text)
        if zone:
            slots["zone"] = zone
        temperature = extract_temperature(text)
        if temperature is not None:
            slots["temperature"] = temperature
        fan_speed = extract_fan_speed(text)
        if fan_speed is not None:
            slots["fan_speed"] = fan_speed
        intent = "set_climate" if temperature is not None or fan_speed is not None else "control_climate"
        return self._result(intent, slots, text, 0.82, "rule")

    def _parse_window(self, text: str) -> dict | None:
        if not any(word in text for word in ["车窗", "窗户", "天窗"]):
            return None
        slots = {}
        action = extract_action(text)
        if action:
            slots["action"] = action
        zone = extract_zone(text)
        if zone:
            slots["zone"] = zone
        target = "sunroof" if "天窗" in text else "window"
        slots["target"] = target
        return self._result("control_window", slots, text, 0.82, "rule")

    def _parse_navigation(self, text: str) -> dict | None:
        if not any(word in text for word in ["导航", "去", "到"]):
            return None
        destination = extract_after_keywords(text, ["导航到", "导航去", "去", "到"])
        if destination in {"", "最低", "最高"}:
            destination = None
        slots = {"destination": destination} if destination else {}
        return self._result("navigate", slots, text, 0.78, "rule")

    def _parse_media(self, text: str) -> dict | None:
        if not any(word in text for word in ["播放", "暂停", "继续播放", "音乐", "歌", "电台", "收音机"]):
            return None
        slots = {}
        action = extract_action(text) or ("play" if "播放" in text else None)
        if action:
            slots["action"] = action
        media = extract_after_keywords(text, ["播放", "听"])
        if media:
            slots["media"] = media
        return self._result("control_media", slots, text, 0.78, "rule")

    def _parse_volume(self, text: str) -> dict | None:
        if not any(word in text for word in ["音量", "声音", "静音"]):
            return None
        slots = {}
        action = extract_action(text)
        if "静音" in text:
            action = "mute"
        if action:
            slots["action"] = action
        level = extract_number(text)
        if level is not None:
            slots["level"] = level
        return self._result("control_volume", slots, text, 0.78, "rule")

    def _parse_seat(self, text: str) -> dict | None:
        if not any(word in text for word in ["座椅", "座位", "靠背"]):
            return None
        slots = {}
        action = extract_action(text)
        if action:
            slots["action"] = action
        zone = extract_zone(text)
        if zone:
            slots["zone"] = zone
        return self._result("control_seat", slots, text, 0.74, "rule")

    @staticmethod
    def _result(intent: str, slots: dict, text: str, confidence: float, method: str) -> dict:
        return {
            "intent": intent,
            "slots": slots,
            "confidence": round(confidence, 2),
            "method": method,
            "source_text": text,
        }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()


def extract_action(text: str) -> str | None:
    if any(word in text for word in ["关闭", "关掉", "关上", "停止"]):
        return "close"
    if any(word in text for word in ["打开", "开启", "开一下"]):
        return "open"
    if any(word in text for word in ["调高", "升高", "大一点", "增加"]):
        return "increase"
    if any(word in text for word in ["调低", "降低", "小一点", "减少"]):
        return "decrease"
    if any(word in text for word in ["暂停"]):
        return "pause"
    if any(word in text for word in ["继续", "播放"]):
        return "play"
    return None


def extract_zone(text: str) -> str | None:
    zone_map = [
        ("all", ["所有", "全部", "全车"]),
        ("driver", ["主驾", "驾驶位", "司机"]),
        ("passenger", ["副驾", "副驾驶"]),
        ("rear", ["后排"]),
        ("left", ["左侧", "左边"]),
        ("right", ["右侧", "右边"]),
    ]
    for zone, keywords in zone_map:
        if any(keyword in text for keyword in keywords):
            return zone
    return None


def extract_temperature(text: str) -> int | None:
    if "最低" in text:
        return 18
    if "最高" in text:
        return 32
    match = re.search(r"(\d{2})(?:度|℃)?", text)
    if match:
        value = int(match.group(1))
        if 16 <= value <= 32:
            return value
    cn_value = chinese_number_to_int(text)
    if cn_value is not None and 16 <= cn_value <= 32:
        return cn_value
    return None


def extract_fan_speed(text: str) -> int | None:
    match = re.search(r"([1-9一二三四五六七八九])档", text)
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else CN_DIGITS.get(token)


def extract_number(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    return chinese_number_to_int(text)


def extract_after_keywords(text: str, keywords: list[str]) -> str | None:
    for keyword in keywords:
        idx = text.find(keyword)
        if idx >= 0:
            value = text[idx + len(keyword):].strip("，。,. ")
            return value or None
    return None


CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def chinese_number_to_int(text: str) -> int | None:
    match = re.search(r"([一二两三四五六七八九]?十[一二三四五六七八九]?|[一二两三四五六七八九])", text)
    if not match:
        return None
    token = match.group(1)
    if "十" not in token:
        return CN_DIGITS.get(token)
    left, _, right = token.partition("十")
    tens = CN_DIGITS.get(left, 1) if left else 1
    ones = CN_DIGITS.get(right, 0) if right else 0
    return tens * 10 + ones


def _coerce_value(value: str):
    if value.isdigit():
        return int(value)
    return value


def _load_custom_rules(path: str | None) -> list[dict]:
    if not path:
        return []
    rule_path = Path(path)
    if not rule_path.exists():
        return []
    if rule_path.suffix.lower() == ".json":
        data = json.loads(rule_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml

            data = yaml.safe_load(rule_path.read_text(encoding="utf-8"))
        except ImportError:
            data = []
    if isinstance(data, dict):
        data = data.get("rules", [])
    return data if isinstance(data, list) else []
