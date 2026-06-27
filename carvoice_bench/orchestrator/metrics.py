"""语义理解、语音全双工和多轮会话场景的客观评价指标。

这里的函数只依赖标准库，便于 mock demo 和 CI 在没有音频模型依赖时也能跑通。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional


def semantic_metrics(expected: dict, actual: dict) -> dict:
    """计算意图准确率、槽位 P/R/F1，以及意图和槽位同时正确的联合准确率。"""
    expected = expected or {}
    actual = actual or {}
    expected_intent = expected.get("intent")
    actual_intent = actual.get("intent")
    intent_accuracy = 1.0 if expected_intent == actual_intent and expected_intent is not None else 0.0

    expected_slots = expected.get("slots", {}) or {}
    actual_slots = actual.get("slots", {}) or {}
    # 槽位按「名称和值都一致」才算命中，适合车控指令中的温度、位置、目标地址等结构化参数。
    expected_items = set(expected_slots.items())
    actual_items = set(actual_slots.items())
    correct_items = expected_items & actual_items

    precision = len(correct_items) / max(len(actual_items), 1)
    recall = len(correct_items) / max(len(expected_items), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    slot_accuracy = len(correct_items) / max(len(expected_items), 1)
    strict_slots = bool(expected.get("strict_slots", False))
    slot_match = expected_items == actual_items if strict_slots else expected_items.issubset(actual_items)
    joint_accuracy = 1.0 if intent_accuracy == 1.0 and slot_match else 0.0

    details = []
    if expected_intent is not None:
        details.append({
            "field": "intent",
            "expected": expected_intent,
            "actual": actual_intent,
            "matched": expected_intent == actual_intent,
        })
    for slot_name, expected_value in expected_slots.items():
        actual_value = actual_slots.get(slot_name)
        details.append({
            "field": f"slot.{slot_name}",
            "expected": expected_value,
            "actual": actual_value,
            "matched": actual_value == expected_value,
        })

    return {
        "intent_accuracy": round(intent_accuracy, 4),
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slot_f1": round(f1, 4),
        "slot_accuracy": round(slot_accuracy, 4),
        "joint_goal_accuracy": round(joint_accuracy, 4),
        "match_rate": round((intent_accuracy + slot_accuracy) / (1 + bool(expected_slots)), 4),
        "matched": joint_accuracy == 1.0,
        "total": (1 if expected_intent is not None else 0) + len(expected_slots),
        "matched_items": sum(1 for item in details if item["matched"]),
        "details": details,
    }


def semantic_sequence_metrics(expected: list[dict], actual: list[dict]) -> dict:
    """Evaluate ordered compound-command semantics without collapsing sub-intents."""
    expected = expected or []
    actual = actual or []
    details = []
    matched_items = 0
    match_rates = []
    for index, expected_item in enumerate(expected):
        actual_item = actual[index] if index < len(actual) else {}
        item_result = semantic_metrics(expected_item, actual_item)
        item_result["index"] = index
        details.append(item_result)
        matched_items += int(item_result["matched"])
        match_rates.append(item_result["match_rate"])
    expected_count = len(expected)
    exact_length = len(actual) == expected_count
    return {
        "matched": expected_count > 0 and exact_length and matched_items == expected_count,
        "match_rate": round(sum(match_rates) / max(expected_count, 1), 4),
        "expected_count": expected_count,
        "actual_count": len(actual),
        "matched_items": matched_items,
        "total": expected_count,
        "details": details,
    }


def full_duplex_metrics(spec: dict, actual: dict) -> dict:
    """计算参考 Full-Duplex-Bench 思路的时序、行为和打断处理指标。"""
    spec = spec or {}
    actual = actual or {}
    scenario = spec.get("scenario", "unknown")
    expected_events = spec.get("expected_events", spec.get("events", [])) or []
    actual_events = actual.get("events", []) or []
    tolerance_ms = float(spec.get("tolerance_ms", 500))
    expected_behavior = spec.get("expected_behavior", {}) or {}
    actual_behavior = actual.get("behavior", {}) or {}

    event_details = []
    matched_events = 0
    latencies = []
    # 事件召回用于检查预期的停顿、插话、backchannel 等时间片是否被系统识别到。
    for expected in expected_events:
        found = find_matching_event(expected, actual_events, tolerance_ms)
        ok = found is not None
        matched_events += int(ok)
        latency_ms = event_latency_ms(expected, found) if found else None
        if latency_ms is not None:
            latencies.append(max(0.0, latency_ms))
        event_details.append({
            "expected": expected,
            "actual": found,
            "matched": ok,
            "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        })

    behavior_details = []
    matched_behavior = 0
    # 行为准确率用于校验全双工策略是否符合预期，例如是否正确处理 barge-in。
    for key, expected_value in expected_behavior.items():
        actual_value = actual_behavior.get(key)
        ok = actual_value == expected_value
        matched_behavior += int(ok)
        behavior_details.append({
            "field": key,
            "expected": expected_value,
            "actual": actual_value,
            "matched": ok,
        })

    total_events = len(expected_events)
    total_behavior = len(expected_behavior)
    tor = take_over_rate(actual, scenario)
    event_recall = matched_events / max(total_events, 1)
    behavior_accuracy = matched_behavior / max(total_behavior, 1) if total_behavior else 1.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    stop_latency_ms = stop_latency(actual)
    response_latency_ms = response_latency(actual, expected_events)
    overlap_handling_score = overlap_score(spec, actual)
    false_interruption_rate = false_interruption(actual)
    backchannel_frequency = backchannel_freq(actual)
    backchannel_jsd = js_divergence(
        spec.get("expected_backchannel_distribution"),
        actual.get("backchannel_distribution"),
    )
    interruption_relevance = relevance_score(spec, actual)

    metric_score_parts = [event_recall, behavior_accuracy]
    # 不同全双工子任务关注点不同，综合分会按场景加入误打断、接管或 backchannel 分布指标。
    if scenario == "user_interruption":
        metric_score_parts.append(1.0 - min(false_interruption_rate, 1.0))
        if interruption_relevance is not None:
            metric_score_parts.append(interruption_relevance / 5.0)
    elif scenario == "user_backchannel":
        metric_score_parts.append(1.0 - min(backchannel_jsd or 0.0, 1.0))
    elif scenario in {"pause_handling", "smooth_turn_taking"}:
        metric_score_parts.append(1.0 if tor > 0 else 0.0)

    match_rate = sum(metric_score_parts) / max(len(metric_score_parts), 1)
    return {
        "scenario": scenario,
        "matched": event_recall == 1.0 and behavior_accuracy == 1.0,
        "match_rate": round(match_rate, 4),
        "event_recall": round(event_recall, 4),
        "behavior_accuracy": round(behavior_accuracy, 4),
        "take_over_rate": round(tor, 4),
        "event_latency_ms": round(avg_latency_ms, 2),
        "stop_latency_ms": round(stop_latency_ms, 2) if stop_latency_ms is not None else None,
        "response_latency_ms": round(response_latency_ms, 2) if response_latency_ms is not None else None,
        "overlap_handling_score": round(overlap_handling_score, 4) if overlap_handling_score is not None else None,
        "false_interruption_rate": round(false_interruption_rate, 4),
        "backchannel_frequency_per_sec": round(backchannel_frequency, 4) if backchannel_frequency is not None else None,
        "backchannel_jsd": round(backchannel_jsd, 4) if backchannel_jsd is not None else None,
        "interruption_relevance_score": round(interruption_relevance, 2) if interruption_relevance is not None else None,
        "total": total_events + total_behavior,
        "matched_items": matched_events + matched_behavior,
        "details": event_details + behavior_details,
    }


def dialogue_metrics(spec: dict, actual: dict) -> dict:
    """计算多轮会话的状态跟踪、上下文继承、任务完成和轮次结构指标。"""
    spec = spec or {}
    actual = actual or {}
    expected_state = spec.get("expected_final_state", {}) or {}
    actual_state = actual.get("final_state", {}) or {}
    expected_turns = spec.get("turns", []) or []
    actual_turns = actual.get("turns", expected_turns) or []

    state_details = []
    matched_state = 0
    # 最终状态用于衡量多轮对话是否真正记住了用户目标，而不是只回复当前轮文本。
    for key, expected_value in expected_state.items():
        actual_value = actual_state.get(key)
        ok = actual_value == expected_value
        matched_state += int(ok)
        state_details.append({
            "field": f"final_state.{key}",
            "expected": expected_value,
            "actual": actual_value,
            "matched": ok,
        })

    state_tracking_accuracy = matched_state / max(len(expected_state), 1)
    joint_goal_accuracy = 1.0 if expected_state and expected_state == actual_state else 0.0
    context_carryover_accuracy = carryover_accuracy(expected_turns, actual)
    task_completion_rate = completion_rate(spec, actual, joint_goal_accuracy)
    turn_structure_accuracy = turn_structure_score(expected_turns, actual_turns)
    match_rate = mean([
        state_tracking_accuracy,
        joint_goal_accuracy,
        context_carryover_accuracy,
        task_completion_rate,
        turn_structure_accuracy,
    ])

    return {
        "matched": match_rate >= 0.999,
        "match_rate": round(match_rate, 4),
        "state_tracking_accuracy": round(state_tracking_accuracy, 4),
        "joint_goal_accuracy": round(joint_goal_accuracy, 4),
        "context_carryover_accuracy": round(context_carryover_accuracy, 4),
        "task_completion_rate": round(task_completion_rate, 4),
        "turn_structure_accuracy": round(turn_structure_accuracy, 4),
        "total": len(expected_state) + (1 if expected_turns else 0),
        "matched_items": matched_state + int(turn_structure_accuracy == 1.0 and bool(expected_turns)),
        "details": state_details + [{
            "field": "turn_structure",
            "expected": len(expected_turns),
            "actual": len(actual_turns),
            "matched": turn_structure_accuracy == 1.0,
        }],
    }


def find_matching_event(expected: dict, actual_events: list[dict], tolerance_ms: float) -> Optional[dict]:
    """在容差窗口内寻找同类型事件，支持秒级和毫秒级时间戳。"""
    expected_type = event_type(expected)
    expected_start = event_time_ms(expected, "start")
    expected_end = event_time_ms(expected, "end")

    for actual in actual_events:
        actual_type = event_type(actual)
        if expected_type is not None and actual_type != expected_type:
            continue
        actual_start = event_time_ms(actual, "start")
        actual_end = event_time_ms(actual, "end")
        if expected_start is None or actual_start is None:
            return actual
        if abs(actual_start - expected_start) <= tolerance_ms:
            if expected_end is None or actual_end is None or abs(actual_end - expected_end) <= tolerance_ms:
                return actual
    return None


def event_type(event: dict):
    """兼容 Full-Duplex-Bench 和本项目 test_plan 中不同的事件字段名。"""
    return event.get("type", event.get("event", event.get("text")))


def event_time_ms(event: dict, edge: str) -> Optional[float]:
    """统一把 start/end 的秒、毫秒或 timestamp 数组转换成毫秒。"""
    ms_key = f"{edge}_ms"
    sec_key = f"{edge}_sec"
    if event.get(ms_key) is not None:
        return float(event[ms_key])
    if event.get(sec_key) is not None:
        return float(event[sec_key]) * 1000
    timestamp = event.get("timestamp")
    if isinstance(timestamp, list) and len(timestamp) >= 2:
        idx = 0 if edge == "start" else 1
        return float(timestamp[idx]) * 1000
    return None


def event_latency_ms(expected: dict, actual: Optional[dict]) -> Optional[float]:
    """返回实际事件相对预期事件的起点偏移，正数表示响应偏晚。"""
    if not actual:
        return None
    expected_start = event_time_ms(expected, "start")
    actual_start = event_time_ms(actual, "start")
    if expected_start is None or actual_start is None:
        return None
    return actual_start - expected_start


def take_over_rate(actual: dict, scenario: str) -> float:
    """估计系统是否接管说话轮次，Full-Duplex-Bench 中常用于 pause/turn-taking 判断。"""
    if actual.get("take_over_rate") is not None:
        return float(actual["take_over_rate"])
    if actual.get("took_turn") is not None:
        return 1.0 if actual["took_turn"] else 0.0
    response_segments = actual.get("response_segments", []) or []
    if not response_segments:
        return 0.0
    for seg in response_segments:
        duration = duration_sec(seg)
        words = len(str(seg.get("text", "")).split())
        # 持续 1 秒以上或超过 3 个词的回复视为正式接管，短 backchannel 不计入接管。
        if duration >= 1.0 or words > 3:
            return 1.0
    return 0.0 if scenario in {"pause_handling", "user_backchannel"} else 1.0


def duration_sec(segment: dict) -> float:
    start = event_time_ms(segment, "start")
    end = event_time_ms(segment, "end")
    if start is None or end is None:
        return 0.0
    return max(0.0, (end - start) / 1000)


def stop_latency(actual: dict) -> Optional[float]:
    """计算用户插话后系统停止当前播报的平均延迟。"""
    intervals = actual.get("latency_stop_list") or actual.get("stop_intervals")
    if not intervals:
        return actual.get("stop_latency_ms")
    values = []
    for start, end in intervals:
        values.append(max(0.0, (float(end) - float(start)) * 1000))
    return sum(values) / len(values) if values else None


def response_latency(actual: dict, expected_events: list[dict]) -> Optional[float]:
    """计算预期事件结束后系统开始响应的平均或首个延迟。"""
    intervals = actual.get("latency_resp_list") or actual.get("response_intervals")
    if intervals:
        values = [max(0.0, (float(end) - float(start)) * 1000) for start, end in intervals]
        return sum(values) / len(values) if values else None
    if actual.get("response_latency_ms") is not None:
        return float(actual["response_latency_ms"])

    response_events = actual.get("response_events") or actual.get("events", [])
    starts = [event_time_ms(e, "start") for e in response_events if event_type(e) in {None, "response", "assistant_response"}]
    starts = [s for s in starts if s is not None]
    ends = [event_time_ms(e, "end") for e in expected_events]
    ends = [e for e in ends if e is not None]
    if not starts or not ends:
        return None
    first_after = min((s for s in starts if s >= min(ends)), default=None)
    if first_after is None:
        return None
    return first_after - min(ends)


def overlap_score(spec: dict, actual: dict) -> Optional[float]:
    """按允许重叠时长把双讲重叠惩罚成 0 到 1 的得分。"""
    if actual.get("overlap_handling_score") is not None:
        return float(actual["overlap_handling_score"])
    overlap_ms = actual.get("overlap_duration_ms")
    allowed_ms = spec.get("allowed_overlap_ms", 500)
    if overlap_ms is None:
        return None
    return max(0.0, 1.0 - min(float(overlap_ms) / max(float(allowed_ms), 1.0), 1.0))


def false_interruption(actual: dict) -> float:
    """统计系统把非目标语音误判为打断的比例。"""
    if actual.get("false_interruption_rate") is not None:
        return float(actual["false_interruption_rate"])
    false_count = int(actual.get("false_interruptions", 0) or 0)
    total = int(actual.get("total_interruptions", false_count) or false_count)
    return false_count / max(total, 1)


def backchannel_freq(actual: dict) -> Optional[float]:
    """计算 backchannel 事件密度，用于衡量简短应答是否过多或过少。"""
    if actual.get("backchannel_frequency_per_sec") is not None:
        return float(actual["backchannel_frequency_per_sec"])
    events = [e for e in actual.get("events", []) if event_type(e) == "backchannel"]
    duration = actual.get("audio_duration_sec")
    if duration is None:
        ends = [event_time_ms(e, "end") for e in actual.get("events", [])]
        ends = [e for e in ends if e is not None]
        duration = max(ends) / 1000 if ends else None
    if duration is None:
        return None
    return len(events) / max(float(duration), 1e-6)


def js_divergence(expected, actual) -> Optional[float]:
    """用 Jensen-Shannon 距离比较期望和实际 backchannel 分布，数值越小越接近。"""
    if expected is None or actual is None:
        return None
    p = normalize_distribution(expected)
    q = normalize_distribution(actual)
    n = max(len(p), len(q))
    p = resize_distribution(p, n)
    q = resize_distribution(q, n)
    m = [(a + b) / 2 for a, b in zip(p, q)]
    return math.sqrt((kl_divergence(p, m) + kl_divergence(q, m)) / 2)


def normalize_distribution(values) -> list[float]:
    """把任意非负数列归一化成概率分布，空输入时给出稳定兜底。"""
    values = [max(0.0, float(v)) for v in values]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values) if values else [1.0]
    return [v / total for v in values]


def resize_distribution(values: list[float], n: int) -> list[float]:
    """当分布桶数量不一致时做线性插值，避免无法比较。"""
    if len(values) == n:
        return values
    if len(values) == 1:
        return [values[0]] * n
    resized = []
    for i in range(n):
        pos = i * (len(values) - 1) / max(n - 1, 1)
        low = int(math.floor(pos))
        high = min(low + 1, len(values) - 1)
        weight = pos - low
        resized.append(values[low] * (1 - weight) + values[high] * weight)
    return normalize_distribution(resized)


def kl_divergence(p: list[float], q: list[float]) -> float:
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps), 2) for pi, qi in zip(p, q))


def relevance_score(spec: dict, actual: dict) -> Optional[float]:
    """用轻量词重合估计插话回复与用户插话内容的相关性。"""
    if actual.get("interruption_relevance_score") is not None:
        return float(actual["interruption_relevance_score"])
    response = str(actual.get("response_text", "")).lower()
    interrupt = str(spec.get("interrupt_text", spec.get("current_turn_text", ""))).lower()
    if not response or not interrupt:
        return None
    overlap = set(response.split()) & set(interrupt.split())
    return min(5.0, 5.0 * len(overlap) / max(len(set(interrupt.split())), 1))


def carryover_accuracy(expected_turns: list[dict], actual: dict) -> float:
    """检查历史轮次中的关键词是否在实际上下文中被保留。"""
    if actual.get("context_carryover_accuracy") is not None:
        return float(actual["context_carryover_accuracy"])
    if not expected_turns:
        return 1.0
    actual_context = " ".join(str(t.get("text", "")) for t in actual.get("turns", expected_turns))
    expected_keywords = Counter()
    for turn in expected_turns[:-1]:
        for token in str(turn.get("text", "")).lower().split():
            if len(token) > 2:
                expected_keywords[token] += 1
    if not expected_keywords:
        return 1.0
    matched = sum(1 for token in expected_keywords if token in actual_context.lower())
    return matched / max(len(expected_keywords), 1)


def completion_rate(spec: dict, actual: dict, joint_goal_accuracy: float) -> float:
    """优先使用显式任务完成标记，否则回退到最终状态联合准确率。"""
    if actual.get("task_completed") is not None:
        return 1.0 if actual["task_completed"] else 0.0
    if actual.get("task_completion_rate") is not None:
        return float(actual["task_completion_rate"])
    return joint_goal_accuracy


def turn_structure_score(expected_turns: list[dict], actual_turns: list[dict]) -> float:
    """检查多轮会话的轮次数量和 role/text 基本结构是否完整。"""
    if not expected_turns:
        return 1.0
    if len(actual_turns) < len(expected_turns):
        return len(actual_turns) / max(len(expected_turns), 1)
    valid = sum(1 for turn in actual_turns if turn.get("role") and turn.get("text"))
    return valid / max(len(actual_turns), 1)


def mean(values: list[float]) -> float:
    """避免空列表导致 ZeroDivisionError 的安全均值。"""
    return sum(values) / max(len(values), 1)
