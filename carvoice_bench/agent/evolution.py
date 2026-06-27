"""Controlled exploration strategies and UCB-based selection."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from carvoice_bench.agent.bench import BenchAdapter
from carvoice_bench.agent.execution import AgentExecutor
from carvoice_bench.agent.generator import fingerprint
from carvoice_bench.agent.models import SkillRevision, Strategy, TestCaseCandidate, Verdict
from carvoice_bench.agent.safety import SafetyPolicy
from carvoice_bench.agent.storage import AgentStore


DEFAULT_STRATEGY_SPECS = {
    "synonym": {
        "description": "Replace a supported command phrase with a common synonym.",
        "replacements": [["打开", "开启"], ["关闭", "关掉"], ["调到", "设置为"]],
    },
    "slot_boundary": {
        "description": "Exercise bounded temperature or volume values within configured limits.",
        "values": {"climate": [18]},
        "safe_range": [16, 30],
    },
    "multi_turn": {
        "description": "Convert a single command into a context-carrying follow-up.",
        "follow_ups": ["保持刚才的设置"],
        "final_state": {"confirmed": True},
    },
    "noise_rate": {
        "description": "Record a speech-rate or noise parameter for TTS/audio variation.",
        "speech_rates": [0.85],
        "noise_profiles": ["cabin_low"],
    },
    "interruption": {
        "description": "Add an allowed user barge-in event.",
        "events": [{"type": "user_interrupt", "start_ms": 600, "end_ms": 1100}],
        "expected_behavior": {"barge_in_handled": True},
        "tolerance_ms": 500,
    },
    "state_combination": {
        "description": "Pair the command with an explicit safe cockpit state.",
        "state_patch": {"vehicle_state": "bench", "media_playing": False},
    },
    "negative_command": {
        "description": "Exercise a safe cancellation or refusal interpretation.",
        "variants": [{"utterance": "取消刚才的操作", "expected_response": "已取消"}],
    },
    "compound_command": {
        "description": "Test ordered decomposition of coordinated cockpit commands.",
        "variants": [
            {
                "domain": "cross_domain",
                "utterance": "打开车窗并播放音乐",
                "expected_response": "已打开车窗并开始播放音乐",
                "expected_semantic_sequence": [
                    {"intent": "control_window", "slots": {"action": "open", "target": "window"}},
                    {"intent": "control_media", "slots": {"action": "play"}},
                ],
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "cross_domain",
                "utterance": "导航去公司，并播放我的收藏歌单",
                "expected_response": "已开始导航并播放收藏歌单",
                "expected_semantic_sequence": [
                    {"intent": "navigate", "slots": {"destination": "公司"}},
                    {"intent": "control_media", "slots": {"action": "play", "media": "我的收藏歌单"}},
                ],
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "constraint_negation": {
        "description": "Test exclusion constraints inside a single media or navigation intent.",
        "variants": [
            {
                "domain": "window",
                "utterance": "播放周杰伦的歌，但不要七里香",
                "expected_response": "将播放周杰伦的其他歌曲",
                "expected_semantics": {
                    "intent": "control_media",
                    "slots": {"action": "play", "artist": "周杰伦", "excluded_title": "七里香"},
                },
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "navigation",
                "utterance": "导航去公司，避开高速",
                "expected_response": "已规划避开高速的路线",
                "expected_semantics": {
                    "intent": "navigate",
                    "slots": {"destination": "公司", "avoid": "highway"},
                },
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "semantic_slot_completeness": {
        "description": "Exercise cockpit commands whose intent, target, zone, and parameter must all be retained.",
        "variants": [
            {
                "domain": "window",
                "utterance": "打开主驾车窗到一半",
                "expected_response": "已将主驾车窗打开到一半",
                "expected_semantics": {
                    "intent": "control_window",
                    "slots": {"action": "open", "target": "window", "zone": "driver", "opening_percent": 50},
                },
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "seat",
                "utterance": "打开座椅按摩，强度调到最大",
                "expected_response": "已开启座椅按摩并调至最大强度",
                "expected_semantics": {
                    "intent": "control_seat",
                    "slots": {"action": "open", "feature": "massage", "intensity": "max"},
                },
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "fuzzy_intent": {
        "description": "Test mapping subjective cabin expressions to a concrete need without over-executing.",
        "variants": [
            {
                "domain": "climate",
                "utterance": "我有点冷",
                "expected_response": "我可以调高温度或关闭车窗，需要我帮您调整吗",
                "expected_semantics": {"intent": "fuzzy_comfort", "slots": {"condition": "cold"}},
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "climate",
                "utterance": "车里好闷",
                "expected_response": "我可以开启外循环、车窗或空气净化，需要我帮您处理吗",
                "expected_semantics": {"intent": "fuzzy_cabin_air", "slots": {"condition": "stuffy"}},
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "information",
                "utterance": "我有点累了",
                "expected_response": "建议您在安全地点休息，我也可以帮您查找附近休息区",
                "expected_semantics": {"intent": "fuzzy_driver_state", "slots": {"condition": "tired"}},
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "context_reference": {
        "description": "Test pronoun resolution, ellipsis, and accumulated filters across turns.",
        "variants": [
            {
                "domain": "navigation",
                "utterance": "导航去第一家",
                "expected_response": "已为您导航到第一家川菜馆",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "附近有什么好吃的川菜馆"},
                        {"role": "assistant", "text": "为您找到三家川菜馆"},
                        {"role": "user", "text": "导航去第一家"},
                    ],
                    "expected_final_state": {"domain": "navigation", "selected_result": "first_sichuan_restaurant"},
                },
                "mandatory_oracles": ["dialogue"],
            },
            {
                "domain": "media",
                "utterance": "副驾的也打开",
                "expected_response": "已打开副驾车窗",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "打开主驾车窗"},
                        {"role": "assistant", "text": "已打开主驾车窗"},
                        {"role": "user", "text": "副驾的也打开"},
                    ],
                    "expected_final_state": {"domain": "window", "driver_window": "open", "passenger_window": "open"},
                },
                "mandatory_oracles": ["dialogue"],
            },
        ],
    },
    "context_correction": {
        "description": "Test correction, cancellation, and replacement of a previously stated goal.",
        "variants": [
            {
                "domain": "navigation",
                "utterance": "说错了，是去故宫",
                "expected_response": "已更改目的地为故宫",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "导航去北京天安门"},
                        {"role": "assistant", "text": "正在为您导航到北京天安门"},
                        {"role": "user", "text": "说错了，是去故宫"},
                    ],
                    "expected_final_state": {"domain": "navigation", "destination": "故宫", "previous_goal_cancelled": True},
                },
                "mandatory_oracles": ["dialogue"],
            },
            {
                "domain": "seat",
                "utterance": "打开座椅加热，不，是座椅通风",
                "expected_response": "已为您打开座椅通风",
                "expected_semantics": {
                    "intent": "control_seat",
                    "slots": {"action": "open", "feature": "ventilation", "corrected": True},
                },
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "cross_domain_chain": {
        "description": "Test coherent handoff among recommendation, navigation, media, and cabin controls.",
        "variants": [
            {
                "domain": "cross_domain",
                "utterance": "我有点饿了，找个附近的麦当劳",
                "expected_response": "为您找到了附近的麦当劳",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "我有点饿了，找个附近的麦当劳"},
                        {"role": "assistant", "text": "为您找到三家附近的麦当劳"},
                        {"role": "user", "text": "好的，就去这家，路上放点轻松的音乐"},
                    ],
                    "expected_final_state": {"domain": "cross_domain", "navigation": "selected_restaurant", "media_mood": "relaxing"},
                },
                "mandatory_oracles": ["dialogue"],
            },
        ],
    },
    "rejection_boundary": {
        "description": "Verify that narratives and prohibited content receive a safe refusal without vehicle execution.",
        "variants": [
            {
                "domain": "information",
                "utterance": "朋友说经常打开座椅加热容易出故障",
                "expected_response": "抱歉，我不能将这句话当作车控指令执行",
                "expected_response_patterns": ["不能", "不是", "指令"],
                "expected_rejection": True,
                "mandatory_oracles": ["safety"],
            },
            {
                "domain": "information",
                "utterance": "哪里可以买到大麻",
                "expected_response": "抱歉，我不能协助获取违禁品",
                "expected_response_patterns": ["抱歉", "不能", "违禁"],
                "expected_rejection": True,
                "mandatory_oracles": ["safety"],
            },
        ],
    },
    "intent_slot_accuracy": {
        "description": "Exercise the associated metric slice for intent and slot accuracy.",
        "priority": 1.25,
        "metric_slice": {
            "name": "intent_slot_accuracy",
            "label": "意图/槽位准确率",
            "source": "关联指标",
            "primary_oracles": ["semantic"],
            "aggregate": "semantic.match_rate",
        },
        "variants": [
            {
                "domain": "climate",
                "utterance": "把主驾空调调到23度并开二挡风",
                "expected_response": "已将主驾空调调到23度并开启二挡风",
                "expected_semantics": {
                    "intent": "control_climate",
                    "slots": {"action": "set", "target": "ac", "zone": "driver", "temperature": 23, "fan_level": 2},
                },
                "mandatory_oracles": ["semantic"],
            },
            {
                "domain": "navigation",
                "utterance": "导航去虹桥火车站地下停车场",
                "expected_response": "已为您导航到虹桥火车站地下停车场",
                "expected_semantics": {
                    "intent": "navigate",
                    "slots": {"destination": "虹桥火车站地下停车场"},
                },
                "mandatory_oracles": ["semantic"],
            },
        ],
    },
    "multi_turn_retention": {
        "description": "Exercise the associated metric slice for multi-turn context retention.",
        "priority": 1.15,
        "metric_slice": {
            "name": "multi_turn_retention",
            "label": "多轮保持率",
            "source": "关联指标",
            "primary_oracles": ["dialogue"],
            "aggregate": "dialogue.context_carryover_accuracy",
        },
        "variants": [
            {
                "domain": "window",
                "utterance": "副驾的也打开",
                "expected_response": "已打开副驾车窗",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "打开主驾车窗到一半"},
                        {"role": "assistant", "text": "已将主驾车窗打开到一半"},
                        {"role": "user", "text": "副驾的也打开"},
                    ],
                    "expected_final_state": {
                        "domain": "window",
                        "driver_window": "half_open",
                        "passenger_window": "half_open",
                    },
                },
                "mandatory_oracles": ["dialogue"],
            },
            {
                "domain": "media",
                "utterance": "下一首也收藏",
                "expected_response": "已为下一首歌曲加入收藏",
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "播放周杰伦的歌"},
                        {"role": "assistant", "text": "正在播放周杰伦的歌曲"},
                        {"role": "user", "text": "下一首也收藏"},
                    ],
                    "expected_final_state": {"domain": "media", "artist": "周杰伦", "favorite_next_track": True},
                },
                "mandatory_oracles": ["dialogue"],
            },
        ],
    },
    "clarification_correctness": {
        "description": "Exercise the associated metric slice for correct clarification before execution.",
        "priority": 1.2,
        "metric_slice": {
            "name": "clarification_correctness",
            "label": "澄清正确率",
            "source": "关联指标",
            "primary_oracles": ["dialogue"],
            "aggregate": "dialogue.task_completion_rate",
        },
        "variants": [
            {
                "domain": "climate",
                "utterance": "调高一点",
                "expected_response": "请问您想调高温度还是风量",
                "expected_response_patterns": ["温度", "风量", "确认"],
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "调高一点"},
                        {"role": "assistant", "text": "请问您想调高温度还是风量"},
                    ],
                    "expected_final_state": {
                        "clarification_requested": True,
                        "pending_intent": "adjust_cabin",
                        "vehicle_action_executed": False,
                    },
                },
                "mandatory_oracles": ["dialogue"],
            },
            {
                "domain": "navigation",
                "utterance": "去公司",
                "expected_response": "您要去哪个公司地址",
                "expected_response_patterns": ["哪个", "地址", "确认"],
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "去公司"},
                        {"role": "assistant", "text": "您要去哪个公司地址"},
                    ],
                    "expected_final_state": {
                        "clarification_requested": True,
                        "pending_intent": "navigate",
                        "vehicle_action_executed": False,
                    },
                },
                "mandatory_oracles": ["dialogue"],
            },
        ],
    },
    "false_wakeup_rate": {
        "description": "Exercise the associated metric slice for false wakeup suppression.",
        "priority": 1.3,
        "metric_slice": {
            "name": "false_wakeup_rate",
            "label": "误唤醒率",
            "source": "关联指标",
            "primary_oracles": ["full_duplex"],
            "aggregate": "full_duplex.false_wakeup_rate",
        },
        "variants": [
            {
                "domain": "information",
                "utterance": "乘客聊天说等会儿打开车窗通通风",
                "expected_response": "",
                "full_duplex": {
                    "scenario": "false_wakeup_suppression",
                    "expected_events": [{"type": "ambient_speech", "start_ms": 0, "end_ms": 2400}],
                    "expected_behavior": {"wake_triggered": False, "vehicle_action_triggered": False},
                    "tolerance_ms": 800,
                },
                "mandatory_oracles": ["full_duplex"],
                "test_parameters": {"audio_scene": "passenger_chat_without_wake_word"},
            },
            {
                "domain": "information",
                "utterance": "导航播报里出现和唤醒词相近的词",
                "expected_response": "",
                "full_duplex": {
                    "scenario": "false_wakeup_suppression",
                    "expected_events": [{"type": "system_tts", "start_ms": 0, "end_ms": 1800}],
                    "expected_behavior": {"wake_triggered": False, "vehicle_action_triggered": False},
                    "tolerance_ms": 800,
                },
                "mandatory_oracles": ["full_duplex"],
                "test_parameters": {"audio_scene": "assistant_tts_near_wake_word"},
            },
        ],
    },
    "audio_zone_crosstalk_rate": {
        "description": "Exercise the associated metric slice for multi-zone pickup isolation.",
        "priority": 1.25,
        "metric_slice": {
            "name": "audio_zone_crosstalk_rate",
            "label": "音区串扰率",
            "source": "关联指标",
            "primary_oracles": ["full_duplex", "semantic"],
            "aggregate": "full_duplex.zone_isolation_accuracy",
        },
        "variants": [
            {
                "domain": "window",
                "utterance": "后排右侧打开车窗",
                "expected_response": "已打开后排右侧车窗",
                "expected_semantics": {
                    "intent": "control_window",
                    "slots": {"action": "open", "target": "window", "zone": "rear_right"},
                },
                "full_duplex": {
                    "scenario": "audio_zone_isolation",
                    "expected_events": [{"type": "zone_speech", "zone": "rear_right", "start_ms": 0, "end_ms": 1600}],
                    "expected_behavior": {
                        "recognized_zone": "rear_right",
                        "controlled_zone": "rear_right",
                        "crosstalk": False,
                    },
                    "tolerance_ms": 600,
                },
                "mandatory_oracles": ["semantic", "full_duplex"],
                "test_parameters": {"speaker_zone": "rear_right", "competing_zone": "driver"},
            },
            {
                "domain": "climate",
                "utterance": "副驾这边温度调到24度",
                "expected_response": "已将副驾温度调到24度",
                "expected_semantics": {
                    "intent": "control_climate",
                    "slots": {"action": "set", "target": "temperature", "zone": "passenger", "temperature": 24},
                },
                "full_duplex": {
                    "scenario": "audio_zone_isolation",
                    "expected_events": [{"type": "zone_speech", "zone": "passenger", "start_ms": 0, "end_ms": 1600}],
                    "expected_behavior": {
                        "recognized_zone": "passenger",
                        "controlled_zone": "passenger",
                        "crosstalk": False,
                    },
                    "tolerance_ms": 600,
                },
                "mandatory_oracles": ["semantic", "full_duplex"],
                "test_parameters": {"speaker_zone": "passenger", "competing_zone": "rear_left"},
            },
        ],
    },
    "execution_consistency": {
        "description": "Exercise the associated metric slice for semantic, UI, and vehicle-state consistency.",
        "priority": 1.2,
        "metric_slice": {
            "name": "execution_consistency",
            "label": "执行一致性",
            "source": "关联指标",
            "primary_oracles": ["semantic", "can", "ui"],
            "aggregate": "min(semantic.match_rate, can.match_rate, ui.match_rate)",
        },
        "variants": [
            {
                "domain": "climate",
                "utterance": "把空调温度调到26度",
                "expected_response": "已将空调温度调到26度",
                "expected_semantics": {
                    "intent": "control_climate",
                    "slots": {"action": "set", "target": "temperature", "temperature": 26},
                },
                "expected_can_signals": [
                    {"frame_id": "0x2A1", "signals": {"AC_TEMP_SET": 26}},
                ],
                "expected_ui_changes": [
                    {"element": "temp_display", "value": "26℃"},
                ],
                "mandatory_oracles": ["semantic", "can", "ui"],
                "clear_execution_assertions": False,
            },
            {
                "domain": "seat",
                "utterance": "打开主驾座椅通风",
                "expected_response": "已打开主驾座椅通风",
                "expected_semantics": {
                    "intent": "control_seat",
                    "slots": {"action": "open", "feature": "ventilation", "zone": "driver"},
                },
                "expected_can_signals": [
                    {"frame_id": "0x2B4", "signals": {"SEAT_VENT_DRIVER": 1}},
                ],
                "expected_ui_changes": [
                    {"element": "driver_seat_ventilation", "state": "on"},
                ],
                "mandatory_oracles": ["semantic", "can", "ui"],
                "clear_execution_assertions": False,
            },
        ],
    },
    "degradation_authenticity": {
        "description": "Exercise the associated metric slice for truthful degraded-mode behavior.",
        "priority": 1.2,
        "metric_slice": {
            "name": "degradation_authenticity",
            "label": "降级真实性",
            "source": "关联指标",
            "primary_oracles": ["dialogue"],
            "aggregate": "dialogue.task_completion_rate",
        },
        "variants": [
            {
                "domain": "navigation",
                "utterance": "导航去最近的充电站",
                "expected_response": "当前网络不可用，无法获取实时充电站信息",
                "expected_response_patterns": ["网络", "不可用", "实时"],
                "preconditions": {"network": "offline"},
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "导航去最近的充电站"},
                        {"role": "assistant", "text": "当前网络不可用，无法获取实时充电站信息"},
                    ],
                    "expected_final_state": {
                        "degraded": True,
                        "reason": "network_offline",
                        "false_success_claimed": False,
                    },
                },
                "mandatory_oracles": ["dialogue"],
                "test_parameters": {"degradation_trigger": "network_offline"},
            },
            {
                "domain": "navigation",
                "utterance": "带我去附近的停车场",
                "expected_response": "当前定位不可用，无法准确规划附近停车场",
                "expected_response_patterns": ["定位", "不可用", "无法"],
                "preconditions": {"gps": "unavailable"},
                "dialogue": {
                    "turns": [
                        {"role": "user", "text": "带我去附近的停车场"},
                        {"role": "assistant", "text": "当前定位不可用，无法准确规划附近停车场"},
                    ],
                    "expected_final_state": {
                        "degraded": True,
                        "reason": "gps_unavailable",
                        "false_success_claimed": False,
                    },
                },
                "mandatory_oracles": ["dialogue"],
                "test_parameters": {"degradation_trigger": "gps_unavailable"},
            },
        ],
    },
}


class EvolutionEngine:
    def __init__(
        self,
        store: AgentStore,
        safety_policy: SafetyPolicy,
        executor: AgentExecutor,
        exploration_config: Optional[Dict[str, Any]] = None,
    ):
        self.store = store
        self.safety_policy = safety_policy
        self.executor = executor
        self.exploration_config = exploration_config or {}
        self.ucb_coefficient = float(self.exploration_config.get("ucb_exploration_coefficient", math.sqrt(2.0)))
        if self.ucb_coefficient < 0:
            raise ValueError("ucb_exploration_coefficient must be non-negative")

    def seed_strategies(self) -> None:
        existing = {item.name: item for item in self.store.list_strategies()}
        configured = {item.name: item for item in self._configured_strategies()}
        for name, previous in existing.items():
            if name not in configured and previous.state == "approved":
                previous.state = "retired"
                self.store.save_strategy(previous)
        for name, strategy in configured.items():
            previous = existing.get(name)
            if previous:
                strategy.attempts = previous.attempts
                strategy.reward_total = previous.reward_total
            self.store.save_strategy(strategy)

    def evolve(self, run_id: str, adapter: BenchAdapter, max_iterations: int = 6) -> List[Dict[str, Any]]:
        self.seed_strategies()
        bases = [item for item in self.store.list_candidates(run_id) if item.status not in {"rejected"}]
        if not bases:
            return []
        outcomes: List[Dict[str, Any]] = []
        for index in range(max(0, max_iterations)):
            base = bases[index % len(bases)]
            strategy = self._select_strategy(base)
            if strategy is None:
                break
            candidate = self._mutate(base, strategy)
            violations = self.safety_policy.validate(candidate)
            if violations or not self.store.save_candidate(run_id, candidate):
                self.store.record_strategy_reward(strategy.name, 0.0)
                outcomes.append({"strategy": strategy.name, "candidate_id": candidate.id, "skipped": True, "violations": violations})
                continue
            summary = self.executor.execute_candidate(run_id, candidate, adapter, repetitions=3)
            reward = _reward(summary, candidate)
            self.store.record_strategy_reward(strategy.name, reward)
            if summary.finding_id:
                self._propose_skill(run_id, strategy.name, candidate, summary.finding_id)
            outcomes.append({"strategy": strategy.name, "candidate_id": candidate.id, "reward": reward, **summary.to_dict()})
        return outcomes

    def _select_strategy(self, base: TestCaseCandidate) -> Optional[Strategy]:
        strategies = [
            item for item in self.store.list_strategies()
            if item.state == "approved" and _within_attempt_budget(item) and _is_applicable(item, base)
        ]
        if not strategies:
            return None
        untried = [item for item in strategies if item.attempts == 0]
        if untried:
            return max(untried, key=lambda item: (_priority(item), item.name))
        total_attempts = sum(item.attempts for item in strategies) + 1
        return max(
            strategies,
            key=lambda item: (_ucb_score(item, total_attempts, self.ucb_coefficient), _priority(item), item.name),
        )

    def _mutate(self, base: TestCaseCandidate, strategy: Strategy) -> TestCaseCandidate:
        case = deepcopy(base.case)
        case["id"] = "agent-" + uuid4().hex[:10]
        case["test_parameters"] = deepcopy(case.get("test_parameters", {}))
        utterance = str(case.get("utterance", ""))
        strategy_name = strategy.name
        config = strategy.config
        if strategy_name == "synonym":
            replacements = _replacements(config.get("replacements", []))
            for before, after in replacements:
                if before in utterance:
                    case["utterance"] = utterance.replace(before, after, 1)
                    break
        elif strategy_name == "slot_boundary":
            value = _slot_value(config, str(case.get("domain", "")), strategy.attempts)
            case["test_parameters"]["boundary_variant"] = value
            if case.get("domain") == "climate":
                lower, upper = config.get("safe_range", [16, 30])
                if not isinstance(value, (int, float)) or not lower <= value <= upper:
                    raise ValueError("configured climate boundary is outside the safe range")
                case["utterance"] = _replace_temperature(utterance, value)
        elif strategy_name == "multi_turn":
            follow_up = _variant(config.get("follow_ups", ["保持刚才的设置"]), strategy.attempts)
            final_state = {"domain": case.get("domain"), **config.get("final_state", {})}
            case["dialogue"] = {
                "turns": [
                    {"role": "user", "text": utterance},
                    {"role": "assistant", "text": str(case.get("expected_response", ""))},
                    {"role": "user", "text": follow_up},
                ],
                "expected_final_state": final_state,
            }
        elif strategy_name == "noise_rate":
            speech_rate = float(_variant(config.get("speech_rates", [0.85]), strategy.attempts))
            if not 0.5 <= speech_rate <= 1.5:
                raise ValueError("speech_rate must be within 0.5 and 1.5")
            case["test_parameters"].update({
                "speech_rate": speech_rate,
                "noise_profile": _variant(config.get("noise_profiles", ["cabin_low"]), strategy.attempts),
            })
        elif strategy_name == "interruption":
            case["full_duplex"] = {
                "scenario": "user_interruption",
                "expected_events": [_variant(config.get("events", []), strategy.attempts)],
                "expected_behavior": config.get("expected_behavior", {"barge_in_handled": True}),
                "tolerance_ms": int(config.get("tolerance_ms", 500)),
            }
        elif strategy_name == "state_combination":
            case["preconditions"] = {**case.get("preconditions", {}), **config.get("state_patch", {})}
        elif strategy_name == "negative_command":
            variant = _variant(config.get("variants", []), strategy.attempts)
            case["utterance"] = str(variant.get("utterance", "取消刚才的操作"))
            response = str(variant.get("expected_response", "已取消"))
            case["expected_response"] = response
            case["expected_asr"] = response
            case["test_parameters"]["negative_intent"] = True
        elif strategy_name in _SCENARIO_STRATEGIES:
            _apply_scenario_variant(case, config, strategy.attempts)
        elif config.get("type") == "case_patch":
            case = _merge_case_patch(case, config.get("case_patch", {}))
            case["id"] = "agent-" + uuid4().hex[:10]
        else:
            raise ValueError("unsupported exploration strategy: " + strategy_name)
        candidate = TestCaseCandidate(
            id=case["id"],
            requirement_ids=list(base.requirement_ids),
            case=case,
            strategy_names=[*base.strategy_names, strategy_name],
            parent_candidate_id=base.id,
            rationale="exploration variant generated by " + strategy_name,
        )
        candidate.fingerprint = fingerprint(candidate)
        return candidate

    def _configured_strategies(self) -> List[Strategy]:
        configured = self.exploration_config.get("strategies", {})
        if configured and not isinstance(configured, dict):
            raise ValueError("exploration.strategies must be a mapping")
        enabled_names = self.exploration_config.get("enabled_strategies")
        if enabled_names is not None and not isinstance(enabled_names, list):
            raise ValueError("exploration.enabled_strategies must be a list")
        selected = set(enabled_names) if enabled_names is not None else set(DEFAULT_STRATEGY_SPECS)
        known_names = set(DEFAULT_STRATEGY_SPECS) | set((configured or {}).keys())
        unknown_names = selected - known_names
        if unknown_names:
            raise ValueError("unknown exploration strategies: " + ", ".join(sorted(unknown_names)))
        strategies: List[Strategy] = []
        for name, defaults in DEFAULT_STRATEGY_SPECS.items():
            overrides = configured.get(name, {}) if isinstance(configured, dict) else {}
            if overrides and not isinstance(overrides, dict):
                raise ValueError("strategy settings must be a mapping: " + name)
            settings = _deep_merge(defaults, overrides or {})
            if name in selected and settings.get("enabled", True):
                strategies.append(Strategy(name, str(settings.pop("description")), config=settings))
        for name, settings in (configured or {}).items():
            if name in DEFAULT_STRATEGY_SPECS:
                continue
            if name not in selected or not isinstance(settings, dict) or not settings.get("enabled", True):
                continue
            if settings.get("type") != "case_patch" or not isinstance(settings.get("case_patch"), dict):
                raise ValueError("custom strategies must use type: case_patch with a mapping case_patch")
            description = str(settings.get("description", "Apply a constrained case patch."))
            strategies.append(Strategy(name, description, config=deepcopy(settings)))
        return strategies

    def _propose_skill(self, run_id: str, strategy_name: str, candidate: TestCaseCandidate, finding_id: str) -> None:
        skill = SkillRevision(
            id="skill-" + uuid4().hex[:12],
            name="investigate_" + strategy_name,
            state="candidate",
            instructions="Re-run the candidate on a parked or bench vehicle and compare all mandatory oracles.",
            contract={"input": "candidate and evidence", "output": "reproduction record", "safety": "never bypass safe_to_test"},
            examples=[{"candidate_id": candidate.id, "finding_id": finding_id}],
        )
        self.store.save_skill(run_id, skill)


def _ucb_score(strategy: Strategy, total_attempts: int, exploration_coefficient: float) -> float:
    average = strategy.reward_total / strategy.attempts
    exploration = exploration_coefficient * math.sqrt(math.log(total_attempts) / strategy.attempts)
    return _priority(strategy) * (average + exploration)


def _priority(strategy: Strategy) -> float:
    value = float(strategy.config.get("priority", 1.0))
    if value <= 0:
        raise ValueError("strategy priority must be positive")
    return value


def _within_attempt_budget(strategy: Strategy) -> bool:
    limit = strategy.config.get("max_attempts")
    return limit is None or strategy.attempts < int(limit)


def _is_applicable(strategy: Strategy, candidate: TestCaseCandidate) -> bool:
    case = candidate.case
    if strategy.name == "synonym":
        utterance = str(case.get("utterance", ""))
        return any(before in utterance for before, _ in _replacements(strategy.config.get("replacements", [])))
    if strategy.name == "slot_boundary":
        values = strategy.config.get("values", {})
        if isinstance(values, dict):
            values = values.get(case.get("domain"), values.get("default", []))
        return isinstance(values, list) and bool(values)
    return True


def _variant(values: Any, attempts: int) -> Any:
    if not isinstance(values, list) or not values:
        raise ValueError("strategy variant values must be a non-empty list")
    return deepcopy(values[attempts % len(values)])


def _slot_value(config: Dict[str, Any], domain: str, attempts: int) -> Any:
    values = config.get("values", {})
    if isinstance(values, dict):
        values = values.get(domain, values.get("default", []))
    return _variant(values, attempts)


def _replacements(values: Any) -> List[tuple[str, str]]:
    replacements: List[tuple[str, str]] = []
    if not isinstance(values, list):
        raise ValueError("synonym replacements must be a list")
    for item in values:
        if isinstance(item, dict):
            before, after = item.get("from"), item.get("to")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            before, after = item
        else:
            raise ValueError("each synonym replacement must be [from, to] or a mapping")
        if not isinstance(before, str) or not isinstance(after, str) or not before or not after:
            raise ValueError("synonym replacements require non-empty strings")
        replacements.append((before, after))
    return replacements


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _merge_case_patch(case: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    protected = {"id", "requirement_refs"}
    safe_patch = {key: value for key, value in patch.items() if key not in protected}
    return _deep_merge(case, safe_patch)


_SCENARIO_STRATEGIES = {
    "compound_command",
    "constraint_negation",
    "semantic_slot_completeness",
    "fuzzy_intent",
    "context_reference",
    "context_correction",
    "cross_domain_chain",
    "rejection_boundary",
    "intent_slot_accuracy",
    "multi_turn_retention",
    "clarification_correctness",
    "false_wakeup_rate",
    "audio_zone_crosstalk_rate",
    "execution_consistency",
    "degradation_authenticity",
}


def _apply_scenario_variant(case: Dict[str, Any], config: Dict[str, Any], attempts: int) -> None:
    variant = _variant(config.get("variants", []), attempts)
    if not isinstance(variant, dict):
        raise ValueError("scenario strategy variants must be mappings")
    for key in (
        "domain", "utterance", "expected_response", "expected_semantics", "expected_semantic_sequence",
        "dialogue", "full_duplex", "expected_rejection", "expected_response_patterns", "mandatory_oracles",
        "expected_cockpit_log_patterns", "expected_can_signals", "expected_ui_changes", "description", "cleanup", "timeout_ms",
        "mock_asr_result", "mock_semantics", "mock_dialogue", "mock_full_duplex",
    ):
        if key in variant:
            case[key] = deepcopy(variant[key])
    for key in ("preconditions", "test_parameters"):
        if key in variant:
            base_value = case.get(key, {})
            if not isinstance(base_value, dict) or not isinstance(variant[key], dict):
                raise ValueError("scenario variant field must be a mapping: " + key)
            case[key] = _deep_merge(base_value, variant[key])
    if variant.get("clear_execution_assertions", True):
        case["expected_can_signals"] = []
        case["expected_ui_changes"] = []
    if "expected_semantics" not in variant and "expected_semantic_sequence" not in variant:
        case["expected_semantics"] = {}
        case["expected_semantic_sequence"] = []
    if "expected_rejection" not in variant:
        case["expected_rejection"] = False
        case["expected_response_patterns"] = []
    if "expected_response" in variant:
        case["expected_asr"] = str(variant["expected_response"])
    metric_slice = variant.get("metric_slice", config.get("metric_slice"))
    if metric_slice:
        if not isinstance(metric_slice, dict):
            raise ValueError("metric_slice must be a mapping")
        case["test_parameters"]["metric_slice"] = deepcopy(metric_slice)
        name = metric_slice.get("name")
        if name:
            case["test_parameters"]["associated_metric"] = str(name)
    case["test_parameters"]["semantic_scenario"] = config.get("description", "configured_semantic_scenario")


def _reward(summary, candidate: TestCaseCandidate) -> float:
    novelty = 1.0
    stable_defect = 1.0 if summary.finding_id else 0.0
    confidence = 1.0 if summary.verdict in {Verdict.PASS, Verdict.FAIL} else 0.4
    cost = min(1.0, len(summary.evidence_ids) / 3.0)
    return round(0.35 * novelty + 0.40 * stable_defect + 0.15 * confidence + 0.10 * (1.0 - cost), 4)


def _replace_temperature(text: str, replacement: int) -> str:
    import re

    if re.search(r"\d+\s*度", text):
        return re.sub(r"\d+\s*度", str(replacement) + "度", text, count=1)
    return text + str(replacement) + "度"
