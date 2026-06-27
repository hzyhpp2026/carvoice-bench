"""Requirement-to-test-case generation with a deterministic offline fallback."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from carvoice_bench.agent.llm import LLMClient
from carvoice_bench.agent.models import Requirement, TestCaseCandidate
from carvoice_bench.agent.safety import SafetyPolicy


class CaseGenerator:
    def __init__(self, rules: Dict[str, Any], safety_policy: SafetyPolicy, llm_client: Optional[LLMClient] = None):
        self.rules = rules
        self.safety_policy = safety_policy
        self.llm_client = llm_client

    def generate(self, requirements: Iterable[Requirement]) -> List[TestCaseCandidate]:
        generated: List[TestCaseCandidate] = []
        for requirement in requirements:
            records = self._from_llm(requirement) if self.llm_client else self._heuristic(requirement)
            for record in records:
                candidate = self._candidate_from_record(requirement, record)
                if not self.safety_policy.validate(candidate):
                    generated.append(candidate)
        return generated

    def _from_llm(self, requirement: Requirement) -> List[Dict[str, Any]]:
        prompt = {
            "task": "Create safe, structured test cases for a stationary vehicle voice assistant.",
            "requirement": requirement.to_dict(),
            "rules": self.rules,
            "allowed_domains": sorted(self.safety_policy.allowed_domains),
            "forbidden_terms": sorted(self.safety_policy.forbidden_terms),
            "schema": {
                "candidates": [{
                    "description": "string", "utterance": "string", "domain": "string",
                    "expected_response": "string", "expected_semantics": {"intent": "string", "slots": {}},
                    "expected_semantic_sequence": [{"intent": "string", "slots": {}}],
                    "expected_cockpit_log_patterns": ["string"],
                    "expected_can_signals": [], "expected_ui_changes": [],
                    "mandatory_oracles": ["voice"], "preconditions": {"vehicle_state": "bench"},
                    "dialogue": {"turns": [], "expected_final_state": {}},
                    "expected_rejection": False, "expected_response_patterns": [],
                    "cleanup": "string", "timeout_ms": 5000, "rationale": "string",
                }],
            },
        }
        response = self.llm_client.complete_json(json.dumps(prompt, ensure_ascii=False))
        records = response.get("candidates", [])
        if not isinstance(records, list):
            raise RuntimeError("LLM candidates must be a list")
        return [record for record in records if isinstance(record, dict)]

    def _heuristic(self, requirement: Requirement) -> List[Dict[str, Any]]:
        text = " ".join(requirement.text.split())
        defaults = self.rules.get("defaults", {}) if isinstance(self.rules, dict) else {}
        domain = str(defaults.get("domain") or _infer_domain(text))
        utterance = str(defaults.get("utterance") or _extract_utterance(text))
        mandatory = list(defaults.get("mandatory_oracles", ["voice"]))
        if domain in {"climate", "window", "seat", "media"} and "can" not in mandatory:
            mandatory.append("can")
        return [{
            "description": requirement.title or utterance,
            "utterance": utterance,
            "domain": domain,
            "expected_response": str(defaults.get("expected_response", utterance)),
            "expected_semantics": defaults.get("expected_semantics", {}),
            "expected_cockpit_log_patterns": deepcopy(defaults.get("expected_cockpit_log_patterns", [])),
            "expected_can_signals": deepcopy(defaults.get("expected_can_signals", [])),
            "expected_ui_changes": deepcopy(defaults.get("expected_ui_changes", [])),
            "mandatory_oracles": mandatory,
            "preconditions": deepcopy(defaults.get("preconditions", {"vehicle_state": "bench"})),
            "cleanup": str(defaults.get("cleanup", "restore the prior cockpit state")),
            "timeout_ms": int(defaults.get("timeout_ms", 5000)),
            "rationale": "offline heuristic derived from requirement text",
        }]

    def _candidate_from_record(self, requirement: Requirement, record: Dict[str, Any]) -> TestCaseCandidate:
        case_id = "agent-" + uuid4().hex[:10]
        expected_response = str(record.get("expected_response") or record.get("utterance") or "")
        case = {
            "id": case_id,
            "description": str(record.get("description") or requirement.title),
            "utterance": str(record.get("utterance") or ""),
            # Existing orchestrators use expected_asr as their text oracle.
            "expected_asr": expected_response,
            "expected_response": expected_response,
            "domain": str(record.get("domain") or ""),
            "expected_semantics": record.get("expected_semantics") or {},
            "expected_semantic_sequence": record.get("expected_semantic_sequence") or [],
            "expected_cockpit_log_patterns": record.get("expected_cockpit_log_patterns") or [],
            "expected_rejection": bool(record.get("expected_rejection", False)),
            "expected_response_patterns": record.get("expected_response_patterns") or [],
            "dialogue": record.get("dialogue") or {},
            "full_duplex": record.get("full_duplex") or {},
            "test_parameters": record.get("test_parameters") or {},
            "expected_can_signals": record.get("expected_can_signals") or [],
            "expected_ui_changes": record.get("expected_ui_changes") or [],
            "mandatory_oracles": list(record.get("mandatory_oracles") or ["voice"]),
            "preconditions": record.get("preconditions") or {},
            "cleanup": str(record.get("cleanup") or ""),
            "timeout_ms": int(record.get("timeout_ms") or 0),
            "requirement_refs": [{"id": requirement.id, "source_ref": requirement.source_ref}],
        }
        candidate = TestCaseCandidate(
            id=case_id,
            requirement_ids=[requirement.id],
            case=case,
            strategy_names=["requirement_generation"],
            rationale=str(record.get("rationale") or ""),
        )
        candidate.fingerprint = fingerprint(candidate)
        return candidate


def fingerprint(candidate: TestCaseCandidate) -> str:
    case = candidate.case
    stable = {
        "requirements": sorted(candidate.requirement_ids),
        "utterance": case.get("utterance", ""),
        "domain": case.get("domain", ""),
        "expected_response": case.get("expected_response", ""),
        "preconditions": case.get("preconditions", {}),
        "expected_semantics": case.get("expected_semantics", {}),
        "expected_semantic_sequence": case.get("expected_semantic_sequence", []),
        "expected_cockpit_log_patterns": case.get("expected_cockpit_log_patterns", []),
        "expected_rejection": case.get("expected_rejection", False),
        "expected_response_patterns": case.get("expected_response_patterns", []),
        "expected_can_signals": case.get("expected_can_signals", []),
        "expected_ui_changes": case.get("expected_ui_changes", []),
        "mandatory_oracles": case.get("mandatory_oracles", []),
        "parameters": case.get("test_parameters", {}),
        "dialogue": case.get("dialogue", {}),
        "full_duplex": case.get("full_duplex", {}),
    }
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _infer_domain(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("空调", "温度", "风量", "climate")):
        return "climate"
    if any(token in normalized for token in ("车窗", "窗户", "window")):
        return "window"
    if any(token in normalized for token in ("座椅", "seat")):
        return "seat"
    if any(token in normalized for token in ("导航", "路线", "navigation")):
        return "navigation"
    if any(token in normalized for token in ("音乐", "播放", "媒体", "media")):
        return "media"
    return "information"


def _extract_utterance(text: str) -> str:
    quoted = re.search(r"[“\"]([^”\"]+)[”\"]", text)
    if quoted:
        return quoted.group(1).strip()[:160]
    for line in text.split("\n"):
        candidate = line.strip("- *:：")
        if candidate:
            return candidate[:160]
    return text[:160]
