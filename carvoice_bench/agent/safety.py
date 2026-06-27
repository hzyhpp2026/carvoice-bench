"""Safety policy for exploratory vehicle voice tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set

from carvoice_bench.agent.models import TestCaseCandidate


DEFAULT_ALLOWED_DOMAINS = {
    "climate", "window", "seat", "media", "navigation", "information", "communication",
    "vehicle_status", "ambient", "app", "cross_domain",
}
DEFAULT_FORBIDDEN_TERMS = {
    "加速", "刹车", "制动", "转向", "方向盘", "变道", "超车", "自动驾驶", "辅助驾驶",
    "drive", "accelerate", "brake", "steer", "autonomous driving",
}


@dataclass
class SafetyPolicy:
    allowed_domains: Set[str] = field(default_factory=lambda: set(DEFAULT_ALLOWED_DOMAINS))
    forbidden_terms: Set[str] = field(default_factory=lambda: set(DEFAULT_FORBIDDEN_TERMS))
    required_vehicle_states: Set[str] = field(default_factory=lambda: {"bench", "parked"})
    max_timeout_ms: int = 30000

    @classmethod
    def from_rules(cls, rules: Dict[str, Any]) -> "SafetyPolicy":
        safety = rules.get("safety", rules) if isinstance(rules, dict) else {}
        return cls(
            allowed_domains=set(safety.get("allowed_domains", DEFAULT_ALLOWED_DOMAINS)),
            forbidden_terms=set(safety.get("forbidden_terms", DEFAULT_FORBIDDEN_TERMS)),
            required_vehicle_states=set(safety.get("required_vehicle_states", {"bench", "parked"})),
            max_timeout_ms=int(safety.get("max_timeout_ms", 30000)),
        )

    def validate(self, candidate: TestCaseCandidate) -> List[str]:
        case = candidate.case
        violations: List[str] = []
        domain = str(case.get("domain", "")).strip().lower()
        if domain not in self.allowed_domains:
            violations.append("domain is missing or not allowlisted")

        text = " ".join(
            str(part) for part in (
                case.get("utterance", ""), case.get("description", ""), case.get("expected_response", ""),
            )
        ).lower()
        expected_rejection = bool(case.get("expected_rejection"))
        matched_terms = sorted(term for term in self.forbidden_terms if term.lower() in text)
        if matched_terms and not expected_rejection:
            violations.append("forbidden control term: " + ", ".join(matched_terms))

        preconditions = case.get("preconditions")
        if not isinstance(preconditions, dict):
            violations.append("preconditions are required")
        else:
            state = str(preconditions.get("vehicle_state", "")).lower()
            if state not in self.required_vehicle_states:
                violations.append("vehicle_state must be bench or parked")
        if not case.get("cleanup"):
            violations.append("cleanup is required")
        timeout_ms = case.get("timeout_ms")
        if not isinstance(timeout_ms, int) or not 0 < timeout_ms <= self.max_timeout_ms:
            violations.append("timeout_ms is missing or outside safety limit")
        if not candidate.requirement_ids:
            violations.append("requirement trace is required")
        if not case.get("mandatory_oracles"):
            violations.append("at least one mandatory oracle is required")
        if expected_rejection:
            if case.get("expected_can_signals") or case.get("expected_ui_changes"):
                violations.append("rejection tests must not expect vehicle execution")
            if "safety" not in case.get("mandatory_oracles", []):
                violations.append("rejection tests require the safety oracle")
        return violations


def require_safe_device_state(device_state: Dict[str, Any]) -> None:
    if not device_state.get("safe_to_test"):
        reason = device_state.get("reason", "bench adapter did not confirm a safe state")
        raise RuntimeError("execution blocked by safety gate: " + str(reason))


def allowed_domains() -> Iterable[str]:
    return sorted(DEFAULT_ALLOWED_DOMAINS)
