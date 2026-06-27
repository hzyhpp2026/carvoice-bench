"""Structured records exchanged by the self-improving test agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NEEDS_REVIEW = "needs_review"


@dataclass
class Requirement:
    id: str
    source_path: str
    source_ref: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Requirement":
        return cls(**value)


@dataclass
class TestCaseCandidate:
    id: str
    requirement_ids: List[str]
    case: Dict[str, Any]
    strategy_names: List[str] = field(default_factory=list)
    status: str = "candidate"
    fingerprint: str = ""
    parent_candidate_id: Optional[str] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TestCaseCandidate":
        return cls(**value)


@dataclass
class ExecutionEvidence:
    id: str
    candidate_id: str
    run_id: str
    started_at: str
    duration_ms: float
    oracles: Dict[str, Dict[str, Any]]
    artifacts: Dict[str, str] = field(default_factory=dict)
    device_state: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ExecutionEvidence":
        return cls(**value)


@dataclass
class Finding:
    id: str
    candidate_id: str
    category: str
    hypothesis: str
    evidence_ids: List[str]
    reproducible: bool
    status: str = "candidate"

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class Strategy:
    name: str
    description: str
    state: str = "approved"
    attempts: int = 0
    reward_total: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


@dataclass
class SkillRevision:
    id: str
    name: str
    state: str
    instructions: str
    contract: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return model_to_dict(self)


def model_to_dict(value: Any) -> Any:
    """Return JSON-compatible data without leaking Enum implementation details."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: model_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): model_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_dict(item) for item in value]
    return value
