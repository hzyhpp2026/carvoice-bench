"""Evidence fusion and failure categorization for agent executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from carvoice_bench.agent.models import ExecutionEvidence, TestCaseCandidate, Verdict


@dataclass
class VerdictResult:
    verdict: Verdict
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0


def decide(candidate: TestCaseCandidate, evidence: ExecutionEvidence, high_confidence: float = 0.8) -> VerdictResult:
    if evidence.error:
        return VerdictResult(Verdict.INCONCLUSIVE, ["execution environment: " + evidence.error], 0.0)
    mandatory = candidate.case.get("mandatory_oracles", [])
    if not mandatory:
        return VerdictResult(Verdict.NEEDS_REVIEW, ["candidate has no mandatory oracle"], 0.0)

    confidence_values: List[float] = []
    missing: List[str] = []
    weak_failure: List[str] = []
    strong_failure: List[str] = []
    for name in mandatory:
        result = evidence.oracles.get(name)
        if not result or not result.get("available"):
            missing.append(name)
            continue
        confidence = float(result.get("confidence", 0.0))
        confidence_values.append(confidence)
        if result.get("matched"):
            continue
        detail = str(result.get("detail", "mismatch"))
        if confidence >= high_confidence:
            strong_failure.append(name + ": " + detail)
        else:
            weak_failure.append(name + ": " + detail)
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    if strong_failure:
        return VerdictResult(Verdict.FAIL, strong_failure, confidence)
    if missing:
        return VerdictResult(Verdict.INCONCLUSIVE, ["missing evidence: " + ", ".join(missing)], confidence)
    if weak_failure:
        return VerdictResult(Verdict.NEEDS_REVIEW, weak_failure, confidence)
    return VerdictResult(Verdict.PASS, [], confidence)


def classify_failure(evidence: ExecutionEvidence) -> str:
    if evidence.error:
        return "environment"
    for name, result in evidence.oracles.items():
        if result.get("available") and not result.get("matched"):
            detail = str(result.get("detail", "")).lower()
            if name == "voice":
                if "wake" in detail or "唤醒" in detail:
                    return "wakeup"
                return "asr"
            if name == "semantic":
                return "nlu"
            if name == "ui":
                return "ui"
            if name == "can":
                return "can"
            if "latency" in detail or "timeout" in detail:
                return "timing"
    return "unknown"
