"""Execution loop that applies safety checks, persists evidence, and creates findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

from carvoice_bench.agent.bench import BenchAdapter
from carvoice_bench.agent.models import Finding, TestCaseCandidate, Verdict
from carvoice_bench.agent.safety import SafetyPolicy, require_safe_device_state
from carvoice_bench.agent.storage import AgentStore
from carvoice_bench.agent.verdict import VerdictResult, classify_failure, decide


@dataclass
class ExecutionSummary:
    candidate_id: str
    verdict: Verdict
    verdicts: List[Verdict]
    evidence_ids: List[str]
    finding_id: Optional[str] = None
    reasons: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "verdict": self.verdict.value,
            "verdicts": [item.value for item in self.verdicts],
            "evidence_ids": self.evidence_ids,
            "finding_id": self.finding_id,
            "reasons": self.reasons or [],
        }


class AgentExecutor:
    def __init__(self, store: AgentStore, safety_policy: SafetyPolicy):
        self.store = store
        self.safety_policy = safety_policy

    def execute_candidate(
        self,
        run_id: str,
        candidate: TestCaseCandidate,
        adapter: BenchAdapter,
        repetitions: int = 1,
    ) -> ExecutionSummary:
        violations = self.safety_policy.validate(candidate)
        if violations:
            raise ValueError("candidate blocked by safety policy: " + "; ".join(violations))
        require_safe_device_state(adapter.safety_state())

        results: List[VerdictResult] = []
        evidence_ids: List[str] = []
        evidence_records = []
        for attempt in range(max(1, repetitions)):
            artifact_dir = self.store.artifacts_dir / run_id / candidate.id / f"attempt-{attempt + 1}"
            evidence = adapter.execute(candidate, artifact_dir, run_id)
            decision = decide(candidate, evidence)
            self.store.record_execution(evidence, decision.verdict)
            results.append(decision)
            evidence_ids.append(evidence.id)
            evidence_records.append(evidence)

        verdicts = [result.verdict for result in results]
        failures = verdicts.count(Verdict.FAIL)
        is_stable_failure = repetitions >= 3 and failures >= 2
        if is_stable_failure:
            self.store.update_candidate_status(candidate.id, "needs_review")
            representative = next(
                evidence for evidence, result in zip(evidence_records, results) if result.verdict == Verdict.FAIL
            )
            finding = Finding(
                id="finding-" + uuid4().hex[:12],
                candidate_id=candidate.id,
                category=classify_failure(representative),
                hypothesis="High-confidence failure reproduced in at least two of three attempts.",
                evidence_ids=evidence_ids,
                reproducible=True,
            )
            self.store.record_finding(run_id, finding)
            return ExecutionSummary(candidate.id, Verdict.FAIL, verdicts, evidence_ids, finding.id, _reasons(results))

        if Verdict.FAIL in verdicts or Verdict.NEEDS_REVIEW in verdicts or Verdict.INCONCLUSIVE in verdicts:
            self.store.update_candidate_status(candidate.id, "needs_review")
        else:
            self.store.update_candidate_status(candidate.id, "executed")
        final = Verdict.FAIL if Verdict.FAIL in verdicts else _most_severe(verdicts)
        return ExecutionSummary(candidate.id, final, verdicts, evidence_ids, reasons=_reasons(results))


def _most_severe(verdicts: List[Verdict]) -> Verdict:
    if Verdict.NEEDS_REVIEW in verdicts:
        return Verdict.NEEDS_REVIEW
    if Verdict.INCONCLUSIVE in verdicts:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


def _reasons(results: List[VerdictResult]) -> List[str]:
    return [reason for result in results for reason in result.reasons]
