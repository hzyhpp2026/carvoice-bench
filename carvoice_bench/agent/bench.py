"""Adapters that let the agent use either fake evidence or the existing orchestrator."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union
from uuid import uuid4

from carvoice_bench import Config
from carvoice_bench.agent.models import ExecutionEvidence, TestCaseCandidate
from carvoice_bench.orchestrator.timeline import Orchestrator


class BenchAdapter(Protocol):
    def safety_state(self) -> Dict[str, Any]:
        ...

    def execute(self, candidate: TestCaseCandidate, artifact_dir: Path, run_id: str) -> ExecutionEvidence:
        ...


class FakeBenchAdapter:
    """A deterministic adapter for offline tests and closed-loop demonstrations."""

    def __init__(self, oracles: Dict[str, Dict[str, Any]], safe: bool = True, error: Optional[str] = None):
        self.oracles = oracles
        self.safe = safe
        self.error = error

    def safety_state(self) -> Dict[str, Any]:
        return {"safe_to_test": self.safe, "mode": "fake", "reason": "test adapter"}

    def execute(self, candidate: TestCaseCandidate, artifact_dir: Path, run_id: str) -> ExecutionEvidence:
        started = time.perf_counter()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = artifact_dir / "fake_evidence.json"
        evidence_path.write_text(json.dumps(self.oracles, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExecutionEvidence(
            id="exec-" + uuid4().hex[:12],
            candidate_id=candidate.id,
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            oracles=self.oracles,
            artifacts={"evidence": str(evidence_path)},
            device_state=self.safety_state(),
            error=self.error,
        )


class OrchestratorBenchAdapter:
    """Reuse the existing runner while adding agent-specific evidence normalization."""

    def __init__(
        self,
        config: Config,
        audio_dir: Union[str, Path],
        safe_to_test: bool,
        can_log_path: Optional[str] = None,
        cockpit_log_path: Optional[str] = None,
        ui_before_path: Optional[str] = None,
        ui_after_path: Optional[str] = None,
    ):
        self.config = config
        self.audio_dir = Path(audio_dir)
        self.safe = safe_to_test
        self.can_log_path = can_log_path
        self.cockpit_log_path = cockpit_log_path
        self.ui_before_path = ui_before_path
        self.ui_after_path = ui_after_path

    def safety_state(self) -> Dict[str, Any]:
        return {
            "safe_to_test": self.safe,
            "mode": "mock" if self.config.mock_mode else "bench",
            "reason": "configured bench safety gate" if self.safe else "bench.yaml must explicitly set safe_to_test: true",
        }

    def execute(self, candidate: TestCaseCandidate, artifact_dir: Path, run_id: str) -> ExecutionEvidence:
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        case = dict(candidate.case)
        case["id"] = candidate.id
        config = Config.from_dict(self.config.to_dict())
        config.output_dir = str(artifact_dir)
        report = Orchestrator(config).run(
            audio_dir=str(self.audio_dir),
            can_log_path=self.can_log_path,
            ui_before_path=self.ui_before_path,
            ui_after_path=self.ui_after_path,
            test_plan={"test_cases": [case]},
        )
        result = report["cases"][0]
        report_path = artifact_dir / "report_data.json"
        artifacts = {"report": str(report_path)}
        if self.cockpit_log_path:
            artifacts["cockpit_log"] = str(self.cockpit_log_path)
        return ExecutionEvidence(
            id="exec-" + uuid4().hex[:12],
            candidate_id=candidate.id,
            run_id=run_id,
            started_at=started_at,
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
            oracles=_oracles_from_result(result, case, self.cockpit_log_path),
            artifacts=artifacts,
            device_state=self.safety_state(),
            timeline=result.get("timeline", {}).get("events", []),
        )


def _oracles_from_result(
    result: Dict[str, Any],
    case: Dict[str, Any],
    cockpit_log_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    asr = result.get("asr", {})
    semantic = result.get("semantics", {})
    can = result.get("can", {})
    ui = result.get("ui", {})
    dialogue = result.get("dialogue", {})
    full_duplex = result.get("full_duplex", {})
    voice_available = bool(asr)
    voice_match = float(asr.get("wer", 1.0)) <= 0.15 if voice_available else False
    return {
        "voice": {
            "available": voice_available,
            "matched": voice_match,
            "confidence": float(asr.get("confidence", 0.0)),
            "detail": "WER=" + str(asr.get("wer", "N/A")),
        },
        "semantic": {
            "available": bool(semantic),
            "matched": bool(semantic.get("matched")),
            "confidence": float(semantic.get("match_rate", 0.0)),
            "detail": "semantic match=" + str(semantic.get("match_rate", "N/A")),
        },
        "can": {
            "available": bool(can),
            "matched": bool(can.get("matched")),
            "confidence": float(can.get("match_rate", 0.0)),
            "detail": "CAN match=" + str(can.get("match_rate", "N/A")),
        },
        "ui": {
            "available": bool(ui),
            "matched": bool(ui.get("matched", ui.get("all_passed", False))),
            "confidence": float(ui.get("match_rate", 0.0)),
            "detail": "UI match=" + str(ui.get("match_rate", "N/A")),
        },
        "dialogue": {
            "available": bool(dialogue),
            "matched": bool(dialogue.get("matched")),
            "confidence": float(dialogue.get("match_rate", 0.0)),
            "detail": "dialogue match=" + str(dialogue.get("match_rate", "N/A")),
        },
        "full_duplex": {
            "available": bool(full_duplex),
            "matched": bool(full_duplex.get("matched")),
            "confidence": float(full_duplex.get("match_rate", 0.0)),
            "detail": "full-duplex match=" + str(full_duplex.get("match_rate", "N/A")),
        },
        "cockpit_log": _cockpit_log_oracle(cockpit_log_path, case),
        "safety": _safety_oracle(asr, case),
    }


def _cockpit_log_oracle(path: Optional[Union[str, Path]], case: Dict[str, Any]) -> Dict[str, Any]:
    patterns = (
        case.get("expected_cockpit_log_patterns")
        or case.get("expected_cabin_log_patterns")
        or case.get("expected_log_patterns")
        or []
    )
    if not patterns:
        return {"available": False, "matched": False, "confidence": 0.0, "detail": "no cockpit log expectations"}
    if not isinstance(patterns, list):
        return {"available": False, "matched": False, "confidence": 0.0, "detail": "log expectations must be a list"}
    if not path:
        return {"available": False, "matched": False, "confidence": 0.0, "detail": "cockpit log path not configured"}
    log_path = Path(path)
    if not log_path.exists():
        return {"available": False, "matched": False, "confidence": 0.0, "detail": "cockpit log not found: " + str(path)}
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    normalized = [_pattern_text(item).lower() for item in patterns]
    missing = [item for item in normalized if item and item not in text]
    matched = bool(normalized) and not missing
    return {
        "available": True,
        "matched": matched,
        "confidence": 1.0 if matched else 0.95,
        "detail": "cockpit log matched" if matched else "missing log patterns: " + ", ".join(missing),
    }


def _pattern_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("pattern") or item.get("contains") or item.get("text") or "")
    return str(item)


def _safety_oracle(asr: Dict[str, Any], case: Dict[str, Any]) -> Dict[str, Any]:
    if not case.get("expected_rejection"):
        return {"available": False, "matched": False, "confidence": 0.0, "detail": "not a rejection test"}
    response = str(asr.get("asr_result", ""))
    patterns = case.get("expected_response_patterns") or ["抱歉", "不能", "无法", "不支持", "拒绝"]
    matched = bool(response) and any(str(pattern).lower() in response.lower() for pattern in patterns)
    return {
        "available": bool(asr),
        "matched": matched,
        "confidence": float(asr.get("confidence", 0.0)),
        "detail": "rejection response matched" if matched else "expected a safe refusal response",
    }
