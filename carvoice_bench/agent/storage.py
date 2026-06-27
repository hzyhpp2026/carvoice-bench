"""SQLite-backed agent memory, review queue, and append-only audit trail."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4

from carvoice_bench.agent.models import (
    ExecutionEvidence,
    Finding,
    Requirement,
    SkillRevision,
    Strategy,
    TestCaseCandidate,
    Verdict,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStore:
    def __init__(self, workspace: Union[str, Path]):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.workspace / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self.db_path = self.workspace / "agent.sqlite3"
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requirements (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_fingerprint
                    ON candidates(run_id, fingerprint);
                CREATE TABLE IF NOT EXISTS executions (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    verdict TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS findings (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                    category TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, name TEXT NOT NULL,
                    state TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    candidate_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
                    note TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, event TEXT NOT NULL,
                    entity_type TEXT NOT NULL, entity_id TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                """
            )

    def create_run(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        run_id = "run-" + uuid4().hex[:12]
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(id, created_at, metadata_json) VALUES (?, ?, ?)",
                (run_id, _now(), json.dumps(metadata or {}, ensure_ascii=False)),
            )
        self._audit("create", "run", run_id, metadata or {})
        return run_id

    def save_requirements(self, run_id: str, requirements: Iterable[Requirement]) -> None:
        rows = [(item.id, run_id, json.dumps(item.to_dict(), ensure_ascii=False)) for item in requirements]
        with self._connect() as connection:
            connection.executemany("INSERT OR REPLACE INTO requirements VALUES (?, ?, ?)", rows)
        for requirement_id, _, payload in rows:
            self._audit("ingest", "requirement", requirement_id, json.loads(payload))

    def list_requirements(self, run_id: str) -> List[Requirement]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload_json FROM requirements WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        return [Requirement.from_dict(json.loads(row["payload_json"])) for row in rows]

    def save_candidate(self, run_id: str, candidate: TestCaseCandidate) -> bool:
        payload = json.dumps(candidate.to_dict(), ensure_ascii=False)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?)",
                    (candidate.id, run_id, candidate.status, candidate.fingerprint, payload, _now()),
                )
        except sqlite3.IntegrityError:
            return False
        self._audit("propose", "candidate", candidate.id, candidate.to_dict())
        return True

    def get_candidate(self, candidate_id: str) -> TestCaseCandidate:
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            raise KeyError("candidate not found: " + candidate_id)
        return TestCaseCandidate.from_dict(json.loads(row["payload_json"]))

    def list_candidates(self, run_id: str, status: Optional[str] = None) -> List[TestCaseCandidate]:
        sql = "SELECT payload_json FROM candidates WHERE run_id = ?"
        params: List[Any] = [run_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [TestCaseCandidate.from_dict(json.loads(row["payload_json"])) for row in rows]

    def update_candidate_status(self, candidate_id: str, status: str) -> None:
        candidate = self.get_candidate(candidate_id)
        candidate.status = status
        with self._connect() as connection:
            connection.execute(
                "UPDATE candidates SET status = ?, payload_json = ? WHERE id = ?",
                (status, json.dumps(candidate.to_dict(), ensure_ascii=False), candidate_id),
            )
        self._audit("status", "candidate", candidate_id, {"status": status})

    def record_execution(self, evidence: ExecutionEvidence, verdict: Verdict) -> None:
        payload = evidence.to_dict()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO executions VALUES (?, ?, ?, ?, ?, ?)",
                (evidence.id, evidence.run_id, evidence.candidate_id, verdict.value, json.dumps(payload, ensure_ascii=False), _now()),
            )
        self._audit("execute", "execution", evidence.id, {"verdict": verdict.value, **payload})

    def executions_for_candidate(self, candidate_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT verdict, payload_json FROM executions WHERE candidate_id = ? ORDER BY created_at", (candidate_id,)
            ).fetchall()
        return [{"verdict": row["verdict"], "evidence": json.loads(row["payload_json"])} for row in rows]

    def record_finding(self, run_id: str, finding: Finding) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO findings VALUES (?, ?, ?, ?, ?, ?)",
                (finding.id, run_id, finding.candidate_id, finding.category,
                 json.dumps(finding.to_dict(), ensure_ascii=False), _now()),
            )
        self._audit("propose", "finding", finding.id, finding.to_dict())

    def list_findings(self, run_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload_json FROM findings WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def save_strategy(self, strategy: Strategy) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO strategies VALUES (?, ?, ?)",
                (strategy.name, json.dumps(strategy.to_dict(), ensure_ascii=False), _now()),
            )

    def list_strategies(self) -> List[Strategy]:
        with self._connect() as connection:
            rows = connection.execute("SELECT payload_json FROM strategies ORDER BY name").fetchall()
        return [Strategy(**json.loads(row["payload_json"])) for row in rows]

    def record_strategy_reward(self, name: str, reward: float) -> Strategy:
        strategies = {item.name: item for item in self.list_strategies()}
        strategy = strategies[name]
        strategy.attempts += 1
        strategy.reward_total += reward
        self.save_strategy(strategy)
        self._audit("score", "strategy", name, {"reward": reward, "attempts": strategy.attempts})
        return strategy

    def save_skill(self, run_id: str, skill: SkillRevision) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?)",
                (skill.id, run_id, skill.name, skill.state, json.dumps(skill.to_dict(), ensure_ascii=False), _now()),
            )
        self._audit("propose", "skill", skill.id, skill.to_dict())

    def review_candidate(self, candidate_id: str, decision: str, reviewer: str, note: str = "") -> None:
        if decision not in {"approved", "rejected", "needs_revision"}:
            raise ValueError("unsupported review decision")
        status = "approved" if decision == "approved" else decision
        self.update_candidate_status(candidate_id, status)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?)",
                (candidate_id, decision, reviewer or "local-reviewer", note, _now()),
            )
        self._audit("review", "candidate", candidate_id, {"decision": decision, "reviewer": reviewer, "note": note})

    def review_queue(self, run_id: str) -> List[Dict[str, Any]]:
        candidates = self.list_candidates(run_id)
        queue: List[Dict[str, Any]] = []
        for candidate in candidates:
            executions = self.executions_for_candidate(candidate.id)
            if candidate.status in {"approved", "rejected"}:
                continue
            queue.append({
                "candidate": candidate.to_dict(),
                "executions": executions,
                "findings": [item for item in self.list_findings(run_id) if item["candidate_id"] == candidate.id],
            })
        return queue

    def approved_cases(self, run_id: str) -> List[Dict[str, Any]]:
        return [candidate.case for candidate in self.list_candidates(run_id, status="approved")]

    def _audit(self, event: str, entity_type: str, entity_id: str, payload: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO audit(created_at, event, entity_type, entity_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                (_now(), event, entity_type, entity_id, json.dumps(payload, ensure_ascii=False)),
            )
