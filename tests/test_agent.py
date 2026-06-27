"""Focused tests for the governed self-improving test-agent loop."""

from pathlib import Path

from carvoice_bench.agent.bench import FakeBenchAdapter, OrchestratorBenchAdapter
from carvoice_bench.agent.evolution import EvolutionEngine
from carvoice_bench.agent.execution import AgentExecutor
from carvoice_bench.agent.generator import CaseGenerator
from carvoice_bench.agent.llm import StaticLLMClient
from carvoice_bench.agent.models import Requirement, Verdict
from carvoice_bench.agent.requirements import RequirementIngestor
from carvoice_bench.agent.review import ReviewService
from carvoice_bench.agent.safety import SafetyPolicy
from carvoice_bench.agent.storage import AgentStore
from carvoice_bench.agent.verdict import decide
from carvoice_bench.config import Config
from carvoice_bench.orchestrator.timeline import Orchestrator
from carvoice_bench.semantic.rule_parser import RuleSemanticParser


def _rules():
    return {
        "defaults": {
            "domain": "information",
            "expected_response": "已完成",
            "mandatory_oracles": ["voice"],
            "preconditions": {"vehicle_state": "bench"},
            "cleanup": "restore state",
            "timeout_ms": 5000,
        }
    }


def _candidate(requirement):
    policy = SafetyPolicy.from_rules(_rules())
    return CaseGenerator(_rules(), policy).generate([requirement])[0]


def _pass_oracles():
    return {"voice": {"available": True, "matched": True, "confidence": 0.98, "detail": "matched"}}


def test_markdown_ingestion_preserves_source_anchor(tmp_path):
    source = tmp_path / "requirements.md"
    source.write_text("# Climate\n\nOpen the climate panel.\n", encoding="utf-8")
    requirements = RequirementIngestor().ingest(source)
    assert len(requirements) == 1
    assert requirements[0].title == "Climate"
    assert requirements[0].source_ref == "lines:1-3"


def test_heuristic_generation_prefers_a_quoted_voice_command():
    requirement = Requirement("req-quoted", "spec.md", "lines:1", "Climate", "用户说“打开主驾空调到26度”后应确认。")
    candidate = _candidate(requirement)
    assert candidate.case["utterance"] == "打开主驾空调到26度"


def test_generation_rejects_dangerous_llm_candidate():
    requirement = Requirement("req-1", "spec.md", "lines:1", "Vehicle", "A safe request")
    client = StaticLLMClient({"candidates": [{
        "description": "unsafe", "utterance": "请加速到 100", "domain": "information",
        "expected_response": "", "mandatory_oracles": ["voice"],
        "preconditions": {"vehicle_state": "bench"}, "cleanup": "restore", "timeout_ms": 1000,
    }]})
    generator = CaseGenerator(_rules(), SafetyPolicy.from_rules(_rules()), client)
    assert generator.generate([requirement]) == []


def test_execution_review_and_regression_export(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-1", "spec.md", "lines:1", "Info", "查询车辆状态")
    store.save_requirements(run_id, [requirement])
    candidate = _candidate(requirement)
    assert store.save_candidate(run_id, candidate)
    executor = AgentExecutor(store, SafetyPolicy.from_rules(_rules()))
    summary = executor.execute_candidate(run_id, candidate, FakeBenchAdapter(_pass_oracles()))
    assert summary.verdict == Verdict.PASS
    store.review_candidate(candidate.id, "approved", "tester", "stable smoke case")
    output = ReviewService(store, run_id).export_approved_cases()
    assert candidate.id in output.read_text(encoding="utf-8")


def test_reproducible_failure_creates_finding(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-2", "spec.md", "lines:2", "Info", "查询车辆状态")
    candidate = _candidate(requirement)
    store.save_candidate(run_id, candidate)
    failing = {"voice": {"available": True, "matched": False, "confidence": 0.95, "detail": "wake response missing"}}
    summary = AgentExecutor(store, SafetyPolicy.from_rules(_rules())).execute_candidate(
        run_id, candidate, FakeBenchAdapter(failing), repetitions=3
    )
    assert summary.verdict == Verdict.FAIL
    assert summary.finding_id
    assert store.list_findings(run_id)[0]["category"] == "wakeup"


def test_evolution_creates_reviewable_variant(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-3", "spec.md", "lines:3", "Info", "查询车辆状态")
    base = _candidate(requirement)
    store.save_candidate(run_id, base)
    policy = SafetyPolicy.from_rules(_rules())
    outcomes = EvolutionEngine(store, policy, AgentExecutor(store, policy)).evolve(
        run_id, FakeBenchAdapter(_pass_oracles()), max_iterations=1
    )
    assert len(outcomes) == 1
    assert len(store.list_candidates(run_id)) == 2


def test_evolution_honors_enabled_strategy_parameters_and_attempt_budget(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-5", "spec.md", "lines:5", "Info", "打开车辆状态")
    base = _candidate(requirement)
    base.case["utterance"] = "打开车辆状态"
    store.save_candidate(run_id, base)
    policy = SafetyPolicy.from_rules(_rules())
    config = {
        "enabled_strategies": ["synonym"],
        "ucb_exploration_coefficient": 0.0,
        "strategies": {
            "synonym": {
                "priority": 2.0,
                "max_attempts": 1,
                "replacements": [{"from": "打开", "to": "启用"}],
            }
        },
    }
    engine = EvolutionEngine(store, policy, AgentExecutor(store, policy), exploration_config=config)
    outcomes = engine.evolve(run_id, FakeBenchAdapter(_pass_oracles()), max_iterations=3)
    assert len(outcomes) == 1
    strategies = store.list_strategies()
    assert [item.name for item in strategies if item.state == "approved"] == ["synonym"]
    variants = store.list_candidates(run_id)
    assert variants[1].case["utterance"] == "启用车辆状态"
    assert next(item for item in strategies if item.name == "synonym").config["max_attempts"] == 1


def test_evolution_supports_constrained_custom_case_patch(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-6", "spec.md", "lines:6", "Info", "查询车辆状态")
    base = _candidate(requirement)
    store.save_candidate(run_id, base)
    policy = SafetyPolicy.from_rules(_rules())
    config = {
        "enabled_strategies": ["cabin_noise_profile"],
        "strategies": {
            "cabin_noise_profile": {
                "type": "case_patch",
                "case_patch": {"test_parameters": {"noise_profile": "cabin_low", "speaker_distance_cm": 70}},
            }
        },
    }
    outcomes = EvolutionEngine(store, policy, AgentExecutor(store, policy), exploration_config=config).evolve(
        run_id, FakeBenchAdapter(_pass_oracles()), max_iterations=1
    )
    assert len(outcomes) == 1
    variant = store.list_candidates(run_id)[1]
    assert variant.strategy_names[-1] == "cabin_noise_profile"
    assert variant.case["test_parameters"]["noise_profile"] == "cabin_low"


def test_missing_oracle_is_inconclusive(tmp_path):
    requirement = Requirement("req-4", "spec.md", "lines:4", "Info", "查询车辆状态")
    candidate = _candidate(requirement)
    evidence = FakeBenchAdapter({}).execute(candidate, Path(tmp_path), "run")
    assert decide(candidate, evidence).verdict == Verdict.INCONCLUSIVE


def test_compound_semantics_are_evaluated_as_an_ordered_sequence(tmp_path):
    expected = [
        {"intent": "control_window", "slots": {"action": "open", "target": "window"}},
        {"intent": "control_media", "slots": {"action": "play"}},
    ]
    parsed = RuleSemanticParser().parse_many("打开车窗并播放音乐")
    assert [item["intent"] for item in parsed] == ["control_window", "control_media"]
    report = Orchestrator(Config(mock_mode=True, output_dir=str(tmp_path / "report"))).run(
        audio_dir=str(tmp_path),
        test_plan={"test_cases": [{
            "id": "compound-001",
            "utterance": "打开车窗并播放音乐",
            "expected_asr": "打开车窗并播放音乐",
            "expected_semantic_sequence": expected,
            "timeout_ms": 1000,
        }]},
    )
    assert report["cases"][0]["semantics"]["matched"] is True
    assert report["cases"][0]["semantics"]["expected_count"] == 2


def test_cockpit_semantic_strategy_uses_only_its_oracle(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-7", "spec.md", "lines:7", "Climate", "打开空调")
    base = _candidate(requirement)
    store.save_candidate(run_id, base)
    policy = SafetyPolicy.from_rules(_rules())
    config = {"enabled_strategies": ["compound_command"]}
    outcomes = EvolutionEngine(store, policy, AgentExecutor(store, policy), exploration_config=config).evolve(
        run_id,
        FakeBenchAdapter({"semantic": {"available": True, "matched": True, "confidence": 0.95, "detail": "sequence matched"}}),
        max_iterations=1,
    )
    assert len(outcomes) == 1
    variant = store.list_candidates(run_id)[1]
    assert variant.case["mandatory_oracles"] == ["semantic"]
    assert len(variant.case["expected_semantic_sequence"]) == 2
    assert variant.case["expected_can_signals"] == []


def test_rejection_strategy_is_allowed_without_vehicle_execution(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-8", "spec.md", "lines:8", "Info", "系统安全性")
    base = _candidate(requirement)
    store.save_candidate(run_id, base)
    policy = SafetyPolicy.from_rules(_rules())
    config = {"enabled_strategies": ["rejection_boundary"]}
    outcomes = EvolutionEngine(store, policy, AgentExecutor(store, policy), exploration_config=config).evolve(
        run_id,
        FakeBenchAdapter({"safety": {"available": True, "matched": True, "confidence": 0.98, "detail": "safe refusal"}}),
        max_iterations=1,
    )
    assert len(outcomes) == 1
    variant = store.list_candidates(run_id)[1]
    assert variant.case["expected_rejection"] is True
    assert variant.case["mandatory_oracles"] == ["safety"]
    assert policy.validate(variant) == []


def test_cockpit_log_oracle_can_be_a_primary_execution_signal(tmp_path):
    store = AgentStore(tmp_path / "workspace")
    run_id = store.create_run()
    requirement = Requirement("req-log", "spec.md", "lines:9", "Climate", "空调日志判定")
    candidate = _candidate(requirement)
    candidate.case["mandatory_oracles"] = ["cockpit_log"]
    candidate.case["expected_cockpit_log_patterns"] = [
        "intent=control_climate",
        "AC_TEMP_SET=23",
        {"pattern": "fan_level=2"},
    ]
    candidate.fingerprint = "log-oracle"
    store.save_candidate(run_id, candidate)

    log_path = tmp_path / "cockpit.log"
    log_path.write_text(
        "voice intent=control_climate slots zone=driver AC_TEMP_SET=23 fan_level=2\n",
        encoding="utf-8",
    )
    adapter = OrchestratorBenchAdapter(
        Config(mock_mode=True, output_dir=str(tmp_path / "report")),
        audio_dir=tmp_path,
        safe_to_test=True,
        cockpit_log_path=str(log_path),
    )
    policy = SafetyPolicy.from_rules(_rules())
    summary = AgentExecutor(store, policy).execute_candidate(run_id, candidate, adapter)

    assert summary.verdict == Verdict.PASS
    execution = store.executions_for_candidate(candidate.id)[0]
    assert execution["evidence"]["oracles"]["cockpit_log"]["matched"] is True
    assert execution["evidence"]["artifacts"]["cockpit_log"] == str(log_path)


def test_associated_metric_slice_strategies_generate_traceable_variants(tmp_path):
    strategy_oracles = {
        "intent_slot_accuracy": {
            "semantic": {"available": True, "matched": True, "confidence": 0.97, "detail": "semantic slice matched"},
        },
        "multi_turn_retention": {
            "dialogue": {"available": True, "matched": True, "confidence": 0.96, "detail": "dialogue slice matched"},
        },
        "clarification_correctness": {
            "dialogue": {"available": True, "matched": True, "confidence": 0.96, "detail": "clarification matched"},
        },
        "false_wakeup_rate": {
            "full_duplex": {"available": True, "matched": True, "confidence": 0.98, "detail": "no false wakeup"},
        },
        "audio_zone_crosstalk_rate": {
            "semantic": {"available": True, "matched": True, "confidence": 0.97, "detail": "zone intent matched"},
            "full_duplex": {"available": True, "matched": True, "confidence": 0.95, "detail": "zone isolation matched"},
        },
        "execution_consistency": {
            "semantic": {"available": True, "matched": True, "confidence": 0.97, "detail": "semantic matched"},
            "can": {"available": True, "matched": True, "confidence": 0.99, "detail": "CAN matched"},
            "ui": {"available": True, "matched": True, "confidence": 0.98, "detail": "UI matched"},
        },
        "degradation_authenticity": {
            "dialogue": {"available": True, "matched": True, "confidence": 0.96, "detail": "degradation truthful"},
        },
    }
    for index, (strategy_name, oracles) in enumerate(strategy_oracles.items(), start=1):
        store = AgentStore(tmp_path / strategy_name)
        run_id = store.create_run()
        requirement = Requirement(f"req-slice-{index}", "voice.xlsx", f"Sheet1!A{index}", "Metric", "关联指标切片")
        base = _candidate(requirement)
        store.save_candidate(run_id, base)
        policy = SafetyPolicy.from_rules(_rules())
        outcomes = EvolutionEngine(
            store,
            policy,
            AgentExecutor(store, policy),
            exploration_config={"enabled_strategies": [strategy_name]},
        ).evolve(run_id, FakeBenchAdapter(oracles), max_iterations=1)

        assert len(outcomes) == 1, strategy_name
        assert not outcomes[0].get("skipped"), strategy_name
        variant = store.list_candidates(run_id)[1]
        metric_slice = variant.case["test_parameters"]["metric_slice"]
        assert metric_slice["name"] == strategy_name
        assert metric_slice["source"] == "关联指标"
        assert variant.case["test_parameters"]["associated_metric"] == strategy_name
        assert variant.strategy_names[-1] == strategy_name
        assert policy.validate(variant) == []

        if strategy_name == "execution_consistency":
            assert variant.case["expected_can_signals"]
            assert variant.case["expected_ui_changes"]
        if strategy_name == "degradation_authenticity":
            assert variant.case["preconditions"]["vehicle_state"] == "bench"
            assert any(key in variant.case["preconditions"] for key in ("network", "gps"))
