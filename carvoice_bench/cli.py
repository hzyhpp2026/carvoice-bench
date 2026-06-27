"""CLI 入口：基于 click 的命令行工具"""

import sys
import json
from pathlib import Path
from typing import Optional

import click

from carvoice_bench import __version__
from carvoice_bench.agent.bench import OrchestratorBenchAdapter
from carvoice_bench.agent.evolution import EvolutionEngine
from carvoice_bench.agent.execution import AgentExecutor
from carvoice_bench.agent.generator import CaseGenerator
from carvoice_bench.agent.llm import build_llm_client
from carvoice_bench.agent.requirements import RequirementIngestor, load_rules
from carvoice_bench.agent.review import ReviewService
from carvoice_bench.agent.safety import SafetyPolicy
from carvoice_bench.agent.storage import AgentStore
from carvoice_bench.config import Config
from carvoice_bench.orchestrator.timeline import Orchestrator
from carvoice_bench.report.report_api import ReportGenerator
from carvoice_bench.utils.env import first_env, load_env_file
from carvoice_bench.utils.logger import setup_logger


@click.group()
@click.version_option(version=__version__, prog_name="carvoice-bench")
def main():
    """CarVoice Bench — 车载语音自动化评测框架"""


@main.command()
@click.option("--audio-dir", "-a", required=True, type=click.Path(exists=True), help="音频目录 (WAV)")
@click.option("--can-log", "-c", required=False, type=click.Path(exists=True), help="CAN日志路径 (.asc/.blf/.csv)")
@click.option("--ui-before", required=False, type=click.Path(exists=True), help="指令前UI截图")
@click.option("--ui-after", required=False, type=click.Path(exists=True), help="指令后UI截图")
@click.option("--test-plan", "-p", required=True, type=click.Path(exists=True), help="测试计划YAML文件")
@click.option("--output-dir", "-o", default="./carvoice_report", help="报告输出目录")
@click.option("--model-asr", default="whisper-base-zh", help="ASR模型 (whisper-base-zh / paraformer / kaldi-gpu)")
@click.option("--device", default="cpu", help="推理设备 (cpu / cuda)")
@click.option("--scoring/--no-scoring", default=True, help="是否生成综合评分表")
@click.option("--mock", "mock_mode", is_flag=True, help="使用内置 mock 链路，不加载 ASR/TTS/CAN/UI 重依赖")
@click.option("--online", "online_mode", is_flag=True, help="使用在线测试链路：缺音频时云 TTS，识别使用云 ASR")
@click.option("--env-file", default=".env", show_default=True, help="阿里云 API Key 的 .env 文件路径")
@click.option("--cloud-provider", default="aliyun", show_default=True, help="云端服务提供方")
@click.option("--cloud-asr-model", default=None, help="阿里云 DashScope ASR 模型，默认读取 .env")
@click.option("--cloud-tts-model", default=None, help="阿里云 DashScope TTS 模型，默认读取 .env")
@click.option("--cloud-tts-voice", default=None, help="阿里云 DashScope TTS 音色，默认读取 .env")
@click.option("--cloud-tts-format", default=None, help="阿里云 DashScope TTS 音频格式，默认读取 .env")
@click.option("--online-capture", is_flag=True, help="在线测试时启用播放和麦克风录音")
@click.option("--play-audio", is_flag=True, help="在线测试时把用例音频播放到默认声卡")
@click.option("--record-seconds", default=0.0, type=float, help="播放后录制默认麦克风的秒数；0 表示不录制")
@click.option("--record-video", is_flag=True, help="使用当前 PC 默认摄像头录制视频")
@click.option("--record-ui-video", is_flag=True, help="预留：录制 UI 视频，当前仅记录配置")
@click.option("--record-cabin-video", is_flag=True, help="使用当前 PC 摄像头录制车内/环境视频")
@click.option("--camera-index", default=None, type=int, help="PC 摄像头索引，默认读取 .env 或 0")
@click.option("--video-fps", default=None, type=float, help="PC 摄像头录制 FPS，默认读取 .env 或 20")
@click.option("--case-pause-seconds", default=None, type=float, help="每条用例之间的暂停秒数，默认读取 .env")
@click.option("--semantic-parser", default="rule", type=click.Choice(["rule", "none", "cloud"]), show_default=True, help="识别文本后的语义解析器")
@click.option("--semantic-rules", default=None, type=click.Path(exists=True), help="自定义语义规则 JSON/YAML 文件")
@click.option("--verbose", "-v", is_flag=True, help="详细日志输出")
@click.option("--debug", is_flag=True, help="调试模式")
def run(
    audio_dir: str,
    can_log: Optional[str],
    ui_before: Optional[str],
    ui_after: Optional[str],
    test_plan: str,
    output_dir: str,
    model_asr: str,
    device: str,
    scoring: bool,
    mock_mode: bool,
    online_mode: bool,
    env_file: str,
    cloud_provider: str,
    cloud_asr_model: str,
    cloud_tts_model: str,
    cloud_tts_voice: str,
    cloud_tts_format: str,
    online_capture: bool,
    play_audio: bool,
    record_seconds: float,
    record_video: bool,
    record_ui_video: bool,
    record_cabin_video: bool,
    camera_index: Optional[int],
    video_fps: Optional[float],
    case_pause_seconds: Optional[float],
    semantic_parser: str,
    semantic_rules: Optional[str],
    verbose: bool,
    debug: bool,
):
    """运行完整评测任务"""
    logger = setup_logger(verbose=verbose, debug=debug)

    # 加载测试计划
    plan = _load_yaml(test_plan)
    load_env_file(env_file, override=True)

    cfg = Config(
        asr_model=model_asr,
        asr_device=device,
        output_dir=output_dir,
        verbose=verbose,
        debug=debug,
        enable_scoring=scoring,
        mock_mode=mock_mode,
        online_mode=online_mode,
        env_file=env_file,
        cloud_provider=cloud_provider,
        cloud_asr_model=cloud_asr_model or first_env("ALIYUN_ASR_MODEL", "DASHSCOPE_ASR_MODEL") or Config.cloud_asr_model,
        cloud_tts_model=cloud_tts_model or first_env("ALIYUN_TTS_MODEL", "DASHSCOPE_TTS_MODEL") or Config.cloud_tts_model,
        cloud_tts_voice=cloud_tts_voice or first_env("ALIYUN_TTS_VOICE", "DASHSCOPE_TTS_VOICE") or Config.cloud_tts_voice,
        cloud_tts_format=cloud_tts_format or first_env("ALIYUN_TTS_FORMAT", "DASHSCOPE_TTS_FORMAT") or Config.cloud_tts_format,
        online_play_audio=play_audio or online_capture or _env_bool("ONLINE_PLAY_AUDIO"),
        online_record_seconds=record_seconds or _env_float("ONLINE_RECORD_SECONDS", 0.0) or (5.0 if online_capture else 0.0),
        online_record_ui_video=record_ui_video or _env_bool("ONLINE_RECORD_UI_VIDEO"),
        online_record_cabin_video=record_video or record_cabin_video or _env_bool("ONLINE_RECORD_VIDEO") or _env_bool("ONLINE_RECORD_CABIN_VIDEO"),
        online_case_pause_seconds=case_pause_seconds if case_pause_seconds is not None else _env_float("ONLINE_CASE_PAUSE_SECONDS", 0.0),
        online_camera_index=camera_index if camera_index is not None else int(_env_float("ONLINE_CAMERA_INDEX", 0)),
        online_video_fps=video_fps if video_fps is not None else _env_float("ONLINE_VIDEO_FPS", 20.0),
        semantic_parser=semantic_parser,
        semantic_rules_path=semantic_rules,
    )

    logger.info("CarVoice Bench v%s 启动", __version__)
    logger.info("测试计划: %s", test_plan)
    logger.info("ASR 模型: %s (device=%s)", model_asr, device)

    # 构建Orchestrator
    orchestrator = Orchestrator(cfg)

    # 执行评测
    report_data = orchestrator.run(
        audio_dir=audio_dir,
        can_log_path=can_log,
        ui_before_path=ui_before,
        ui_after_path=ui_after,
        test_plan=plan,
    )

    # 生成报告
    generator = ReportGenerator(cfg)
    report_paths = generator.generate(report_data)

    # 输出摘要
    _print_summary(report_data, report_paths)


@main.command()
@click.option("--output-dir", "-o", default="./carvoice_report", help="JSON报告输入目录")
@click.option("--format", "-f", type=click.Choice(["html", "pdf", "both"]), default="both", help="报告格式")
def report(output_dir: str, format: str):
    """从已有JSON结果重新生成报告"""
    logger = setup_logger()
    cfg = Config(output_dir=output_dir)
    result_path = Path(output_dir) / "report_data.json"

    if not result_path.exists():
        click.echo(f"未找到评测结果: {result_path}", err=True)
        sys.exit(1)

    with open(result_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    generator = ReportGenerator(cfg)
    paths = generator.generate(report_data, formats=[format] if format != "both" else ["html", "pdf"])
    click.echo(f"报告已生成: {paths}")


@main.command()
@click.option("--output", "-o", default="carvoice_bench_config.yaml", help="配置文件路径")
def init(output: str):
    """生成默认测试计划模板"""
    template = {
        "project": {"name": "MyCarVoiceTest", "version": "1.0.0"},
        "asr": {"model": "whisper-base-zh", "language": "zh", "sample_rate": 16000},
        "can": {"bus_type": "socketcan", "channel": "can0", "db_path": "./vehicle.dbc"},
        "test_cases": [
            {
                "id": "tc-001",
                "description": "打开主驾空调",
                "utterance": "打开主驾空调到26度",
                "expected_asr": "打开主驾空调到26度",
                "expected_can_signals": [
                    {"frame_id": "0x2A1", "signals": {"AC_MAIN_DRIVER": 1, "AC_TEMP_SET": 26}}
                ],
                "expected_ui_changes": [
                    {"element": "ac_panel", "state": "visible"},
                    {"element": "temp_display", "value": "26℃"},
                ],
                "timeout_ms": 5000,
            }
        ],
    }
    _dump_yaml(output, template)
    click.echo(f"模板已创建: {output}")


@main.group()
def agent():
    """Manage the governed self-improving test-agent workflow."""


@agent.command("ingest")
@click.option("--requirements", "requirements_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Markdown, DOCX, or PDF requirements")
@click.option("--rules", "rules_path", required=False, type=click.Path(exists=True, dir_okay=False), help="YAML or JSON test rules")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=False, help="Append requirements to an existing agent run")
def agent_ingest(requirements_path: str, rules_path: Optional[str], workspace: str, run_id: Optional[str]):
    """Normalize requirements into traceable local agent memory."""
    store = AgentStore(workspace)
    rules = load_rules(rules_path)
    run_id = run_id or store.create_run({"requirements": requirements_path, "rules_path": rules_path or "", "rules": rules})
    requirements = RequirementIngestor().ingest(requirements_path)
    store.save_requirements(run_id, requirements)
    click.echo(json.dumps({"run_id": run_id, "requirements": len(requirements)}, ensure_ascii=False))


@agent.command("generate")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=True, help="Agent run ID")
@click.option("--rules", "rules_path", required=False, type=click.Path(exists=True, dir_okay=False), help="YAML or JSON test rules")
@click.option("--provider", type=click.Choice(["auto", "heuristic", "dashscope", "openai", "compatible"]), default="auto", show_default=True)
@click.option("--model", default="qwen-plus", show_default=True, help="Generation model")
@click.option("--endpoint", default="", help="OpenAI-compatible endpoint for compatible providers")
@click.option("--env-file", default=".env", show_default=True, help="LLM environment file")
def agent_generate(workspace: str, run_id: str, rules_path: Optional[str], provider: str, model: str, endpoint: str, env_file: str):
    """Create safety-checked test-case candidates from ingested requirements."""
    load_env_file(env_file, override=True)
    store = AgentStore(workspace)
    rules = load_rules(rules_path)
    policy = SafetyPolicy.from_rules(rules)
    generator = CaseGenerator(rules, policy, build_llm_client(provider, model, endpoint))
    generated = generator.generate(store.list_requirements(run_id))
    saved = sum(1 for candidate in generated if store.save_candidate(run_id, candidate))
    blocked = len(generated) - saved
    click.echo(json.dumps({"run_id": run_id, "generated": len(generated), "saved": saved, "deduplicated": blocked}, ensure_ascii=False))


def _build_agent_adapter(
    workspace: str,
    mock: bool,
    online: bool,
    audio_dir: str,
    bench_path: Optional[str],
) -> OrchestratorBenchAdapter:
    bench = _load_yaml(bench_path) if bench_path else {}
    if online:
        load_env_file(str(bench.get("env_file", ".env")), override=True)
    config = Config(
        output_dir=str(Path(workspace) / "execution_report"),
        mock_mode=mock,
        online_mode=online,
        online_play_audio=bool(bench.get("play_audio", False)),
        online_record_seconds=float(bench.get("record_seconds", 0.0)),
        online_record_cabin_video=bool(bench.get("record_video", False)),
        online_camera_index=int(bench.get("camera_index", 0)),
        online_video_fps=float(bench.get("video_fps", 20.0)),
        semantic_parser=str(bench.get("semantic_parser", "rule")),
        semantic_rules_path=bench.get("semantic_rules"),
    )
    return OrchestratorBenchAdapter(
        config=config,
        audio_dir=audio_dir,
        safe_to_test=bool(bench.get("safe_to_test", mock)),
        can_log_path=bench.get("can_log"),
        cockpit_log_path=bench.get("cockpit_log") or bench.get("cabin_log") or bench.get("device_log"),
        ui_before_path=bench.get("ui_before"),
        ui_after_path=bench.get("ui_after"),
    )


def _agent_executor(workspace: str, rules_path: Optional[str]) -> tuple[AgentStore, SafetyPolicy, AgentExecutor]:
    store = AgentStore(workspace)
    policy = SafetyPolicy.from_rules(load_rules(rules_path))
    return store, policy, AgentExecutor(store, policy)


@agent.command("execute")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=True, help="Agent run ID")
@click.option("--candidate", "candidate_id", required=True, help="Candidate ID")
@click.option("--rules", "rules_path", required=False, type=click.Path(exists=True, dir_okay=False), help="YAML or JSON test rules")
@click.option("--audio-dir", default="./test_run/audio", show_default=True, help="Input/generated audio directory")
@click.option("--bench", "bench_path", required=False, type=click.Path(exists=True, dir_okay=False), help="Bench YAML with safe_to_test and capture paths")
@click.option("--mock", is_flag=True, help="Use the deterministic built-in benchmark path")
@click.option("--online", is_flag=True, help="Use cloud TTS/ASR and configured local capture")
@click.option("--repetitions", default=1, type=click.IntRange(1, 3), show_default=True)
def agent_execute(workspace: str, run_id: str, candidate_id: str, rules_path: Optional[str], audio_dir: str, bench_path: Optional[str], mock: bool, online: bool, repetitions: int):
    """Run one safe candidate and persist evidence plus a verdict."""
    store, _, executor = _agent_executor(workspace, rules_path)
    candidate = store.get_candidate(candidate_id)
    adapter = _build_agent_adapter(workspace, mock, online, audio_dir, bench_path)
    summary = executor.execute_candidate(run_id, candidate, adapter, repetitions=repetitions)
    click.echo(json.dumps(summary.to_dict(), ensure_ascii=False))


@agent.command("evolve")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=True, help="Agent run ID")
@click.option("--rules", "rules_path", required=False, type=click.Path(exists=True, dir_okay=False), help="YAML or JSON test rules")
@click.option("--audio-dir", default="./test_run/audio", show_default=True, help="Input/generated audio directory")
@click.option("--bench", "bench_path", required=False, type=click.Path(exists=True, dir_okay=False), help="Bench YAML with safe_to_test and capture paths")
@click.option("--mock", is_flag=True, help="Use the deterministic built-in benchmark path")
@click.option("--online", is_flag=True, help="Use cloud TTS/ASR and configured local capture")
@click.option("--max-iterations", default=6, type=click.IntRange(1, 30), show_default=True)
def agent_evolve(workspace: str, run_id: str, rules_path: Optional[str], audio_dir: str, bench_path: Optional[str], mock: bool, online: bool, max_iterations: int):
    """Run controlled, non-promoting exploration candidates."""
    store, policy, executor = _agent_executor(workspace, rules_path)
    adapter = _build_agent_adapter(workspace, mock, online, audio_dir, bench_path)
    exploration_config = load_rules(rules_path).get("exploration", {})
    outcomes = EvolutionEngine(store, policy, executor, exploration_config=exploration_config).evolve(
        run_id, adapter, max_iterations=max_iterations
    )
    click.echo(json.dumps({"run_id": run_id, "outcomes": outcomes}, ensure_ascii=False))


def _serve_review(workspace: str, run_id: str, host: str, port: int) -> None:
    ReviewService(AgentStore(workspace), run_id).serve(host, port)


@agent.command("review")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=True, help="Agent run ID")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=click.IntRange(1024, 65535), show_default=True)
def agent_review(workspace: str, run_id: str, host: str, port: int):
    """Start the local-only review console."""
    _serve_review(workspace, run_id, host, port)


@main.command("review")
@click.option("--workspace", default="./carvoice_agent", show_default=True, help="Agent workspace")
@click.option("--run-id", required=True, help="Agent run ID")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, type=click.IntRange(1024, 65535), show_default=True)
def review(workspace: str, run_id: str, host: str, port: int):
    """Alias for `agent review`."""
    _serve_review(workspace, run_id, host, port)


def _load_yaml(path: str) -> dict:
    plan_path = Path(path)
    if plan_path.suffix.lower() == ".json":
        with open(plan_path, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        import yaml
    except ImportError as exc:
        raise click.ClickException(
            "PyYAML 未安装，无法读取 YAML 测试计划。请运行 `pip install pyyaml`，"
            "或使用 `python examples/mock_demo.py` 验证 mock demo。"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(path: str, data: dict):
    try:
        import yaml
    except ImportError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def _env_bool(name: str) -> bool:
    return str(first_env(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float = 0.0) -> float:
    value = first_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _print_summary(report_data: dict, report_paths: dict):
    """打印评测摘要"""
    click.echo("\n" + "=" * 60)
    click.echo("CarVoice Bench 评测摘要")
    click.echo("=" * 60)

    for key, path in report_paths.items():
        click.echo(f"  {key.upper()}: {path}")

    results = report_data.get("summary", {})
    total = results.get("total_cases", 0)
    passed = results.get("passed", 0)
    failed = results.get("failed", 0)

    click.echo(f"\n  总用例: {total} | 通过: {passed} | 失败: {failed}")

    if "asr" in report_data:
        asr = report_data["asr"]
        click.echo(f"  ASR 平均延迟: {_fmt_number(asr.get('avg_latency_ms'), 1)}ms")
        click.echo(f"  ASR 平均 WER: {_fmt_percent(asr.get('avg_wer'), 2)}")

    if "tts" in report_data:
        tts = report_data["tts"]
        click.echo(f"  TTS 平均 MOS: {_fmt_number(tts.get('avg_mos'), 2)}")

    if "can" in report_data:
        can = report_data["can"]
        click.echo(f"  CAN 信号匹配率: {_fmt_percent(can.get('match_rate'), 1)}")

    click.echo("=" * 60)


def _fmt_number(value, digits: int = 1) -> str:
    if isinstance(value, bool) or value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "N/A"


def _fmt_percent(value, digits: int = 1) -> str:
    if isinstance(value, bool) or value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}%}"
    return "N/A"


if __name__ == "__main__":
    main()
