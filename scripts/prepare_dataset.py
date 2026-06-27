"""为 carvoice-bench 准备外部开源数据集。

当前优先支持 Full-Duplex-Bench v1/v1.5：把包含 ``input.wav`` 和任务标注的样本目录转换成：

```text
<output>/
  audio/<case_id>.wav
  test_plan.yaml
  manifest.json
```

生成的 test_plan 可直接用于 ``carvoice-bench run --mock`` 做结构校验；安装可选音频依赖后，
也可以切换到真实 ASR/TTS 模块进行实测。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


# Full-Duplex-Bench 的不同任务目录命名不完全一致，这里统一映射到本项目的场景名和标注文件。
FULL_DUPLEX_TASKS = {
    "pause_handling": {
        "annotation": "pause.json",
        "event_type": "pause",
        "scenario": "pause_handling",
    },
    "smooth_turn_taking": {
        "annotation": "turn_taking.json",
        "event_type": "turn_taking",
        "scenario": "smooth_turn_taking",
    },
    "user_interruption": {
        "annotation": "interrupt.json",
        "event_type": "user_interrupt",
        "scenario": "user_interruption",
    },
    "user_backchannel": {
        "annotation": "metadata.json",
        "event_type": "backchannel",
        "scenario": "user_backchannel",
    },
    "talking_to_other": {
        "annotation": "metadata.json",
        "event_type": "third_party_overlap",
        "scenario": "talking_to_other",
    },
    "background_speech": {
        "annotation": "metadata.json",
        "event_type": "background_speech",
        "scenario": "background_speech",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare datasets for carvoice-bench.")
    parser.add_argument("--source", required=True, type=Path, help="Source dataset root.")
    parser.add_argument("--output", default=Path("datasets/prepared/full_duplex"), type=Path)
    parser.add_argument("--adapter", choices=["full-duplex"], default="full-duplex")
    parser.add_argument("--task", default="all", help="Task name or 'all'.")
    parser.add_argument("--limit", type=int, default=0, help="Max samples per task; 0 means no limit.")
    parser.add_argument("--copy-audio", action="store_true", help="Copy WAV files into output/audio.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 预留 adapter 参数，后续可继续接入语义理解、TTS MOS 或车载噪声数据集。
    if args.adapter == "full-duplex":
        cases, manifest = prepare_full_duplex(
            source=args.source,
            output_audio_dir=audio_dir,
            task=args.task,
            limit=args.limit,
            copy_audio=args.copy_audio,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported adapter: {args.adapter}")

    plan = {
        "project": {
            "name": "PreparedCarVoiceDataset",
            "source": str(args.source),
            "adapter": args.adapter,
        },
        "test_cases": cases,
    }
    write_yaml(args.output / "test_plan.yaml", plan)
    write_json(args.output / "test_plan.json", plan)
    write_json(args.output / "manifest.json", manifest)

    print(f"Prepared {len(cases)} cases")
    print(f"test_plan: {args.output / 'test_plan.yaml'}")
    print(f"test_plan_json: {args.output / 'test_plan.json'}")
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


def prepare_full_duplex(
    source: Path,
    output_audio_dir: Path,
    task: str,
    limit: int,
    copy_audio: bool,
    overwrite: bool,
) -> tuple[list[dict], list[dict]]:
    """把 Full-Duplex-Bench 样本转换成本项目统一的 test_cases 和 manifest。"""
    tasks = list(FULL_DUPLEX_TASKS) if task == "all" else [task]
    cases = []
    manifest = []

    for task_name in tasks:
        if task_name not in FULL_DUPLEX_TASKS:
            raise ValueError(f"Unknown Full-Duplex task: {task_name}")

        spec = FULL_DUPLEX_TASKS[task_name]
        sample_dirs = list(find_task_sample_dirs(source, task_name, spec["annotation"]))
        if limit > 0:
            sample_dirs = sample_dirs[:limit]

        for sample_dir in sample_dirs:
            case_id = f"fd_{task_name}_{sample_dir.name}"
            input_wav = sample_dir / "input.wav"
            clean_input = sample_dir / "clean_input.wav"
            target_audio = output_audio_dir / f"{case_id}.wav"
            # 默认只记录原始音频路径；显式 --copy-audio 时才复制到 prepared/audio，避免大数据集重复占盘。
            if copy_audio and input_wav.exists():
                if overwrite or not target_audio.exists():
                    shutil.copy2(input_wav, target_audio)

            annotation_path = sample_dir / spec["annotation"]
            annotation = read_json(annotation_path) if annotation_path.exists() else None
            # 标注中的 timestamp 统一转换成 expected_events，供全双工事件召回和延迟指标使用。
            expected_events = extract_events(annotation, spec["event_type"])
            utterance = extract_utterance(sample_dir, annotation, task_name)

            case = {
                "id": case_id,
                "utterance": utterance,
                "expected_asr": utterance,
                "timeout_ms": 3000,
                "audio_path": str(target_audio if copy_audio else input_wav),
                "full_duplex": {
                    "scenario": spec["scenario"],
                    "expected_events": expected_events,
                    "expected_behavior": expected_behavior_for(task_name),
                    "tolerance_ms": 500,
                    "source_annotation": str(annotation_path),
                },
            }

            # 带上下文或第三方语音的全双工样本，同时可作为多轮会话/上下文保持的 smoke 数据。
            if task_name in {"user_interruption", "user_backchannel", "talking_to_other", "background_speech"}:
                case["dialogue"] = dialogue_from_annotation(annotation, task_name)

            cases.append(case)
            manifest.append({
                "case_id": case_id,
                "task": task_name,
                "sample_dir": str(sample_dir),
                "input_wav": str(input_wav),
                "clean_input_wav": str(clean_input) if clean_input.exists() else "",
                "annotation": str(annotation_path),
                "copied_audio": str(target_audio) if copy_audio and input_wav.exists() else "",
            })

    return cases, manifest


def find_task_sample_dirs(source: Path, task_name: str, annotation_name: str) -> Iterable[Path]:
    """按标注文件和 input.wav 反查样本目录，兼容原数据集多层目录结构。"""
    for path in sorted(source.rglob(annotation_name)):
        if ".venv" in path.parts or "node_modules" in path.parts or "__pycache__" in path.parts:
            continue
        parent = path.parent
        if not (parent / "input.wav").exists():
            continue
        normalized = str(parent).lower().replace("-", "_")
        if task_name in normalized or task_name.replace("user_", "") in normalized:
            yield parent


def extract_events(annotation, event_type: str) -> list[dict]:
    """从 Full-Duplex-Bench 标注中抽取事件类型和起止时间。"""
    events = []
    if annotation is None:
        return events

    rows = annotation if isinstance(annotation, list) else [annotation]
    for row in rows:
        timestamp = row.get("timestamp") or row.get("timestamps")
        if not timestamp or len(timestamp) < 2:
            continue
        event_name = row.get("text", event_type)
        if isinstance(event_name, str) and event_name.startswith("[") and event_name.endswith("]"):
            event_name = event_name.strip("[]").lower().replace("-", "_")
        if event_name in {"pause", "turn_taking"}:
            event_type = event_name
        events.append({
            "type": event_type,
            "start_sec": float(timestamp[0]),
            "end_sec": float(timestamp[1]),
        })
    return events


def extract_utterance(sample_dir: Path, annotation, task_name: str) -> str:
    """优先使用 transcription.json，其次从任务标注中提取当前轮文本。"""
    transcription_path = sample_dir / "transcription.json"
    if transcription_path.exists():
        transcript = read_json(transcription_path)
        text = flatten_transcript(transcript)
        if text:
            return text

    rows = annotation if isinstance(annotation, list) else [annotation] if annotation else []
    for row in rows:
        for key in ("current_turn_text", "interrupt", "context"):
            if row.get(key):
                return str(row[key])
    if isinstance(annotation, dict):
        return str(annotation.get("current_turn_text") or annotation.get("context_text") or task_name)
    return task_name


def flatten_transcript(transcript) -> str:
    """把分段转写合并成一句话，跳过 [pause] 这类事件标签。"""
    rows = transcript if isinstance(transcript, list) else transcript.get("segments", []) if isinstance(transcript, dict) else []
    texts = []
    for row in rows:
        text = row.get("text") if isinstance(row, dict) else None
        if text and not str(text).startswith("["):
            texts.append(str(text).strip())
    return " ".join(texts).strip()


def expected_behavior_for(task_name: str) -> dict:
    """为每类全双工任务生成最小可评价的行为期望。"""
    return {
        "pause_handling": {"pause_respected": True},
        "smooth_turn_taking": {"smooth_turn_taking": True},
        "user_interruption": {"barge_in_handled": True},
        "user_backchannel": {"backchannel_ignored_or_acknowledged": True},
        "talking_to_other": {"third_party_suppressed": True},
        "background_speech": {"background_speech_robust": True},
    }.get(task_name, {})


def dialogue_from_annotation(annotation, task_name: str) -> dict:
    """从带上下文的全双工标注中构造一个多轮会话评价样例。"""
    row = annotation[0] if isinstance(annotation, list) and annotation else annotation if isinstance(annotation, dict) else {}
    context = row.get("context") or row.get("context_text") or ""
    current = row.get("interrupt") or row.get("current_turn_text") or ""
    turns = []
    if context:
        turns.append({"role": "user", "text": str(context)})
    if current:
        turns.append({"role": "user", "text": str(current)})
    return {
        "turns": turns,
        "expected_final_state": {
            "scenario": task_name,
            "context_preserved": True,
        },
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, data) -> None:
    """优先使用 PyYAML；未安装时用内置简易 YAML 写出器保证脚本可运行。"""
    try:
        import yaml

        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except ImportError:
        path.write_text(to_simple_yaml(data), encoding="utf-8")


def to_simple_yaml(data, indent: int = 0) -> str:
    """覆盖本项目 test_plan 需要的 dict/list/scalar 子集，不追求完整 YAML 规范。"""
    spaces = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.append(to_simple_yaml(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {format_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{spaces}-")
                lines.append(to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{spaces}- {format_scalar(item)}")
        return "\n".join(lines)
    return f"{spaces}{format_scalar(data)}"


def format_scalar(value) -> str:
    """把 Python 标量转换成可读 YAML 文本。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    escaped = text.replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == "__main__":
    raise SystemExit(main())
