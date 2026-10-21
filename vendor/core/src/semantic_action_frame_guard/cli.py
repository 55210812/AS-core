from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from semantic_action_frame_guard.guard import SemanticFrameAlignmentGuard
from semantic_action_frame_guard.llm import OpenAITransport, SemanticPipeline, TraceRecorder
from semantic_action_frame_guard.models import GuardInput
from semantic_action_frame_guard.prompts import task_frame_request
from semantic_action_frame_guard.v3_models import CompletedCallSummarySubmission
from semantic_action_frame_guard.v3_prompts import (
    ACTION_FRAME_SYSTEM as V3_ACTION_FRAME_SYSTEM,
    ALIGNMENT_SYSTEM as V3_ALIGNMENT_SYSTEM,
    TASK_AGENT_TRACE_CONTRACT,
    TASK_FRAME_SYSTEM as V3_TASK_FRAME_SYSTEM,
    TRACE_SUMMARY_SYSTEM as V3_TRACE_SUMMARY_SYSTEM,
)


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _guard(trace_dir: str | None) -> SemanticFrameAlignmentGuard:
    transport = OpenAITransport(
        os.getenv("SAFG_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
        api_key=os.getenv("SAFG_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("SAFG_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        reasoning_effort=os.getenv("SAFG_REASONING_EFFORT"),
    )
    return SemanticFrameAlignmentGuard(SemanticPipeline(transport, TraceRecorder(trace_dir)))


def cmd_print_prompts(_: argparse.Namespace) -> None:
    prompts = {
        "task_agent_trace_contract": TASK_AGENT_TRACE_CONTRACT,
        "task_frame": V3_TASK_FRAME_SYSTEM,
        "trace_summary": V3_TRACE_SUMMARY_SYSTEM,
        "action_frame": V3_ACTION_FRAME_SYSTEM,
        "alignment": V3_ALIGNMENT_SYSTEM,
        "submit_completed_call_summaries_schema": (
            CompletedCallSummarySubmission.model_json_schema()
        ),
    }
    _print(prompts)


def cmd_inspect_task(args: argparse.Namespace) -> None:
    data = _load(args.input)
    task_id = data.get("task_id", "inspect-task")
    user_task = data["user_task"]
    guard = _guard(args.trace_dir)
    frames = guard.semantic_pipeline.task_frame.run(
        task_frame_request(task_id, user_task), data.get("tool_call_id", "inspect-task")
    )
    _print(frames.model_dump(mode="json"))


def cmd_inspect_call(args: argparse.Namespace) -> None:
    guard_input = GuardInput.model_validate(_load(args.input))
    result = _guard(args.trace_dir).evaluate(guard_input)
    _print(result.model_dump(mode="json"))


def _run_cases(args: argparse.Namespace, *, one: bool) -> None:
    data = _load(args.input)
    cases = [data] if one else data
    guard = _guard(args.trace_dir)
    output = []
    for case in cases:
        guard_input = GuardInput.model_validate(case.get("guard_input", case))
        result = guard.evaluate(guard_input)
        output.append(
            {
                "case_id": case.get("case_id", guard_input.tool_call_id),
                "result": result.model_dump(mode="json"),
            }
        )
    if getattr(args, "output", None):
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safg")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("print-prompts").set_defaults(func=cmd_print_prompts)

    inspect_task = sub.add_parser("inspect-task")
    inspect_task.add_argument("input")
    inspect_task.add_argument("--trace-dir", default="results/traces")
    inspect_task.set_defaults(func=cmd_inspect_task)

    inspect_call = sub.add_parser("inspect-call")
    inspect_call.add_argument("input")
    inspect_call.add_argument("--trace-dir", default="results/traces")
    inspect_call.set_defaults(func=cmd_inspect_call)

    replay = sub.add_parser("replay-case")
    replay.add_argument("input")
    replay.add_argument("--trace-dir", default="results/traces")
    replay.set_defaults(func=lambda args: _run_cases(args, one=True))

    run_cases = sub.add_parser("run-cases")
    run_cases.add_argument("input")
    run_cases.add_argument("--trace-dir", default="results/traces")
    run_cases.add_argument("--output")
    run_cases.set_defaults(func=lambda args: _run_cases(args, one=False))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
