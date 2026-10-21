from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

from semantic_action_frame_guard.llm import LLMTransport, StageFailure, TraceRecorder
from semantic_action_frame_guard.v3_models import (
    AlignmentJudgment,
    CompactExecutionTrace,
    GuardInput,
    SemanticActionFrame,
    TaskFrameSet,
    relevant_prior_results,
)
from semantic_action_frame_guard.v3_prompts import (
    ACTION_FRAME_SYSTEM,
    ALIGNMENT_SYSTEM,
    TASK_FRAME_SYSTEM,
    TRACE_SUMMARY_SYSTEM,
    action_frame_request,
    alignment_request,
    task_frame_request,
    trace_summary_request,
)

T = TypeVar("T", bound=BaseModel)
JsonResponseNormalizer = Callable[[str], tuple[str, bool, str | None]]


def normalize_strict_json_code_fence(raw: str) -> tuple[str, bool, str | None]:
    """Remove only a complete outer `````json`` code fence."""

    stripped = raw.strip()
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0] == "```json" and lines[-1] == "```":
        return "\n".join(lines[1:-1]).strip(), True, "markdown_json_fence_removed"
    return raw, False, None


@dataclass(frozen=True)
class CacheEntry:
    result: BaseModel
    origin_tool_call_id: str


class V3StructuredStage:
    def __init__(
        self,
        namespace: str,
        system_prompt: str,
        output_model: type[T],
        transport: LLMTransport,
        recorder: TraceRecorder,
        *,
        cache_namespace: str,
        cache_enabled: bool = False,
        response_normalizer: JsonResponseNormalizer | None = None,
    ) -> None:
        self.namespace = namespace
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.transport = transport
        self.recorder = recorder
        self.cache_namespace = cache_namespace
        self.cache: dict[str, CacheEntry] = {}
        self.cache_enabled = cache_enabled
        self.response_normalizer = response_normalizer

    def run(
        self,
        payload: dict[str, Any],
        tool_call_id: str,
        validator: Callable[[T], None] | None = None,
    ) -> T:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        cache_material = f"{self.cache_namespace}\n{self.system_prompt}\n{serialized}"
        cache_key = hashlib.sha256(cache_material.encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if self.cache_enabled and cached is not None:
            result = cached.result
            assert isinstance(result, self.output_model)
            if validator is not None:
                validator(result)
            self.recorder.record(
                {
                    "stage": self.namespace,
                    "tool_call_id": tool_call_id,
                    "cache_namespace": self.cache_namespace,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "origin_tool_call_id": cached.origin_tool_call_id,
                    "origin_trace_file": f"{cached.origin_tool_call_id}.jsonl",
                    "structured_output": result.model_dump(mode="json"),
                }
            )
            return result

        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": serialized},
        ]
        first_raw = ""
        last_error = ""
        for attempt in (1, 2):
            messages = base_messages
            if attempt == 2:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "repair_once": True,
                                "original_request": payload,
                                "invalid_json": first_raw,
                                "validation_error": last_error,
                                "required_schema": self.output_model.model_json_schema(),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
            started = time.perf_counter()
            try:
                response = self.transport.complete(self.namespace, messages)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.recorder.record(
                    {
                        "stage": self.namespace,
                        "tool_call_id": tool_call_id,
                        "cache_namespace": self.cache_namespace,
                        "cache_key": cache_key,
                        "cache_hit": False,
                        "attempt": attempt,
                        "prompt": messages,
                        "raw_response": None,
                        "elapsed_seconds": elapsed,
                        "tokens": {"input": 0, "output": 0, "total": 0},
                        "transport_error": f"{type(exc).__name__}: {exc}",
                        "fail_closed": True,
                    }
                )
                raise StageFailure(
                    f"{self.namespace} transport request failed: {type(exc).__name__}"
                ) from exc
            elapsed = time.perf_counter() - started
            if attempt == 1:
                first_raw = response.raw
            event = {
                "stage": self.namespace,
                "tool_call_id": tool_call_id,
                "cache_namespace": self.cache_namespace,
                "cache_key": cache_key,
                "cache_hit": False,
                "attempt": attempt,
                "prompt": messages,
                "raw_response": response.raw,
                "reasoning_content": response.reasoning_content,
                "usage": response.usage,
                "provider_response": response.response,
                "elapsed_seconds": elapsed,
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                    "total": response.input_tokens + response.output_tokens,
                },
            }
            normalized_response = response.raw
            normalization_applied = False
            local_format_normalization = None
            if self.response_normalizer is not None:
                (
                    normalized_response,
                    normalization_applied,
                    local_format_normalization,
                ) = self.response_normalizer(response.raw)
            event.update(
                {
                    "normalized_response": normalized_response,
                    "normalization_applied": normalization_applied,
                    "local_format_normalization": local_format_normalization,
                }
            )
            try:
                parsed = self.output_model.model_validate_json(normalized_response)
                if validator is not None:
                    validator(parsed)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                self.recorder.record(
                    {**event, "validation_error": last_error, "fail_closed": attempt == 2}
                )
                continue
            self.recorder.record(
                {**event, "structured_output": parsed.model_dump(mode="json")}
            )
            if self.cache_enabled:
                self.cache[cache_key] = CacheEntry(parsed, tool_call_id)
            return parsed
        raise StageFailure(
            f"{self.namespace} failed validation after one repair: {last_error}"
        )


def validate_trace(
    result: CompactExecutionTrace,
    guard_input: GuardInput,
) -> None:
    entries = result.compact_execution_trace
    summaries = guard_input.completed_call_summaries
    if len(entries) != len(summaries):
        raise ValueError("compact trace must contain one entry per completed call summary")
    for entry, summary in zip(entries, summaries, strict=True):
        if not entry.startswith(summary.call_id):
            raise ValueError("each compact trace entry must begin with the exact call_id")


def validate_action_frame(frame: SemanticActionFrame, guard_input: GuardInput) -> None:
    if frame.frame_id != guard_input.tool_call_id:
        raise ValueError("action frame must copy the exact tool_call_id")
    expected_keys = {*guard_input.tool.args, "action_evidence", "parameter_evidence"}
    if set(frame.constraints) != expected_keys:
        raise ValueError("action frame constraints must contain exact arguments and two evidence fields")
    for key, value in guard_input.tool.args.items():
        if frame.constraints[key] != value:
            raise ValueError(f"action frame changed tool argument {key}")
    for key in ("action_evidence", "parameter_evidence"):
        if not isinstance(frame.constraints[key], str) or not frame.constraints[key].strip():
            raise ValueError(f"{key} must be a non-empty string")


class V3SemanticPipeline:
    def __init__(self, transport: LLMTransport, recorder: TraceRecorder | None = None) -> None:
        self.recorder = recorder or TraceRecorder()
        self.task_frame = V3StructuredStage(
            "task_frame",
            TASK_FRAME_SYSTEM,
            TaskFrameSet,
            transport,
            self.recorder,
            cache_namespace="v3_general_task_frame_1",
            cache_enabled=True,
        )
        self.trace_summary = V3StructuredStage(
            "trace_summary",
            TRACE_SUMMARY_SYSTEM,
            CompactExecutionTrace,
            transport,
            self.recorder,
            cache_namespace="v3_short_trace_summary_1",
        )
        self.action_frame = V3StructuredStage(
            "action_frame",
            ACTION_FRAME_SYSTEM,
            SemanticActionFrame,
            transport,
            self.recorder,
            cache_namespace="v3_general_action_frame_1",
        )
        self.alignment = V3StructuredStage(
            "alignment",
            ALIGNMENT_SYSTEM,
            AlignmentJudgment,
            transport,
            self.recorder,
            cache_namespace="v3_general_alignment_1",
        )

    def analyze(
        self, guard_input: GuardInput
    ) -> tuple[TaskFrameSet, CompactExecutionTrace, SemanticActionFrame, AlignmentJudgment]:
        frames = self.task_frame.run(
            task_frame_request(guard_input.user_task), guard_input.tool_call_id
        )
        trace = self.trace_summary.run(
            trace_summary_request(guard_input.completed_call_summaries),
            guard_input.tool_call_id,
            lambda result: validate_trace(result, guard_input),
        )
        tool_call = {
            "tool_call_id": guard_input.tool_call_id,
            "tool_name": guard_input.tool.name,
            "tool_description": guard_input.tool.description,
            "tool_json_schema": guard_input.tool.json_schema,
            "arguments": guard_input.tool.args,
        }
        prior_results = relevant_prior_results(
            guard_input.completed_call_summaries, guard_input.tool.args
        )
        action = self.action_frame.run(
            action_frame_request(
                guard_input.user_task, tool_call, trace, prior_results
            ),
            guard_input.tool_call_id,
            lambda result: validate_action_frame(result, guard_input),
        )
        judgment = self.alignment.run(
            alignment_request(frames, action), guard_input.tool_call_id
        )
        return frames, trace, action, judgment
