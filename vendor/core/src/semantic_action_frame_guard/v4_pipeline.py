from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from semantic_action_frame_guard.llm import LLMTransport, StageFailure, TraceRecorder
from semantic_action_frame_guard.v3_models import TaskFrameSet
from semantic_action_frame_guard.v3_pipeline import V3StructuredStage
from semantic_action_frame_guard.v3_prompts import TASK_FRAME_SYSTEM, task_frame_request
from semantic_action_frame_guard.v4_models import (
    AuthorizationDecision,
    ExecutionRecord,
    ExecutionStatus,
    GuardInput,
    RuntimeAuthorization,
)
from semantic_action_frame_guard.v4_prompts import (
    RUNTIME_AUTHORIZATION_SYSTEM,
    runtime_authorization_request,
)

_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_LOOSE_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*|\d+")
_INDEX_ALIAS = re.compile(r"^(trusted_task_frames|execution_tool_history)\[(\d+)\](.*)$")
_LABELED_SCALAR = re.compile(r"""^\s*["']?([A-Za-z_][A-Za-z0-9_-]*)["']?\s*[:=]\s*(.*?)\s*$""")


@dataclass(frozen=True)
class _ReferenceCatalog:
    task_references: frozenset[str]
    history_references: frozenset[str]
    task_scalars: dict[str, tuple[str, ...]]
    history_scalars: dict[str, tuple[str, ...]]
    task_index_ids: tuple[str, ...]
    history_index_ids: tuple[str, ...]


@dataclass(frozen=True)
class _AuthorizationCacheEntry:
    result: RuntimeAuthorization
    origin_tool_call_id: str


def _tokens(reference: str) -> list[str | int]:
    tokens: list[str | int] = []
    position = 0
    for match in _PATH_TOKEN.finditer(reference):
        if match.start() != position and reference[position : match.start()] != ".":
            raise ValueError(f"invalid reference syntax: {reference}")
        tokens.append(int(match.group(2)) if match.group(2) is not None else match.group(1))
        position = match.end()
    if not tokens or position != len(reference):
        raise ValueError(f"invalid reference syntax: {reference}")
    return tokens


def _resolve(value: Any, path: list[str | int], reference: str) -> Any:
    current = value
    for token in path:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise ValueError(f"reference does not exist: {reference}")
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                raise ValueError(f"reference does not exist: {reference}")
            current = current[token]
    return current


def _reference_tree(
    value: Any,
    prefix: str,
) -> tuple[set[str], dict[str, set[str]]]:
    references = {prefix}
    scalars: dict[str, set[str]] = {}

    def visit(current: Any, path: str) -> None:
        references.add(path)
        if isinstance(current, dict):
            for key, item in current.items():
                visit(item, f"{path}.{key}")
            return
        if isinstance(current, list):
            for index, item in enumerate(current):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(current, str):
            scalars.setdefault(current, set()).add(path)
        elif current is not None:
            scalars.setdefault(json.dumps(current, ensure_ascii=False), set()).add(path)

    visit(value, prefix)
    return references, scalars


def _merge_scalars(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for value, paths in source.items():
        target.setdefault(value, set()).update(paths)


def _reference_catalog(
    task_frames: TaskFrameSet,
    history: list[ExecutionRecord],
) -> _ReferenceCatalog:
    task_references: set[str] = set()
    history_references: set[str] = set()
    task_scalars: dict[str, set[str]] = {}
    history_scalars: dict[str, set[str]] = {}

    for frame in task_frames.root:
        references, scalars = _reference_tree(frame.model_dump(mode="json"), frame.frame_id)
        task_references.update(references)
        _merge_scalars(task_scalars, scalars)

    for record in history:
        if record.status != ExecutionStatus.SUCCESS:
            continue
        references, scalars = _reference_tree(record.model_dump(mode="json"), record.call_id)
        history_references.update(references)
        _merge_scalars(history_scalars, scalars)

    return _ReferenceCatalog(
        task_references=frozenset(task_references),
        history_references=frozenset(history_references),
        task_scalars={value: tuple(sorted(paths)) for value, paths in task_scalars.items()},
        history_scalars={value: tuple(sorted(paths)) for value, paths in history_scalars.items()},
        task_index_ids=tuple(frame.frame_id for frame in task_frames.root),
        history_index_ids=tuple(record.call_id for record in history),
    )


def _strip_wrapping_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1]
    return stripped


def _expand_index_alias(reference: str, catalog: _ReferenceCatalog) -> str:
    match = _INDEX_ALIAS.match(reference.strip())
    if match is None:
        return reference.strip()
    collection, raw_index, suffix = match.groups()
    index = int(raw_index)
    ids = catalog.task_index_ids if collection == "trusted_task_frames" else catalog.history_index_ids
    if index >= len(ids):
        return reference.strip()
    return ids[index] + suffix


def _loose_tokens(value: str) -> tuple[str, ...]:
    return tuple(_LOOSE_TOKEN.findall(value))


def _canonical_reference(
    reference: str,
    allowed_references: frozenset[str],
    scalar_paths: dict[str, tuple[str, ...]],
    catalog: _ReferenceCatalog,
) -> str | None:
    candidate = _expand_index_alias(reference, catalog)
    if candidate in allowed_references:
        return candidate

    candidate_tokens = _loose_tokens(candidate)
    prefix_matches: list[tuple[int, str]] = []
    for allowed in allowed_references:
        allowed_tokens = tuple(str(token) for token in _tokens(allowed))
        if len(candidate_tokens) >= len(allowed_tokens) and candidate_tokens[: len(allowed_tokens)] == allowed_tokens:
            prefix_matches.append((len(allowed_tokens), allowed))
    if prefix_matches:
        max_length = max(length for length, _ in prefix_matches)
        longest = {allowed for length, allowed in prefix_matches if length == max_length}
        if len(longest) == 1:
            if len(candidate_tokens) > max_length:
                token_matches = list(_LOOSE_TOKEN.finditer(candidate))
                separator = candidate[
                    token_matches[max_length - 1].end() : token_matches[max_length].start()
                ]
                if "." in separator or "[" in separator:
                    return None
                if max_length == 1 and ":" in separator:
                    remainder = candidate[token_matches[max_length].start() :]
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_-]*\s*(?:[.=]|—|–|$)", remainder):
                        return None
            return longest.pop()

    scalar_candidate = _strip_wrapping_quotes(candidate)
    scalar_matches = set(scalar_paths.get(scalar_candidate, ()))
    labeled = _LABELED_SCALAR.match(candidate)
    if labeled is not None:
        label, raw_value = labeled.groups()
        value = _strip_wrapping_quotes(raw_value.rstrip(","))
        scalar_matches.update(path for path in scalar_paths.get(value, ()) if _tokens(path)[-1] == label)
    scalar_matches.intersection_update(allowed_references)
    if len(scalar_matches) == 1:
        return scalar_matches.pop()
    return None


def _combined_scalars(catalog: _ReferenceCatalog) -> dict[str, tuple[str, ...]]:
    combined: dict[str, set[str]] = {}
    for mapping in (catalog.task_scalars, catalog.history_scalars):
        for value, paths in mapping.items():
            combined.setdefault(value, set()).update(paths)
    return {value: tuple(sorted(paths)) for value, paths in combined.items()}


def normalize_runtime_authorization_references(
    result: RuntimeAuthorization,
    task_frames: TaskFrameSet,
    history: list[ExecutionRecord],
) -> tuple[RuntimeAuthorization, list[dict[str, str]], list[dict[str, str]]]:
    catalog = _reference_catalog(task_frames, history)
    rewrites: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []

    def normalize(
        reference: str,
        *,
        location: str,
        allowed: frozenset[str],
        scalars: dict[str, tuple[str, ...]],
    ) -> str:
        canonical = _canonical_reference(reference, allowed, scalars, catalog)
        if canonical is None:
            unresolved.append({"location": location, "original": reference})
            return reference
        if canonical != reference:
            rewrites.append(
                {
                    "location": location,
                    "original": reference,
                    "canonical": canonical,
                }
            )
        return canonical

    action_support = []
    for support_index, support in enumerate(result.action_support):
        root = normalize(
            support.root,
            location=f"action_support[{support_index}].root",
            allowed=catalog.task_references,
            scalars=catalog.task_scalars,
        )
        via = [
            normalize(
                reference,
                location=f"action_support[{support_index}].via[{via_index}]",
                allowed=catalog.history_references,
                scalars=catalog.history_scalars,
            )
            for via_index, reference in enumerate(support.via)
        ]
        action_support.append(support.model_copy(update={"root": root, "via": via}))

    source_references = catalog.task_references | catalog.history_references
    source_scalars = _combined_scalars(catalog)
    parameter_grounding = {}
    for argument_name, grounding in result.parameter_grounding.items():
        sources = [
            normalize(
                reference,
                location=(f"parameter_grounding.{argument_name}.sources[{source_index}]"),
                allowed=source_references,
                scalars=source_scalars,
            )
            for source_index, reference in enumerate(grounding.sources)
        ]
        parameter_grounding[argument_name] = grounding.model_copy(update={"sources": sources})

    return (
        result.model_copy(
            update={
                "action_support": action_support,
                "parameter_grounding": parameter_grounding,
            }
        ),
        rewrites,
        unresolved,
    )


def _reference_maps(
    task_frames: TaskFrameSet,
    history: list[ExecutionRecord],
) -> tuple[dict[str, dict[str, Any]], dict[str, ExecutionRecord]]:
    tasks = {frame.frame_id: frame.model_dump(mode="json") for frame in task_frames.root}
    calls = {record.call_id: record for record in history}
    return tasks, calls


def _validate_reference(
    reference: str,
    task_frames: TaskFrameSet,
    history: list[ExecutionRecord],
    *,
    task_only: bool = False,
    history_only: bool = False,
) -> None:
    tokens = _tokens(reference)
    head = tokens[0]
    if not isinstance(head, str):
        raise TypeError(f"reference must begin with a frame or call ID: {reference}")
    tasks, calls = _reference_maps(task_frames, history)
    if head in tasks and not history_only:
        _resolve(tasks[head], tokens[1:], reference)
        return
    if not task_only and head in calls:
        record = calls[head]
        if record.status != ExecutionStatus.SUCCESS:
            raise ValueError(f"failed tool results cannot ground authorization: {reference}")
        _resolve(record.model_dump(mode="json"), tokens[1:], reference)
        return
    raise ValueError(f"reference does not exist: {reference}")


def validate_runtime_authorization(
    result: RuntimeAuthorization,
    guard_input: GuardInput,
    task_frames: TaskFrameSet,
) -> None:
    history = guard_input.execution_tool_history
    argument_names = set(guard_input.proposed_tool_call.arguments)
    unknown_arguments = set(result.parameter_grounding) - argument_names
    if unknown_arguments:
        raise ValueError("parameter_grounding contains unknown arguments: " + ", ".join(sorted(unknown_arguments)))
    for support in result.action_support:
        _validate_reference(support.root, task_frames, history, task_only=True)
        for reference in support.via:
            _validate_reference(reference, task_frames, history, history_only=True)
    for grounding in result.parameter_grounding.values():
        for reference in grounding.sources:
            _validate_reference(reference, task_frames, history)
    if result.decision == AuthorizationDecision.ALLOW:
        if not result.action_support:
            raise ValueError("allow requires non-empty action_support")
        if any(not item.sources for item in result.parameter_grounding.values()):
            raise ValueError("allow cannot contain an ungrounded parameter")


def _validation_error_kind(exc: ValidationError | ValueError) -> str:
    if isinstance(exc, ValidationError):
        if any(error.get("type") == "json_invalid" for error in exc.errors()):
            return "json_format"
        return "schema_format"
    message = str(exc)
    if (
        "reference does not exist" in message
        or "invalid reference syntax" in message
        or "reference must begin" in message
        or "failed tool results cannot ground" in message
    ):
        return "reference_format"
    return "authorization_structure"


def _masked_authorization(result: RuntimeAuthorization) -> dict[str, Any]:
    return {
        "action_support": [
            {
                "root": "<reference>",
                "via": ["<reference>"] * len(item.via),
                "relation": item.relation,
            }
            for item in result.action_support
        ],
        "parameter_grounding": {
            key: {
                "sources": ["<reference>"] * len(value.sources),
                "derivation": value.derivation,
            }
            for key, value in result.parameter_grounding.items()
        },
        "decision": result.decision.value,
    }


def _allowed_reference_payload(catalog: _ReferenceCatalog) -> dict[str, list[str]]:
    return {
        "root": sorted(catalog.task_references),
        "via": sorted(catalog.history_references),
        "sources": sorted(catalog.task_references | catalog.history_references),
    }


class V4RuntimeAuthorizationStage:
    def __init__(
        self,
        system_prompt: str,
        transport: LLMTransport,
        recorder: TraceRecorder,
    ) -> None:
        self.namespace = "runtime_action_authorization"
        self.system_prompt = system_prompt
        self.transport = transport
        self.recorder = recorder
        self.cache_namespace = "v4_runtime_action_authorization_2"
        self.cache: dict[str, _AuthorizationCacheEntry] = {}

    def run(
        self,
        payload: dict[str, Any],
        tool_call_id: str,
        guard_input: GuardInput,
        task_frames: TaskFrameSet,
    ) -> RuntimeAuthorization:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        cache_material = f"{self.cache_namespace}\n{self.system_prompt}\n{serialized}"
        cache_key = hashlib.sha256(cache_material.encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            validate_runtime_authorization(cached.result, guard_input, task_frames)
            self.recorder.record(
                {
                    "stage": self.namespace,
                    "tool_call_id": tool_call_id,
                    "cache_namespace": self.cache_namespace,
                    "cache_key": cache_key,
                    "cache_hit": True,
                    "origin_tool_call_id": cached.origin_tool_call_id,
                    "origin_trace_file": f"{cached.origin_tool_call_id}.jsonl",
                    "structured_output": cached.result.model_dump(mode="json"),
                    "reference_rewrites": [],
                    "local_repair_succeeded": False,
                    "model_retry_required": False,
                }
            )
            return cached.result

        catalog = _reference_catalog(task_frames, guard_input.execution_tool_history)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": serialized},
        ]
        initial_parsed: RuntimeAuthorization | None = None
        first_error_kind = ""
        first_error = ""
        last_error = ""

        for attempt in (1, 2):
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
                        "model_retry_required": False,
                    }
                )
                raise StageFailure(f"{self.namespace} transport request failed: {type(exc).__name__}") from exc

            elapsed = time.perf_counter() - started
            tokens = {
                "input": response.input_tokens,
                "output": response.output_tokens,
                "total": response.input_tokens + response.output_tokens,
            }
            event: dict[str, Any] = {
                "stage": self.namespace,
                "tool_call_id": tool_call_id,
                "cache_namespace": self.cache_namespace,
                "cache_key": cache_key,
                "cache_hit": False,
                "attempt": attempt,
                "prompt": messages,
                "raw_response": response.raw,
                "elapsed_seconds": elapsed,
                "tokens": tokens,
            }
            if attempt == 2:
                event["retry_tokens"] = tokens
                event["retry_elapsed_seconds"] = elapsed

            parsed: RuntimeAuthorization | None = None
            rewrites: list[dict[str, str]] = []
            unresolved: list[dict[str, str]] = []
            try:
                parsed = RuntimeAuthorization.model_validate_json(response.raw)
                if (
                    attempt == 2
                    and first_error_kind == "reference_format"
                    and initial_parsed is not None
                    and _masked_authorization(parsed) != _masked_authorization(initial_parsed)
                ):
                    raise ValueError("reference repair changed non-reference authorization fields")
                normalized, rewrites, unresolved = normalize_runtime_authorization_references(
                    parsed,
                    task_frames,
                    guard_input.execution_tool_history,
                )
                if unresolved:
                    raise ValueError("reference does not exist: " + "; ".join(item["original"] for item in unresolved))
                validate_runtime_authorization(normalized, guard_input, task_frames)
            except (ValidationError, ValueError) as exc:
                error_kind = _validation_error_kind(exc)
                last_error = str(exc)
                if attempt == 1:
                    initial_parsed = parsed
                    first_error_kind = error_kind
                    first_error = last_error
                self.recorder.record(
                    {
                        **event,
                        "validation_error_kind": (first_error_kind if attempt == 2 else error_kind),
                        "repair_failure_kind": (error_kind if attempt == 2 else None),
                        "validation_error": last_error,
                        "reference_rewrites": rewrites,
                        "unresolved_references": unresolved,
                        "local_repair_succeeded": False,
                        "model_retry_required": attempt == 1,
                        "fail_closed": attempt == 2,
                    }
                )
                if attempt == 2:
                    break
                repair_payload = {
                    "repair_once": True,
                    "repair_scope": "format_only",
                    "invalid_output": (parsed.model_dump(mode="json") if parsed is not None else response.raw),
                    "validation_error_kind": first_error_kind,
                    "validation_error": first_error,
                    "allowed_references": _allowed_reference_payload(catalog),
                    "requirements": [
                        "Return RuntimeAuthorization JSON only.",
                        "Correct only JSON, schema, or reference formatting errors.",
                        "For reference_format, replace only root, via, or sources strings and preserve every other field and list length exactly.",
                    ],
                }
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(repair_payload, ensure_ascii=False),
                    },
                ]
                continue

            assert parsed is not None
            self.recorder.record(
                {
                    **event,
                    "structured_output": normalized.model_dump(mode="json"),
                    "reference_rewrites": rewrites,
                    "unresolved_references": [],
                    "local_repair_succeeded": bool(rewrites),
                    "model_retry_required": False,
                }
            )
            self.cache[cache_key] = _AuthorizationCacheEntry(normalized, tool_call_id)
            return normalized

        raise StageFailure(f"{self.namespace} failed validation after one format repair: {last_error}")


class V4SemanticPipeline:
    def __init__(self, transport: LLMTransport, recorder: TraceRecorder | None = None) -> None:
        self.recorder = recorder or TraceRecorder()
        self.task_frame = V3StructuredStage(
            "task_frame",
            TASK_FRAME_SYSTEM,
            TaskFrameSet,
            transport,
            self.recorder,
            cache_namespace="v4_task_frame_1",
            cache_enabled=True,
        )
        self.runtime_authorization = V4RuntimeAuthorizationStage(
            RUNTIME_AUTHORIZATION_SYSTEM,
            transport,
            self.recorder,
        )

    def analyze(self, guard_input: GuardInput) -> tuple[TaskFrameSet, RuntimeAuthorization]:
        task_frames = self.task_frame.run(
            task_frame_request(guard_input.trusted_user_task),
            guard_input.proposed_tool_call.tool_call_id,
        )
        authorization = self.runtime_authorization.run(
            runtime_authorization_request(
                task_frames,
                guard_input.proposed_tool_call,
                guard_input.execution_tool_history,
            ),
            guard_input.proposed_tool_call.tool_call_id,
            guard_input,
            task_frames,
        )
        return task_frames, authorization
