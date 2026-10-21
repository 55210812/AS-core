from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from semantic_action_frame_guard.llm import LLMTransport, StageFailure, TraceRecorder
from semantic_action_frame_guard.v3_models import TaskFrameSet
from semantic_action_frame_guard.v3_pipeline import V3StructuredStage
from semantic_action_frame_guard.v3_prompts import TASK_FRAME_SYSTEM, task_frame_request
from semantic_action_frame_guard.v4_pipeline import (
    _canonical_reference,
    _merge_scalars,
    _reference_tree,
    _ReferenceCatalog,
    _tokens,
    _validation_error_kind,
)
from semantic_action_frame_guard.v5_models import AuthorizationDecision, RuntimeAuthorization
from semantic_action_frame_guard.v6_models import GuardInput
from semantic_action_frame_guard.v6_prompts import (
    RUNTIME_AUTHORIZATION_SYSTEM,
    runtime_authorization_request,
)
from semantic_action_frame_guard.v6_retrieval import (
    build_action_context_candidates,
    build_argument_source_candidates,
)
from semantic_action_frame_guard.v6_reference_selection import (
    ReferenceSelectionMode,
    V6ReferenceSelectionStage,
    build_reference_selection_catalog,
    selection_request,
    selection_system_prompt,
)


@dataclass(frozen=True)
class _AuthorizationCacheEntry:
    result: RuntimeAuthorization
    origin_tool_call_id: str


def _candidate_catalog(
    task_frames: TaskFrameSet,
    argument_candidates: dict[str, list[Any]],
    action_candidates: list[Any] | None = None,
) -> _ReferenceCatalog:
    task_references: set[str] = set()
    task_scalars: dict[str, set[str]] = {}
    history_references: set[str] = set()
    history_scalars: dict[str, set[str]] = {}
    history_index_ids: list[str] = []
    for frame in task_frames.root:
        refs, scalars = _reference_tree(frame.model_dump(mode="json"), frame.frame_id)
        task_references.update(refs)
        _merge_scalars(task_scalars, scalars)
    provenances = []
    for values in argument_candidates.values():
        for candidate in values:
            provenances.extend(candidate.provenance)
    provenances.extend(action_candidates or [])
    for provenance in provenances:
        ref = provenance.ref
        history_references.add(ref)
        head = ref.split(".", 1)[0]
        if head not in history_index_ids:
            history_index_ids.append(head)
    return _ReferenceCatalog(
        task_references=frozenset(task_references),
        history_references=frozenset(history_references),
        task_scalars={key: tuple(sorted(value)) for key, value in task_scalars.items()},
        history_scalars={key: tuple(sorted(value)) for key, value in history_scalars.items()},
        task_index_ids=tuple(frame.frame_id for frame in task_frames.root),
        history_index_ids=tuple(history_index_ids),
    )


def _reference_allowed(
    reference: str,
    allowed: frozenset[str],
    scalars: dict[str, tuple[str, ...]],
    catalog: _ReferenceCatalog,
) -> str | None:
    candidate = reference.strip()
    if candidate in allowed:
        return candidate
    try:
        return _canonical_reference(candidate, allowed, scalars, catalog)
    except (TypeError, ValueError):
        return None


def normalize_runtime_authorization_references(
    result: RuntimeAuthorization,
    task_frames: TaskFrameSet,
    argument_candidates: dict[str, list[Any]],
    action_candidates: list[Any] | None = None,
) -> tuple[RuntimeAuthorization, list[dict[str, str]], list[dict[str, str]]]:
    catalog = _candidate_catalog(task_frames, argument_candidates, action_candidates)
    rewrites: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []

    def normalize(
        reference: str,
        *,
        location: str,
        allowed: frozenset[str],
        scalars: dict[str, tuple[str, ...]],
    ) -> str:
        canonical = _reference_allowed(reference, allowed, scalars, catalog)
        if canonical is None:
            unresolved.append({"location": location, "original": reference})
            return reference
        if canonical != reference:
            rewrites.append({"location": location, "original": reference, "canonical": canonical})
        return canonical

    support_entries = []
    for index, support in enumerate(result.action_support):
        root = normalize(
            support.root,
            location=f"action_support[{index}].root",
            allowed=catalog.task_references,
            scalars=catalog.task_scalars,
        )
        via = [
            normalize(
                reference,
                location=f"action_support[{index}].via[{via_index}]",
                allowed=catalog.history_references,
                scalars=catalog.history_scalars,
            )
            for via_index, reference in enumerate(support.via)
        ]
        support_entries.append(support.model_copy(update={"root": root, "via": via}))

    source_refs = catalog.task_references | catalog.history_references
    source_scalars = dict(catalog.task_scalars)
    for key, paths in catalog.history_scalars.items():
        source_scalars.setdefault(key, ())
        source_scalars[key] = tuple(sorted(set(source_scalars[key]) | set(paths)))
    grounding: dict[str, Any] = {}
    for argument_name, value in result.parameter_grounding.items():
        sources = [
            normalize(
                reference,
                location=f"parameter_grounding.{argument_name}.sources[{source_index}]",
                allowed=source_refs,
                scalars=source_scalars,
            )
            for source_index, reference in enumerate(value.sources)
        ]
        grounding[argument_name] = value.model_copy(update={"sources": sources})
    return (
        result.model_copy(update={"action_support": support_entries, "parameter_grounding": grounding}),
        rewrites,
        unresolved,
    )


def _validate_candidate_reference(
    reference: str,
    argument_candidates: dict[str, list[Any]],
    action_candidates: list[Any] | None = None,
) -> None:
    provenances = []
    for values in argument_candidates.values():
        for candidate in values:
            provenances.extend(candidate.provenance)
    provenances.extend(action_candidates or [])
    if reference in {provenance.ref for provenance in provenances}:
        return
    raise ValueError(f"reference does not exist: {reference}")


def _validate_task_reference(reference: str, task_frames: TaskFrameSet) -> None:
    tokens = _tokens(reference)
    if not isinstance(tokens[0], str):
        raise TypeError(f"reference must begin with a frame ID: {reference}")
    frames = {frame.frame_id: frame.model_dump(mode="json") for frame in task_frames.root}
    if tokens[0] not in frames:
        raise ValueError(f"reference does not exist: {reference}")
    current: Any = frames[tokens[0]]
    for token in tokens[1:]:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise ValueError(f"reference does not exist: {reference}")
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                raise ValueError(f"reference does not exist: {reference}")
            current = current[token]


def validate_runtime_authorization(
    result: RuntimeAuthorization,
    guard_input: GuardInput,
    task_frames: TaskFrameSet,
) -> None:
    argument_candidates = guard_input.argument_source_candidates
    action_candidates = guard_input.action_context_candidates
    argument_names = set(guard_input.proposed_tool_call.arguments)
    unknown = set(result.parameter_grounding) - argument_names
    if unknown:
        raise ValueError("parameter_grounding contains unknown arguments: " + ", ".join(sorted(unknown)))
    for support in result.action_support:
        _validate_task_reference(support.root, task_frames)
        for reference in support.via:
            _validate_candidate_reference(reference, argument_candidates, action_candidates)
    for grounding in result.parameter_grounding.values():
        for reference in grounding.sources:
            try:
                _validate_task_reference(reference, task_frames)
            except (TypeError, ValueError):
                _validate_candidate_reference(reference, argument_candidates, action_candidates)
    if result.decision == AuthorizationDecision.ALLOW:
        if not result.action_support:
            raise ValueError("allow requires non-empty action_support")
        if any(not item.sources for item in result.parameter_grounding.values()):
            raise ValueError("allow cannot contain an ungrounded parameter")


def _masked_authorization(result: RuntimeAuthorization) -> dict[str, Any]:
    return {
        "action_support": [
            {"root": "<reference>", "via": ["<reference>"] * len(item.via), "relation": item.relation}
            for item in result.action_support
        ],
        "parameter_grounding": {
            key: {"sources": ["<reference>"] * len(item.sources), "derivation": item.derivation}
            for key, item in result.parameter_grounding.items()
        },
        "decision": result.decision.value,
    }


def _allowed_reference_payload(
    task_frames: TaskFrameSet,
    argument_candidates: dict[str, list[Any]],
    action_candidates: list[Any] | None = None,
) -> dict[str, list[str]]:
    task_refs: set[str] = set()
    for frame in task_frames.root:
        refs, _ = _reference_tree(frame.model_dump(mode="json"), frame.frame_id)
        task_refs.update(refs)
    history_refs: set[str] = set()
    provenances = []
    for values in argument_candidates.values():
        for candidate in values:
            provenances.extend(candidate.provenance)
    provenances.extend(action_candidates or [])
    for provenance in provenances:
        history_refs.add(provenance.ref)
    return {"root": sorted(task_refs), "via": sorted(history_refs), "sources": sorted(task_refs | history_refs)}


class V6RuntimeAuthorizationStage:
    def __init__(self, system_prompt: str, transport: LLMTransport, recorder: TraceRecorder) -> None:
        self.namespace = "runtime_action_authorization"
        self.system_prompt = system_prompt
        self.transport = transport
        self.recorder = recorder
        self.cache_namespace = "v6_canonical_candidates_runtime_authorization_1"
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
                    "structured_output": cached.result.model_dump(mode="json"),
                    "argument_candidate_count": sum(
                        len(values) for values in guard_input.argument_source_candidates.values()
                    ),
                    "action_context_candidate_count": len(guard_input.action_context_candidates),
                }
            )
            return cached.result

        argument_candidates = guard_input.argument_source_candidates
        action_candidates = guard_input.action_context_candidates
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
                    parsed, task_frames, argument_candidates, action_candidates
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
                        "validation_error_kind": first_error_kind if attempt == 2 else error_kind,
                        "repair_failure_kind": error_kind if attempt == 2 else None,
                        "validation_error": last_error,
                        "reference_rewrites": rewrites,
                        "unresolved_references": unresolved,
                        "model_retry_required": attempt == 1,
                        "fail_closed": attempt == 2,
                    }
                )
                if attempt == 2:
                    break
                repair_payload = {
                    "repair_once": True,
                    "repair_scope": "format_only",
                    "invalid_output": parsed.model_dump(mode="json") if parsed is not None else response.raw,
                    "validation_error_kind": first_error_kind,
                    "validation_error": first_error,
                    "allowed_references": _allowed_reference_payload(
                        task_frames, argument_candidates, action_candidates
                    ),
                    "requirements": [
                        "Return RuntimeAuthorization JSON only.",
                        "Correct only JSON, schema, or reference formatting errors.",
                        "For reference_format, replace only root, via, or sources strings and preserve every other field and list length exactly.",
                    ],
                }
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
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


class V6SemanticPipeline:
    def __init__(
        self,
        transport: LLMTransport,
        recorder: TraceRecorder | None = None,
        *,
        task_frame_cache_path: str | Path | None = None,
        selection_mode: str | ReferenceSelectionMode | None = None,
    ) -> None:
        self.recorder = recorder or TraceRecorder()
        self.selection_mode = (
            ReferenceSelectionMode(selection_mode) if selection_mode is not None else None
        )
        self.task_frame = V3StructuredStage(
            "task_frame",
            TASK_FRAME_SYSTEM,
            TaskFrameSet,
            transport,
            self.recorder,
            cache_namespace="v6_task_frame_1",
            cache_enabled=True,
        )
        if self.selection_mode is None:
            self.runtime_authorization = V6RuntimeAuthorizationStage(
                RUNTIME_AUTHORIZATION_SYSTEM, transport, self.recorder
            )
        else:
            self.runtime_authorization = V6ReferenceSelectionStage(
                selection_system_prompt(self.selection_mode),
                transport,
                self.recorder,
                self.selection_mode,
            )
        self.last_argument_source_candidates: dict[str, list[Any]] = {}
        self.last_action_context_candidates: list[Any] = []
        self.last_retrieval_diagnostics: dict[str, Any] = {}
        self.last_reference_catalog: dict[str, Any] = {}
        self.task_frame_cache: dict[str, Any] = {}
        self.require_task_frame_cache = task_frame_cache_path is not None
        if task_frame_cache_path:
            path = Path(task_frame_cache_path)
            if path.exists():
                self.task_frame_cache = json.loads(path.read_text(encoding="utf-8"))

    def _cached_task_frame(self, trusted_user_task: str, call_id: str) -> TaskFrameSet | None:
        digest = hashlib.sha256(trusted_user_task.encode()).hexdigest()
        entry = self.task_frame_cache.get(digest)
        if entry is None and isinstance(self.task_frame_cache.get("frames_by_task_hash"), dict):
            entry = self.task_frame_cache["frames_by_task_hash"].get(digest)
        if entry is None:
            return None
        if isinstance(entry, dict):
            stored_hash = entry.get("user_task_sha256")
            stored_task = entry.get("trusted_user_task")
            if stored_hash is not None and stored_hash != digest:
                raise ValueError("task frame cache hash does not match trusted task")
            if stored_task is not None and stored_task != trusted_user_task:
                raise ValueError("task frame cache trusted task does not match request")
            expected_case_id = os.getenv("SAFG_TASK_FRAME_CACHE_CASE_ID")
            case_ids = entry.get("case_ids")
            if expected_case_id and isinstance(case_ids, list) and expected_case_id not in case_ids:
                raise ValueError("task frame cache case ID does not match request")
        if isinstance(entry, dict) and "task_frames" in entry:
            entry = entry["task_frames"]
        frames = TaskFrameSet.model_validate(entry)
        self.recorder.record(
            {
                "stage": "task_frame",
                "tool_call_id": call_id,
                "cache_hit": True,
                "cache_source": "deepseek_clean",
                "task_frame_cache_key": digest,
                "structured_output": frames.model_dump(mode="json"),
            }
        )
        return frames

    def analyze(self, guard_input: GuardInput) -> tuple[TaskFrameSet, RuntimeAuthorization]:
        call_id = guard_input.proposed_tool_call.tool_call_id
        task_frames = self._cached_task_frame(guard_input.trusted_user_task, call_id)
        if task_frames is None:
            if self.require_task_frame_cache:
                raise ValueError("task frame cache has no frame for trusted task")
            task_frames = self.task_frame.run(
                task_frame_request(guard_input.trusted_user_task), call_id
            )
        argument_candidates, retrieval_diagnostics = build_argument_source_candidates(
            guard_input.proposed_tool_call.arguments,
            guard_input.execution_tool_history,
            task_frames,
            guard_input.history_metadata,
        )
        action_candidates, action_diagnostics = build_action_context_candidates(
            guard_input.proposed_tool_call.tool_name,
            guard_input.proposed_tool_call.tool_description,
            guard_input.proposed_tool_call.arguments,
            guard_input.execution_tool_history,
            task_frames,
            guard_input.history_metadata,
            guard_input.current_visible_plan,
            argument_candidates,
        )
        prepared_input = guard_input.model_copy(
            update={
                "argument_source_candidates": argument_candidates,
                "action_context_candidates": action_candidates,
            }
        )
        self.last_argument_source_candidates = argument_candidates
        self.last_action_context_candidates = action_candidates
        self.last_retrieval_diagnostics = {
            "argument": retrieval_diagnostics,
            "action": action_diagnostics,
        }
        reference_catalog = None
        if self.selection_mode is not None:
            reference_catalog = build_reference_selection_catalog(
                task_frames,
                argument_candidates,
                action_candidates,
                self.selection_mode,
            )
            self.last_reference_catalog = reference_catalog.model_dump(mode="json")
        self.recorder.record(
            {
                "stage": "v6_source_retrieval",
                "tool_call_id": call_id,
                "argument_candidate_count": sum(
                    len(items) for items in argument_candidates.values()
                ),
                "argument_candidates": {
                    key: [item.model_dump(mode="json") for item in values]
                    for key, values in argument_candidates.items()
                },
                "action_context_candidate_count": len(action_candidates),
                "action_context_candidates": [
                    item.model_dump(mode="json") for item in action_candidates
                ],
                "diagnostics": retrieval_diagnostics,
                "action_diagnostics": action_diagnostics,
                "reference_selection_mode": self.selection_mode.value if self.selection_mode else None,
                "reference_catalog": reference_catalog.model_dump(mode="json") if reference_catalog else None,
            }
        )
        if reference_catalog is None:
            authorization = self.runtime_authorization.run(
                runtime_authorization_request(
                    task_frames,
                    prepared_input.proposed_tool_call,
                    argument_candidates,
                    action_candidates,
                ),
                call_id,
                prepared_input,
                task_frames,
            )
        else:
            authorization = self.runtime_authorization.run(
                selection_request(
                    task_frames,
                    prepared_input.proposed_tool_call,
                    argument_candidates,
                    action_candidates,
                    reference_catalog,
                ),
                call_id,
                prepared_input,
                task_frames,
                reference_catalog,
                lambda result: validate_runtime_authorization(result, prepared_input, task_frames),
            )
        return task_frames, authorization


__all__ = [
    "V6RuntimeAuthorizationStage",
    "V6SemanticPipeline",
    "normalize_runtime_authorization_references",
    "validate_runtime_authorization",
]
