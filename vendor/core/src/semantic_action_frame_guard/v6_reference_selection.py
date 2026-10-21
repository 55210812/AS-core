from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from semantic_action_frame_guard.llm import LLMTransport, StageFailure, TraceRecorder
from semantic_action_frame_guard.v3_models import TaskFrameSet
from semantic_action_frame_guard.v5_models import (
    AuthorizationDecision,
    RuntimeAuthorization,
)
from semantic_action_frame_guard.v6_models import (
    ActionContextCandidate,
    ArgumentSourceCandidate,
    GuardInput,
)
from semantic_action_frame_guard.v6_prompts import (
    RUNTIME_AUTHORIZATION_SYSTEM,
)


class ReferenceSelectionMode(StrEnum):
    ALL = "all"
    TASK_ONLY = "task_only"


class InvalidSelectionError(ValueError):
    """The model selected an ID that is not valid for this request."""


class ReferenceOption:
    def __init__(self, option_id: str, ref: str) -> None:
        self.id = option_id
        self.ref = ref

    def model_dump(self, *, mode: str = "python") -> dict[str, str]:
        del mode
        return {"id": self.id, "ref": self.ref}


class ReferenceSelectionCatalog:
    def __init__(
        self,
        task_reference_options: list[ReferenceOption],
        history_reference_options: list[ReferenceOption],
        mode: ReferenceSelectionMode,
    ) -> None:
        self.task_reference_options = task_reference_options
        self.history_reference_options = history_reference_options
        self.mode = mode
        self.task_by_id = {item.id: item.ref for item in task_reference_options}
        self.history_by_id = {item.id: item.ref for item in history_reference_options}

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {
            "task_reference_options": [item.model_dump() for item in self.task_reference_options],
            "history_reference_options": [item.model_dump() for item in self.history_reference_options],
        }


class SelectionActionSupport:
    def __init__(self, root_id: str, via_ids: list[str], relation: str) -> None:
        self.root_id = root_id
        self.via_ids = via_ids
        self.relation = relation


class SelectionParameterGrounding:
    def __init__(self, source_ids: list[str], derivation: str) -> None:
        self.source_ids = source_ids
        self.derivation = derivation


class SelectionRuntimeAuthorization:
    def __init__(
        self,
        action_support: list[SelectionActionSupport],
        parameter_grounding: dict[str, SelectionParameterGrounding],
        decision: AuthorizationDecision,
    ) -> None:
        self.action_support = action_support
        self.parameter_grounding = parameter_grounding
        self.decision = decision

    @classmethod
    def from_json(cls, raw: str) -> SelectionRuntimeAuthorization:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("selection output must be an object")
        allowed = {"action_support", "parameter_grounding", "decision"}
        if set(value) != allowed:
            raise ValueError("selection output must contain exactly action_support, parameter_grounding, decision")
        supports = value["action_support"]
        if not isinstance(supports, list):
            raise ValueError("action_support must be a list")
        action_support: list[SelectionActionSupport] = []
        for item in supports:
            if not isinstance(item, dict) or set(item) != {"root_id", "via_ids", "relation"}:
                raise ValueError("action_support entries must contain root_id, via_ids, relation")
            if not isinstance(item["root_id"], str) or not isinstance(item["via_ids"], list):
                raise ValueError("selection IDs must have the expected types")
            if not all(isinstance(ref, str) for ref in item["via_ids"]):
                raise ValueError("via_ids must contain strings")
            if not isinstance(item["relation"], str) or not item["relation"].strip():
                raise ValueError("relation must be a non-empty string")
            action_support.append(SelectionActionSupport(item["root_id"], item["via_ids"], item["relation"]))
        grounding_value = value["parameter_grounding"]
        if not isinstance(grounding_value, dict):
            raise ValueError("parameter_grounding must be an object")
        grounding: dict[str, SelectionParameterGrounding] = {}
        for argument, item in grounding_value.items():
            if not isinstance(item, dict) or set(item) != {"source_ids", "derivation"}:
                raise ValueError("parameter_grounding entries must contain source_ids, derivation")
            if not isinstance(item["source_ids"], list) or not all(
                isinstance(ref, str) for ref in item["source_ids"]
            ):
                raise ValueError("source_ids must contain strings")
            if not isinstance(item["derivation"], str) or not item["derivation"].strip():
                raise ValueError("derivation must be a non-empty string")
            grounding[argument] = SelectionParameterGrounding(item["source_ids"], item["derivation"])
        try:
            decision = AuthorizationDecision(value["decision"])
        except (KeyError, ValueError) as exc:
            raise ValueError("decision must be allow, repair, or block") from exc
        return cls(action_support, grounding, decision)


def _register_nested(
    options: list[ReferenceOption],
    value: Any,
    path: str,
    option_id: str,
) -> None:
    options.append(ReferenceOption(option_id, path))
    if isinstance(value, dict):
        for index, (key, child) in enumerate(value.items(), start=1):
            _register_nested(options, child, f"{path}.{key}", f"{option_id}_k{index}")
    elif isinstance(value, list):
        for index, child in enumerate(value, start=1):
            _register_nested(options, child, f"{path}[{index - 1}]", f"{option_id}_i{index}")


def _task_options(task_frames: TaskFrameSet) -> list[ReferenceOption]:
    options: list[ReferenceOption] = []
    fixed_fields = {
        "frame_id": "id",
        "predicate": "p",
        "object": "o",
        "constraints": "c",
        "source": "src",
        "output": "out",
        "sink": "sink",
        "side_effect": "eff",
    }
    for task_index, frame in enumerate(task_frames.root, start=1):
        data = frame.model_dump(mode="json")
        frame_ref = frame.frame_id
        frame_id = f"t{task_index}"
        options.append(ReferenceOption(frame_id, frame_ref))
        for field_name, suffix in fixed_fields.items():
            value = data[field_name]
            field_ref = f"{frame_ref}.{field_name}"
            if field_name == "constraints" and isinstance(value, dict):
                options.append(ReferenceOption(f"{frame_id}_c", field_ref))
                for constraint_index, (key, child) in enumerate(value.items(), start=1):
                    _register_nested(
                        options,
                        child,
                        f"{field_ref}.{key}",
                        f"{frame_id}_c{constraint_index}",
                    )
            elif isinstance(value, (dict, list)):
                _register_nested(options, value, field_ref, f"{frame_id}_{suffix}")
            else:
                options.append(ReferenceOption(f"{frame_id}_{suffix}", field_ref))
    return options


def _history_refs(
    argument_candidates: dict[str, list[ArgumentSourceCandidate]],
    action_candidates: list[ActionContextCandidate],
    task_refs: set[str],
) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for values in argument_candidates.values():
        for candidate in values:
            for provenance in candidate.provenance:
                if provenance.ref in task_refs:
                    continue
                if provenance.ref not in seen:
                    seen.add(provenance.ref)
                    refs.append(provenance.ref)
    for candidate in action_candidates:
        if candidate.ref not in seen:
            seen.add(candidate.ref)
            refs.append(candidate.ref)
    return refs


def build_reference_selection_catalog(
    task_frames: TaskFrameSet,
    argument_candidates: dict[str, list[ArgumentSourceCandidate]],
    action_candidates: list[ActionContextCandidate],
    mode: ReferenceSelectionMode,
) -> ReferenceSelectionCatalog:
    task_options = _task_options(task_frames)
    task_refs = {item.ref for item in task_options}
    history_options: list[ReferenceOption] = []
    if mode == ReferenceSelectionMode.ALL:
        history_options = [ReferenceOption(f"h{index}", ref) for index, ref in enumerate(
            _history_refs(argument_candidates, action_candidates, task_refs), start=1
        )]
    return ReferenceSelectionCatalog(task_options, history_options, mode)


def _candidate_reference_id(ref: str, catalog: ReferenceSelectionCatalog) -> str:
    for option_id, option_ref in catalog.task_by_id.items():
        if option_ref == ref:
            return option_id
    for option_id, option_ref in catalog.history_by_id.items():
        if option_ref == ref:
            return option_id
    if catalog.mode == ReferenceSelectionMode.TASK_ONLY:
        return ref
    raise InvalidSelectionError(f"candidate provenance is absent from the reference catalog: {ref}")


def _candidate_payload(
    argument_candidates: dict[str, list[ArgumentSourceCandidate]],
    action_candidates: list[ActionContextCandidate],
    catalog: ReferenceSelectionCatalog,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    arguments: dict[str, list[dict[str, Any]]] = {}
    for key, values in argument_candidates.items():
        arguments[key] = []
        for candidate in values:
            arguments[key].append({
                "candidate_id": candidate.candidate_id,
                "value": candidate.value,
                "provenance": [
                    {
                        "ref": _candidate_reference_id(item.ref, catalog),
                        "tool_description": item.tool_description,
                        "plan": item.plan,
                    }
                    for item in candidate.provenance
                ],
            })
    actions = [
        {
            "candidate_id": item.candidate_id,
            "ref": _candidate_reference_id(item.ref, catalog),
            "tool_description": item.tool_description,
            "plan": item.plan,
        }
        for item in action_candidates
    ]
    return arguments, actions


def selection_request(
    task_frames: TaskFrameSet,
    proposed_tool_call: Any,
    argument_candidates: dict[str, list[ArgumentSourceCandidate]],
    action_candidates: list[ActionContextCandidate],
    catalog: ReferenceSelectionCatalog,
) -> dict[str, Any]:
    serialized_arguments, serialized_actions = _candidate_payload(
        argument_candidates, action_candidates, catalog
    )
    return {
        "trusted_task_frames": task_frames.model_dump(mode="json"),
        "proposed_tool_call": proposed_tool_call.model_dump(mode="json"),
        "argument_source_candidates": serialized_arguments,
        "action_context_candidates": serialized_actions,
        "reference_selection_mode": catalog.mode.value,
        "task_reference_options": [item.model_dump() for item in catalog.task_reference_options],
        "history_reference_options": [item.model_dump() for item in catalog.history_reference_options],
    }


def _resolve_id(
    value: str,
    mapping: dict[str, str],
    kind: str,
) -> str:
    if value not in mapping:
        raise InvalidSelectionError(f"unknown {kind} selection ID: {value}")
    return mapping[value]


def _unique(values: list[str], field: str) -> None:
    if len(values) != len(set(values)):
        raise InvalidSelectionError(f"duplicate selection ID in {field}")


def resolve_selection(
    selection: SelectionRuntimeAuthorization,
    catalog: ReferenceSelectionCatalog,
) -> RuntimeAuthorization:
    supports = []
    for index, item in enumerate(selection.action_support):
        root = _resolve_id(item.root_id, catalog.task_by_id, "root")
        _unique(item.via_ids, f"action_support[{index}].via_ids")
        if catalog.mode == ReferenceSelectionMode.ALL:
            via = [_resolve_id(value, catalog.history_by_id, "via") for value in item.via_ids]
        else:
            if any(value.startswith("t") for value in item.via_ids):
                raise InvalidSelectionError("task-only via must use real history references")
            via = list(item.via_ids)
        supports.append({"root": root, "via": via, "relation": item.relation})

    grounding: dict[str, Any] = {}
    for argument, item in selection.parameter_grounding.items():
        _unique(item.source_ids, f"parameter_grounding.{argument}.source_ids")
        if catalog.mode == ReferenceSelectionMode.ALL:
            sources = []
            for value in item.source_ids:
                if value.startswith("t"):
                    sources.append(_resolve_id(value, catalog.task_by_id, "source"))
                else:
                    sources.append(_resolve_id(value, catalog.history_by_id, "source"))
        else:
            sources = []
            for value in item.source_ids:
                if value.startswith("t"):
                    sources.append(_resolve_id(value, catalog.task_by_id, "source"))
                else:
                    sources.append(value)
        grounding[argument] = {"sources": sources, "derivation": item.derivation}
    return RuntimeAuthorization.model_validate(
        {
            "action_support": supports,
            "parameter_grounding": grounding,
            "decision": selection.decision.value,
        }
    )


def selection_system_prompt(mode: ReferenceSelectionMode) -> str:
    history_rule = (
        "For via_ids and history sources, choose only h* IDs from history_reference_options."
        if mode == ReferenceSelectionMode.ALL
        else "For via and history sources, copy the real call_*.output reference strings; do not use h* IDs."
    )
    prompt = RUNTIME_AUTHORIZATION_SYSTEM
    prompt = prompt.replace(
        "Each support entry must contain exactly:\n\nroot, via, relation",
        "Each support entry must contain exactly:\n\nroot_id, via_ids, relation",
    )
    prompt = prompt.replace(
        "- root must reference an exact semantic element in trusted_task_frames.",
        "- root_id must select an exact semantic element from task_reference_options.",
    )
    prompt = prompt.replace(
        "- via may reference zero or more candidate provenance refs that instantiate, resolve, or connect the rooted task requirement to the proposed action.",
        "- via_ids may reference zero or more candidate provenance options that instantiate, resolve, or connect the rooted task requirement to the proposed action.",
    )
    prompt = prompt.replace(
        "- `action_support.root` must remain rooted in trusted_task_frames; action candidates can only be cited as the intermediate `via` link.",
        "- `action_support.root_id` must select a task_reference_options entry; action candidates can only be cited through `via_ids`.",
    )
    prompt = prompt.replace(
        "For each task-relevant argument, return exactly:\n\nsources, derivation",
        "For each task-relevant argument, return exactly:\n\nsource_ids, derivation",
    )
    prompt = prompt.replace(
        "- sources reference trusted-task elements or candidate provenance refs used to determine the argument value.",
        "- source_ids select trusted-task or candidate provenance options used to determine the argument value.",
    )
    output_start = prompt.index("OUTPUT FORMAT")
    output_end = prompt.index("Return JSON only.", output_start) + len("Return JSON only.")
    output = '''OUTPUT FORMAT

{
  "action_support": [
    {
      "root_id": "t1_c4",
      "via_ids": ["h1"],
      "relation": "<concise semantic relation>"
    }
  ],
  "parameter_grounding": {
    "<argument_name>": {
      "source_ids": ["t1_c4", "h1"],
      "derivation": "<concise description of how the argument is obtained from the selected sources>"
    }
  },
  "decision": "<allow | repair | block>"
}

Return JSON only.'''
    prompt = prompt[:output_start] + output + prompt[output_end:]
    return prompt + f'''\n\nREFERENCE SELECTION PROTOCOL\n\nThis request uses a program-generated reference catalog. The catalog IDs, not natural-language paths, are the only valid Task Frame references in the authorization output.\n\n- root_id must be one task_reference_options.id.\n- {history_rule}\n- For all-reference mode, source_ids may contain t* IDs or h* IDs.\n- For task-frame-only mode, Task Frame sources use t* IDs and history sources use their real call_*.output strings.\n- Empty support or grounding uses an empty array; do not invent null IDs.\n- The keys root, via, and sources are invalid output keys; use root_id, via_ids, and source_ids.\n- relation and derivation remain concise natural-language explanations.\n- Unknown, duplicated, or cross-type IDs are invalid; do not guess a replacement.\n\nReturn exactly:\n{{\n  "action_support": [{{"root_id": "t1_c4", "via_ids": ["h1"], "relation": "..."}}],\n  "parameter_grounding": {{"argument": {{"source_ids": ["t1_c4", "h1"], "derivation": "..."}}}},\n  "decision": "allow"\n}}\n'''


@dataclass(frozen=True)
class _SelectionCacheEntry:
    result: RuntimeAuthorization
    origin_tool_call_id: str


class V6ReferenceSelectionStage:
    def __init__(
        self,
        system_prompt: str,
        transport: LLMTransport,
        recorder: TraceRecorder,
        mode: ReferenceSelectionMode,
        *,
        cache_namespace: str | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.namespace = "runtime_action_authorization"
        self.system_prompt = system_prompt
        self.transport = transport
        self.recorder = recorder
        self.mode = mode
        self.cache_namespace = cache_namespace or f"v6_reference_selection_{mode.value}_1"
        self.cache_enabled = cache_enabled
        self.cache: dict[str, _SelectionCacheEntry] = {}

    @staticmethod
    def _masked(result: RuntimeAuthorization) -> dict[str, Any]:
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

    def run(
        self,
        payload: dict[str, Any],
        tool_call_id: str,
        guard_input: GuardInput,
        task_frames: TaskFrameSet,
        catalog: ReferenceSelectionCatalog,
        validator: Callable[[RuntimeAuthorization], None],
    ) -> RuntimeAuthorization:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        cache_key = hashlib.sha256(
            f"{self.cache_namespace}\n{self.system_prompt}\n{serialized}".encode()
        ).hexdigest()
        cached = self.cache.get(cache_key)
        if self.cache_enabled and cached is not None:
            validator(cached.result)
            self.recorder.record({
                "stage": self.namespace,
                "tool_call_id": tool_call_id,
                "cache_namespace": self.cache_namespace,
                "cache_key": cache_key,
                "cache_hit": True,
                "origin_tool_call_id": cached.origin_tool_call_id,
                "selection_mode": self.mode.value,
                "reference_catalog": catalog.model_dump(mode="json"),
                "structured_output": cached.result.model_dump(mode="json"),
            })
            return cached.result

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": serialized},
        ]
        first_raw = ""
        last_error = ""
        for attempt in (1, 2):
            started = time.perf_counter()
            try:
                response = self.transport.complete(self.namespace, messages)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.recorder.record({
                    "stage": self.namespace,
                    "tool_call_id": tool_call_id,
                    "cache_namespace": self.cache_namespace,
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "attempt": attempt,
                    "selection_mode": self.mode.value,
                    "prompt": messages,
                    "raw_response": None,
                    "elapsed_seconds": elapsed,
                    "tokens": {"input": 0, "output": 0, "total": 0},
                    "transport_error": f"{type(exc).__name__}: {exc}",
                    "fail_closed": True,
                })
                raise StageFailure(f"{self.namespace} transport request failed: {type(exc).__name__}") from exc
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
                "selection_mode": self.mode.value,
                "reference_catalog": catalog.model_dump(mode="json"),
                "prompt": messages,
                "raw_response": response.raw,
                "elapsed_seconds": elapsed,
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                    "total": response.input_tokens + response.output_tokens,
                },
            }
            try:
                parsed = SelectionRuntimeAuthorization.from_json(response.raw)
                result = resolve_selection(parsed, catalog)
                validator(result)
            except InvalidSelectionError as exc:
                self.recorder.record({**event, "validation_error_kind": "invalid_selection", "validation_error": str(exc), "fail_closed": True})
                raise StageFailure(f"{self.namespace} invalid selection: {exc}") from exc
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self.recorder.record({
                    **event,
                    "validation_error_kind": "schema_or_json",
                    "validation_error": last_error,
                    "model_retry_required": attempt == 1,
                    "fail_closed": attempt == 2,
                })
                if attempt == 2:
                    break
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps({
                        "repair_once": True,
                        "repair_scope": "json_and_schema_only",
                        "original_request": payload,
                        "invalid_json": first_raw,
                        "validation_error": last_error,
                        "requirements": [
                            "Return only the selection schema with root_id, via_ids, and source_ids.",
                            "Copy IDs from the supplied options; do not output real paths.",
                            "Do not change the intended decision or natural-language explanations.",
                        ],
                    }, ensure_ascii=False)},
                ]
                continue
            self.recorder.record({
                **event,
                "selection_output": {
                    "action_support": [
                        {"root_id": item.root_id, "via_ids": item.via_ids, "relation": item.relation}
                        for item in parsed.action_support
                    ],
                    "parameter_grounding": {
                        key: {"source_ids": item.source_ids, "derivation": item.derivation}
                        for key, item in parsed.parameter_grounding.items()
                    },
                    "decision": parsed.decision.value,
                },
                "resolved_output": result.model_dump(mode="json"),
                "model_retry_required": False,
            })
            if self.cache_enabled:
                self.cache[cache_key] = _SelectionCacheEntry(result, tool_call_id)
            return result
        raise StageFailure(f"{self.namespace} failed selection validation after one format repair: {last_error}")


__all__ = [
    "InvalidSelectionError",
    "ReferenceOption",
    "ReferenceSelectionCatalog",
    "ReferenceSelectionMode",
    "SelectionRuntimeAuthorization",
    "V6ReferenceSelectionStage",
    "build_reference_selection_catalog",
    "resolve_selection",
    "selection_request",
    "selection_system_prompt",
]
