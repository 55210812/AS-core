from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from semantic_action_frame_guard.dependency_chain import DependencyValidation, validate_dependency_chain
from semantic_action_frame_guard.models import (
    ChainRefinementJudgment,
    EvidenceLedger,
    GroundingType,
    GuardInput,
    ObservationEvidence,
    RefinementJudgment,
    TaskFrameSet,
    ToolSemanticEffect,
)
from semantic_action_frame_guard.prompts import (
    CHAIN_REFINEMENT_SYSTEM,
    REFINEMENT_SYSTEM,
    TASK_FRAME_SYSTEM,
    REFINEMENT_WITH_LEDGER_SYSTEM,
    TOOL_EFFECT_SYSTEM,
    TOOL_EFFECT_WITH_LEDGER_SYSTEM,
    chain_refinement_request,
    refinement_request,
    task_frame_request,
    tool_effect_request,
)

T = TypeVar("T", bound=BaseModel)
TASK_FRAME_NAMESPACE = "task_frame"
TOOL_EFFECT_NAMESPACE = "tool_effect"
REFINEMENT_NAMESPACE = "refinement"
CHAIN_REFINEMENT_NAMESPACE = "refinement_dependency_chain"
TASK_FRAME_CACHE_NAMESPACE = "task_frame_grounded_evidence_v5"
TOOL_EFFECT_CACHE_NAMESPACE = "tool_effect_grounded_evidence_v5"
TOOL_EFFECT_LEDGER_CACHE_NAMESPACE = "tool_effect_raw_observation_refinement_v2"
REFINEMENT_CACHE_NAMESPACE = "refinement_grounded_evidence_v5"
CHAIN_REFINEMENT_CACHE_NAMESPACE = "refinement_dependency_chain_grounded_evidence_v5"
REFINEMENT_LEDGER_CACHE_NAMESPACE = (
    "refinement_raw_observation_available_tools_v1"
)
SEMANTIC_NAMESPACES = {
    TASK_FRAME_NAMESPACE,
    TOOL_EFFECT_NAMESPACE,
    REFINEMENT_NAMESPACE,
    CHAIN_REFINEMENT_NAMESPACE,
}


@dataclass(frozen=True)
class LLMResponse:
    raw: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_content: str | None = None
    usage: dict[str, Any] | None = None
    response: dict[str, Any] | None = None


class LLMTransport(Protocol):
    def complete(self, namespace: str, messages: list[dict[str, str]]) -> LLMResponse: ...


class OpenAITransport:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, namespace: str, messages: list[dict[str, str]]) -> LLMResponse:
        del namespace
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            request["extra_body"] = {"enable_thinking": self.enable_thinking}
        response = self.client.chat.completions.create(**request)
        usage = response.usage
        message = response.choices[0].message
        usage_payload = None
        if usage is not None:
            usage_payload = (
                usage.model_dump(mode="json")
                if hasattr(usage, "model_dump")
                else {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
            )
        response_payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else None
        return LLMResponse(
            raw=message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            reasoning_content=getattr(message, "reasoning_content", None),
            usage=usage_payload,
            response=response_payload,
        )


class TraceRecorder:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self.events: list[dict[str, Any]] = []
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(value: object) -> str:
        return str(value).replace("/", "_").replace("..", "_")

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        if not self.directory:
            return
        payload = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        call_id = self._safe(event.get("tool_call_id", "unknown"))
        with (self.directory / f"{call_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload)
        with (self.directory / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(payload)


class StageFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class CacheEntry:
    result: BaseModel
    origin_tool_call_id: str


class StructuredStage:
    def __init__(
        self,
        namespace: str,
        system_prompt: str,
        output_model: type[T],
        transport: LLMTransport,
        recorder: TraceRecorder,
        *,
        cache_namespace: str | None = None,
    ) -> None:
        if namespace not in SEMANTIC_NAMESPACES:
            raise ValueError("unknown semantic stage namespace")
        self.namespace = namespace
        self.cache_namespace = cache_namespace or namespace
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.transport = transport
        self.recorder = recorder
        self.cache: dict[str, CacheEntry] = {}

    def run(
        self,
        payload: dict[str, Any],
        tool_call_id: str,
        result_validator: Callable[[T], None] | None = None,
    ) -> T:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        cache_key = hashlib.sha256(serialized.encode()).hexdigest()
        cached = self.cache.get(cache_key)
        if cached is not None:
            result = cached.result
            assert isinstance(result, self.output_model)
            if result_validator is not None:
                result_validator(result)
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

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": serialized},
        ]
        result = self._request_and_validate(
            messages,
            payload,
            tool_call_id,
            cache_key,
            result_validator,
        )
        self.cache[cache_key] = CacheEntry(result=result, origin_tool_call_id=tool_call_id)
        return result

    def _request_and_validate(
        self,
        messages: list[dict[str, str]],
        payload: dict[str, Any],
        tool_call_id: str,
        cache_key: str,
        result_validator: Callable[[T], None] | None,
    ) -> T:
        first_raw = ""
        last_error = ""
        for attempt in (1, 2):
            request_messages = messages
            if attempt == 2:
                request_messages = [
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
                response = self.transport.complete(self.namespace, request_messages)
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
                        "prompt": request_messages,
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
            event: dict[str, Any] = {
                "stage": self.namespace,
                "tool_call_id": tool_call_id,
                "cache_namespace": self.cache_namespace,
                "cache_key": cache_key,
                "cache_hit": False,
                "attempt": attempt,
                "prompt": request_messages,
                "raw_response": response.raw,
                "elapsed_seconds": elapsed,
                "tokens": {
                    "input": response.input_tokens,
                    "output": response.output_tokens,
                    "total": response.input_tokens + response.output_tokens,
                },
            }
            try:
                parsed = self.output_model.model_validate_json(response.raw)
                if result_validator is not None:
                    result_validator(parsed)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                self.recorder.record({**event, "validation_error": last_error, "fail_closed": attempt == 2})
                continue
            self.recorder.record({**event, "structured_output": parsed.model_dump(mode="json")})
            return parsed
        raise StageFailure(f"{self.namespace} failed validation after one repair: {last_error}")


def validate_tool_effect_evidence(
    effect: ToolSemanticEffect,
    guard_input: GuardInput,
    frames: TaskFrameSet,
) -> None:
    ledger = guard_input.evidence_ledger
    if ledger is not None and ledger.task_id != guard_input.task_id:
        raise ValueError("evidence ledger task_id does not match the guard input")
    if ledger is not None:
        if effect.constraints.get("observation_evidence") is not None:
            raise ValueError("ledger mode does not accept observation_evidence")
        if effect.constraints.get("supporting_observation_ids") is not None:
            raise ValueError(
                "ledger mode leaves evidence selection to refinement"
            )
        return

    raw_evidence = effect.constraints.get("observation_evidence")
    if raw_evidence is None:
        return
    observations = {item.observation_id: item for item in guard_input.observations}
    task_references = {frame.source for frame in frames.root} | {
        frame.output for frame in frames.root
    }
    selected_source_references = {
        frame.source
        for frame in frames.root
        if frame.predicate == "perform_actions_described_in_source"
        or "bounded_by_selected_source_actions"
        in json.dumps(
            {"object": frame.object, "sink": frame.sink, "side_effect": frame.side_effect},
            ensure_ascii=False,
            sort_keys=True,
        ).lower()
    }
    derived_objects = {
        frame.output
        for frame in frames.root
        if any(
            token in frame.predicate.lower()
            for token in ("select", "filter", "rank", "choose", "extremum")
        )
        or {
            "criterion",
            "extremum",
            "filter",
            "ranking",
            "fallback",
            "condition",
        }.intersection(frame.constraints)
    }

    evidence_items = [
        ObservationEvidence.model_validate(item) for item in raw_evidence
    ]
    delegated_abstract_objects = {
        str(evidence.grounded_fields["abstract_object"])
        for evidence in evidence_items
        if evidence.grounding_type == GroundingType.SOURCE_INSTRUCTION
        and evidence.grounding_status == "verified"
        and evidence.authorized_source in selected_source_references
        and "abstract_object" in evidence.grounded_fields
    }

    for evidence in evidence_items:
        observation = observations.get(evidence.observation_id)
        if observation is None:
            raise ValueError(
                f"unknown observation_evidence observation_id: {evidence.observation_id}"
            )
        if evidence.excerpt not in observation.content:
            raise ValueError(
                "observation_evidence excerpt is not an exact substring of "
                f"{evidence.observation_id}"
            )
        if (
            evidence.grounding_type == GroundingType.SOURCE_INSTRUCTION
            and evidence.authorized_source not in selected_source_references
        ):
            raise ValueError(
                "source_instruction authorized_source does not match a selected source"
            )
        if evidence.authorized_source and evidence.authorized_source not in task_references:
            raise ValueError(
                "observation_evidence authorized_source is not a task-frame source or output"
            )
        if evidence.dependency_source and (
            evidence.dependency_source not in observations
            and evidence.dependency_source not in task_references
        ):
            raise ValueError("observation_evidence dependency_source is unknown")

        grounded_fields = evidence.grounded_fields
        for field_name in ("predicate", "object", "sink", "side_effect"):
            if (
                field_name in grounded_fields
                and grounded_fields[field_name] != getattr(effect, field_name)
            ):
                raise ValueError(
                    f"grounded {field_name} does not match the raw semantic effect"
                )

        if evidence.grounding_type == GroundingType.ARGUMENT_BINDING:
            argument_binding = grounded_fields.get("argument")
            if not isinstance(argument_binding, dict) or not argument_binding:
                raise ValueError(
                    "argument_binding requires a non-empty grounded_fields.argument object"
                )
            for key, value in argument_binding.items():
                if key not in guard_input.tool.args or guard_input.tool.args[key] != value:
                    raise ValueError(
                        "grounded argument does not exactly match the proposed tool args"
                    )

        if evidence.grounding_type in {
            GroundingType.OBJECT_BINDING,
            GroundingType.BRANCH_RESOLUTION,
        }:
            abstract_object = grounded_fields.get("abstract_object")
            if abstract_object not in derived_objects | delegated_abstract_objects:
                raise ValueError(
                    "derived-object evidence must name a known unresolved abstract_object"
                )



def validate_refinement_derivations(
    judgment: RefinementJudgment,
    ledger: EvidenceLedger,
) -> None:
    known_observation_ids = {
        item.observation_id for item in ledger.observations
    }
    for binding in judgment.derived_bindings:
        for observation_id in binding.supporting_observation_ids:
            if observation_id not in known_observation_ids:
                raise ValueError(
                    "derived binding references an unknown observation_id: "
                    f"{observation_id}"
                )


class SemanticPipeline:
    def __init__(self, transport: LLMTransport, recorder: TraceRecorder | None = None) -> None:
        self.recorder = recorder or TraceRecorder()
        self.dependency_chain_enabled = os.getenv("SAFG_DEPENDENCY_CHAIN") == "1"
        self.evidence_ledger_enabled = os.getenv("SAFG_EVIDENCE_LEDGER") == "1"
        self.task_frame = StructuredStage(
            TASK_FRAME_NAMESPACE,
            TASK_FRAME_SYSTEM,
            TaskFrameSet,
            transport,
            self.recorder,
            cache_namespace=TASK_FRAME_CACHE_NAMESPACE,
        )
        self.tool_effect = StructuredStage(
            TOOL_EFFECT_NAMESPACE,
            TOOL_EFFECT_WITH_LEDGER_SYSTEM if self.evidence_ledger_enabled else TOOL_EFFECT_SYSTEM,
            ToolSemanticEffect,
            transport,
            self.recorder,
            cache_namespace=(
                TOOL_EFFECT_LEDGER_CACHE_NAMESPACE
                if self.evidence_ledger_enabled
                else TOOL_EFFECT_CACHE_NAMESPACE
            ),
        )
        if self.dependency_chain_enabled:
            self.refinement = StructuredStage(
                CHAIN_REFINEMENT_NAMESPACE,
                CHAIN_REFINEMENT_SYSTEM,
                ChainRefinementJudgment,
                transport,
                self.recorder,
                cache_namespace=CHAIN_REFINEMENT_CACHE_NAMESPACE,
            )
        else:
            self.refinement = StructuredStage(
                REFINEMENT_NAMESPACE,
                REFINEMENT_WITH_LEDGER_SYSTEM if self.evidence_ledger_enabled else REFINEMENT_SYSTEM,
                RefinementJudgment,
                transport,
                self.recorder,
                cache_namespace=(
                    REFINEMENT_LEDGER_CACHE_NAMESPACE
                    if self.evidence_ledger_enabled
                    else REFINEMENT_CACHE_NAMESPACE
                ),
            )

    def analyze(
        self,
        guard_input: GuardInput,
    ) -> tuple[
        TaskFrameSet,
        ToolSemanticEffect,
        RefinementJudgment,
        DependencyValidation | None,
    ]:
        frames = self.task_frame.run(
            task_frame_request(guard_input.task_id, guard_input.user_task), guard_input.tool_call_id
        )
        evidence_ledger = None
        if self.evidence_ledger_enabled:
            evidence_ledger = guard_input.evidence_ledger or EvidenceLedger(task_id=guard_input.task_id)

        tool_call_payload = {
            "tool_call_id": guard_input.tool_call_id,
            "tool_name": guard_input.tool.name,
            "tool_description": guard_input.tool.description,
            "tool_json_schema": guard_input.tool.json_schema,
            "tool_args": guard_input.tool.args,
        }
        if evidence_ledger is None:
            tool_call_payload["trajectory_observations"] = [
                item.model_dump(mode="json") for item in guard_input.observations
            ]
        validation_input = (
            guard_input.model_copy(update={"evidence_ledger": evidence_ledger})
            if evidence_ledger is not None and guard_input.evidence_ledger is None
            else guard_input
        )
        effect = self.tool_effect.run(
            tool_effect_request(
                tool_call_payload,
                frames,
                raw_effect_only=evidence_ledger is not None,
            ),
            guard_input.tool_call_id,
            lambda result: validate_tool_effect_evidence(result, validation_input, frames),
        )
        if effect.frame_id != guard_input.tool_call_id:
            raise StageFailure("tool semantic effect returned a different frame_id")

        if evidence_ledger is not None:
            self.recorder.record(
                {
                    "stage": "evidence_ledger_delivery",
                    "tool_call_id": guard_input.tool_call_id,
                    "cache_namespace": REFINEMENT_LEDGER_CACHE_NAMESPACE,
                    "raw_observation_ledger": evidence_ledger.model_dump(
                        mode="json"
                    ),
                    "available_tools": (
                        [
                            item.model_dump(mode="json")
                            for item in guard_input.available_tools
                        ]
                        if guard_input.available_tools is not None
                        else None
                    ),
                }
            )

        dependency_validation = None
        if self.dependency_chain_enabled:
            chain_refinement = self.refinement.run(
                chain_refinement_request(frames, effect, guard_input),
                guard_input.tool_call_id,
            )
            if not isinstance(chain_refinement, ChainRefinementJudgment):
                raise StageFailure("chain refinement returned the wrong structured type")
            refinement = RefinementJudgment.model_validate(
                chain_refinement.model_dump(exclude={"dependency_links"})
            )
        else:
            refinement = self.refinement.run(
                refinement_request(
                    frames,
                    effect,
                    evidence_ledger,
                    (
                        guard_input.available_tools
                        if evidence_ledger is not None
                        else None
                    ),
                ),
                guard_input.tool_call_id,
                (
                    lambda result: validate_refinement_derivations(
                        result, evidence_ledger
                    )
                    if evidence_ledger is not None
                    else None
                ),
            )
            if not isinstance(refinement, RefinementJudgment):
                raise StageFailure("refinement returned the wrong structured type")

        if refinement.refined_frame_id is not None:
            known = {frame.frame_id for frame in frames.root}
            if refinement.refined_frame_id not in known:
                raise StageFailure("refinement references an unknown task frame")

        if self.dependency_chain_enabled:
            dependency_validation = validate_dependency_chain(
                chain_refinement, frames, guard_input
            )
            self.recorder.record(
                {
                    "stage": "dependency_chain_validation",
                    "tool_call_id": guard_input.tool_call_id,
                    "cache_namespace": CHAIN_REFINEMENT_CACHE_NAMESPACE,
                    "dependency_chain_enabled": True,
                    "raw_links": [
                        link.model_dump(mode="json")
                        for link in chain_refinement.dependency_links
                    ],
                    **dependency_validation.trace_fields(),
                }
            )
        return frames, effect, refinement, dependency_validation
